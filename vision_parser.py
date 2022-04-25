import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2

from utils import sinkhorn_knopp

from absl import flags

FLAGS = flags.FLAGS


class VisionParser(nn.Module):

    def __init__(self):
        super(VisionParser, self).__init__()

        self.net = timm.create_model('resnest26d', features_only=True, pretrained=False, out_indices=(2,))

        #self.prototypes = nn.Linear(FLAGS.embd_dim, FLAGS.num_prototypes, bias=False)
        #self.initialize_and_normalize_prototypes()
        self.prototypes = torch.randn(1, FLAGS.num_prototypes, FLAGS.embd_dim).to('cuda')

    def initialize_and_normalize_prototypes(self):
        #with torch.set_grad_enabled(not FLAGS.sg_cluster_assign):
        with torch.set_grad_enabled(False):
            w = torch.randn(FLAGS.num_prototypes, FLAGS.embd_dim)
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

    def swav_loss(self, clust_features, sim_features, valid_clusters):
        if clust_features.shape[0] < FLAGS.min_valid_clusts:
            return 0.

        clust_features = F.normalize(clust_features, dim=1)
        sim_features = F.normalize(sim_features, dim=1)

        clust_sims = self.prototypes(clust_features)[:,valid_clusters]
        feat_sims = self.prototypes(sim_features)[:,valid_clusters]

        with torch.set_grad_enabled(not FLAGS.sg_cluster_assign):
            q = sinkhorn_knopp(clust_sims)

        loss = -(q * F.log_softmax(feat_sims/FLAGS.cl_temp,dim=1)).sum(dim=1).mean()

        return loss

    def assign_nearest_clusters(self, feature_maps):
        bs,c,h,w = feature_maps.shape
        features_flat = feature_maps.movedim(1,3).reshape(-1,1,FLAGS.embd_dim)

        #sims = F.softmax(((features_flat - self.prototypes)**2).sum(dim=1)/FLAGS.assign_temp, dim=1) # B*h*w,num_prototypes
        sims = F.softmax(-torch.cdist(features_flat,self.prototypes).squeeze(1)/FLAGS.assign_temp, dim=1) # num_embds, num_prototypes
        
        sims = sims.reshape(bs, h, w, FLAGS.num_prototypes).movedim(3,1) # B,num_prototypes,img_size/8,img_size/8

        nearest_cluster = sims.argmax(dim=1) # B,img_size/8,img_size/8
        mask = F.one_hot(nearest_cluster, FLAGS.num_prototypes).movedim(3,1) # B,num_prototypes,img_size/8,img_size/8
        mask = (mask.sum((2,3)) > 3).sum()

        return sims, mask/FLAGS.batch_size

    def spatial_map_cluster_assign(self, feature_maps, masks, crop_dims):
        overlap_dims = (max(crop_dims[0][0],crop_dims[1][0]), max(crop_dims[0][1],crop_dims[1][1]), \
                        min(crop_dims[0][0]+crop_dims[0][2],crop_dims[1][0]+crop_dims[1][2]), min(crop_dims[0][1]+crop_dims[0][2],crop_dims[1][1]+crop_dims[1][2]))

        fm_a_crop = feature_maps[0][:,:, \
                    round((overlap_dims[1]-crop_dims[0][1])*feature_maps[0].shape[2]/crop_dims[0][2]): \
                    round((overlap_dims[3]-crop_dims[0][1])*feature_maps[0].shape[2]/crop_dims[0][2]), \
                    round((overlap_dims[0]-crop_dims[0][0])*feature_maps[0].shape[3]/crop_dims[0][2]): \
                    round((overlap_dims[2]-crop_dims[0][0])*feature_maps[0].shape[3]/crop_dims[0][2])] # B,c,fm_a_h,fm_a_w

        fm_b_crop = feature_maps[1][:,:, \
                    round((overlap_dims[1]-crop_dims[1][1])*feature_maps[1].shape[2]/crop_dims[1][2]): \
                    round((overlap_dims[3]-crop_dims[1][1])*feature_maps[1].shape[2]/crop_dims[1][2]), \
                    round((overlap_dims[0]-crop_dims[1][0])*feature_maps[1].shape[3]/crop_dims[1][2]): \
                    round((overlap_dims[2]-crop_dims[1][0])*feature_maps[1].shape[3]/crop_dims[1][2])] # B,c,fm_b_h,fm_b_w

        cropped_mask = masks[:,:,round(overlap_dims[1]*masks.shape[2]):round(overlap_dims[3]*masks.shape[2]), \
                                 round(overlap_dims[0]*masks.shape[3]):round(overlap_dims[2]*masks.shape[3])]
        
        resized_mask_a = F.interpolate(cropped_mask, size=(fm_a_crop.shape[2],fm_a_crop.shape[3]),mode='bilinear',align_corners=False)
        #if FLAGS.assign_thresh > 0.:
        #    resized_mask_a = torch.where(resized_mask_a > FLAGS.assign_thresh, 1., 0.) # B,num_protos,fm_a_h,fm_a_w

        resized_mask_b = F.interpolate(cropped_mask, size=(fm_b_crop.shape[2],fm_b_crop.shape[3]),mode='bilinear',align_corners=False)
        #if FLAGS.assign_thresh > 0.:
        #    resized_mask_b = torch.where(resized_mask_b > FLAGS.assign_thresh, 1., 0.) # B,num_protos,fm_b_h,fm_b_w

        mask_a_count = resized_mask_a.sum(dim=(2,3)) # B,num_protos
        mask_b_count = resized_mask_b.sum(dim=(2,3)) # B,num_protos


        '''num_valid_embds = (mask_a_count >= FLAGS.min_per_img_embds).sum(0)
        valid_protos_a = num_valid_embds >= FLAGS.num_embds_per_cluster # num_protos
        num_samples_a = num_valid_embds[valid_protos_a].min()
        proto_features_a = fm_a_crop.unsqueeze(1) * resized_mask_a[:,valid_protos_a].unsqueeze(2) # B,num_valid_a,c,fm_a_h,fm_a_w
        proto_features_a = proto_features_a.sum(dim=(3,4)) / (mask_a_count[:,valid_protos_a].unsqueeze(2) + 1e-6) # B,num_valid_a,c
        sampled_indices_a = torch.multinomial((mask_a_count >= FLAGS.min_per_img_embds)[:,valid_protos_a].movedim(0,1).float(), \
                                              num_samples_a).unsqueeze(2).tile(1,1,FLAGS.embd_dim) # num_valid_a,num_embds_per_clust,c
        features_clust_a = torch.gather(proto_features_a.movedim(0,1),1,sampled_indices_a) # num_valid_a,num_embds_per_clust,c
        
        sim_features_b = fm_b_crop.unsqueeze(1) * resized_mask_b[:,valid_protos_a].unsqueeze(2) # B,num_valid_a,c,fm_a_h,fm_a_w
        sim_features_b = sim_features_b.sum(dim=(3,4)) / (mask_b_count[:,valid_protos_a].unsqueeze(2) + 1e-6) # B,num_valid_a,c
        features_sim_b = torch.gather(sim_features_b.movedim(0,1),1,sampled_indices_a) # num_valid_a,num_embds_per_clust,c

        
        num_valid_embds = (mask_b_count >= FLAGS.min_per_img_embds).sum(0)
        valid_protos_b = num_valid_embds >= FLAGS.num_embds_per_cluster # num_protos
        num_samples_b = num_valid_embds[valid_protos_b].min()
        proto_features_b = fm_b_crop.unsqueeze(1) * resized_mask_b[:,valid_protos_b].unsqueeze(2) # B,num_valid_b,c,fm_b_h,fm_b_w
        proto_features_b = proto_features_b.sum(dim=(3,4)) / (mask_b_count[:,valid_protos_b].unsqueeze(2) + 1e-6) # B,num_valid_b,c
        sampled_indices_b = torch.multinomial((mask_b_count >= FLAGS.min_per_img_embds)[:,valid_protos_b].movedim(0,1).float(), \
                                              num_samples_b).unsqueeze(2).tile(1,1,FLAGS.embd_dim) # num_valid_b,num_embds_per_clust,c
        features_clust_b = torch.gather(proto_features_b.movedim(0,1),1,sampled_indices_b) # num_valid_b,num_embds_per_clust,c
        
        sim_features_a = fm_a_crop.unsqueeze(1) * resized_mask_a[:,valid_protos_b].unsqueeze(2) # B,num_valid_b,c,fm_a_h,fm_a_w
        sim_features_a = sim_features_a.sum(dim=(3,4)) / (mask_a_count[:,valid_protos_b].unsqueeze(2) + 1e-6) # B,num_valid_b,c
        features_sim_a = torch.gather(sim_features_a.movedim(0,1),1,sampled_indices_b) # num_valid_b,num_embds_per_clust,c


        return features_clust_a.reshape(-1,FLAGS.embd_dim), features_clust_b.reshape(-1,FLAGS.embd_dim), \
               features_sim_a.reshape(-1,FLAGS.embd_dim), features_sim_b.reshape(-1,FLAGS.embd_dim), \
               valid_protos_a, valid_protos_b, num_samples_a'''

        return fm_a_crop, fm_b_crop, resized_mask_a, resized_mask_b

    def loss_calc(self, features, codes):
        features = features.movedim(1,3).reshape(-1,1,FLAGS.embd_dim)

        codes = codes.movedim(1,3).reshape(-1,FLAGS.num_prototypes) # num_embds, num_prototypes

        #clust_sims = ((features - self.prototypes)**2).sum(dim=1) # num_embds, num_prototypes
        clust_sims = -torch.cdist(features,self.prototypes).squeeze(1) # num_embds, num_prototypes

        clust_max = clust_sims.max(dim=1)[0].mean()

        loss = -(codes * F.log_softmax(clust_sims/FLAGS.crop_temp,dim=1)).sum(dim=1).mean()

        return loss, clust_max


    def extract_feature_map(self, images):
        return self.net(images)[0]
