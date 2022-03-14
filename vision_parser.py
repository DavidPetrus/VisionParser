import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2

from utils import display_reconst_img, plot_embeddings, sinkhorn_knopp

import warnings

from absl import flags

FLAGS = flags.FLAGS


class VisionParser(nn.Module):

    def __init__(self):
        super(VisionParser, self).__init__()

        self.model = timm.create_model('resnest26d',features_only=True,pretrained=True,out_indices=(2,))

        self.prototypes = nn.Linear(512, FLAGS.num_prototypes, bias=False)
        self.normalize_prototypes()

    def normalize_prototypes(self):
        #with torch.set_grad_enabled(not FLAGS.sg_cluster_assign):
        with torch.set_grad_enabled(False):
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

    def swav_loss(self, full_img_embds, crop_img_embds):
        # full_img_embds (B,512)
        # crop_img_embds (B,512)
        B,c = full_img_embds.shape
        full_img_embds = F.normalize(full_img_embds, dim=1)
        crop_img_embds = F.normalize(crop_img_embds, dim=1)

        full_sims = self.prototypes(full_img_embds)
        crop_sims = self.prototypes(crop_img_embds)

        # Calculate codes
        if FLAGS.single_code_assign:
            all_sims = torch.cat([full_sims,crop_sims])
            with torch.set_grad_enabled(not FLAGS.sg_cluster_assign):
                q = sinkhorn_knopp(all_sims)
            loss = -(q[B:] * F.log_softmax(full_sims/FLAGS.cl_temp,dim=1)).sum(dim=1).mean() - \
                   (q[:B] * F.log_softmax(crop_sims/FLAGS.cl_temp,dim=1)).sum(dim=1).mean()
        else:
            with torch.set_grad_enabled(not FLAGS.sg_cluster_assign):
                q_full = sinkhorn_knopp(full_sims)
                q_crops = sinkhorn_knopp(crop_sims)
            loss = -(q_full * F.log_softmax(crop_sims/FLAGS.cl_temp,dim=1)).sum(dim=1).mean() - \
                   (q_crops * F.log_softmax(full_sims/FLAGS.cl_temp,dim=1)).sum(dim=1).mean()

        return loss

    def assign_nearest_clusters(self, feature_maps):
        feature_maps = F.normalize(feature_maps, dim=1) # B,512,img_size/8,img_size/8
        sims = self.prototypes(feature_maps) # B,num_prototypes,img_size/8,img_size/8

        nearest_cluster = sims.argmax(dim=1) # B,img_size/8,img_size/8
        mask = F.one_hot(nearest_cluster, FLAGS.num_prototypes).movedim(3,1) # B,num_prototypes,img_size/8,img_size/8

        return mask

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
        
        resized_mask_a = F.interpolate(cropped_mask, size=(fm_a_crop.shape[2],fm_a_crop.shape[3]),mode='bilinear',align_corners=True)
        resized_mask_a = torch.where(resized_mask_a > FLAGS.assign_thresh, 1., 0.) # B,num_protos,fm_a_h,fm_a_w

        resized_mask_b = F.interpolate(cropped_mask, size=(fm_b_crop.shape[2],fm_b_crop.shape[3]),mode='bilinear',align_corners=True)
        resized_mask_b = torch.where(resized_mask_b > FLAGS.assign_thresh, 1., 0.) # B,num_protos,fm_b_h,fm_b_w

        mask_a_count = resized_mask_a.sum(dim=(2,3)) # B,num_protos
        valid_protos_a = (mask_a_count >= FLAGS.min_per_img_embds).sum(0) >= FLAGS.num_embds_per_cluster # num_protos
        proto_features_a = fm_a_crop.unsqueeze(1) * resized_mask_a[:,valid_protos_a].unsqueeze(2) # B,num_valid_a,c,fm_a_h,fm_a_w
        proto_features_a = proto_features_a.sum(dim=(3,4)) / (mask_a_count.unsqueeze(2) + 0.001) # B,num_valid_a,c
        sampled_indices_a = torch.multinomial((mask_a_count >= FLAGS.min_per_img_embds)[:,valid_protos_a].movedim(0,1), FLAGS.num_embds_per_cluster) # num_valid,num_embds_per_clust
        features_clust_a = torch.gather(proto_features_a.movedim(0,1),1,sampled_indices_a) # num_valid_a,num_embds_per_clust
        sim_features_b = fm_b_crop.unsqueeze(1) * resized_mask_b[:,valid_protos_a].unsqueeze(2) # B,num_valid_a,c,fm_a_h,fm_a_w
        sim_features_b = sim_features_b.sum(dim=(3,4)) / (mask_a_count.unsqueeze(2) + 0.001) # B,num_valid_a,c
        features_sim_b = torch.gather(sim_features_b.movedim(0,1),1,sampled_indices_a) # num_valid_a,num_embds_per_clust


        mask_b_count = resized_mask_b.sum(dim=(2,3)) # B,num_protos
        valid_protos_b = (mask_b_count >= FLAGS.min_per_img_embds).sum(0) >= FLAGS.num_embds_per_cluster # num_protos
        proto_features_b = fm_b_crop.unsqueeze(1) * resized_mask_b[:,valid_protos_b].unsqueeze(2) # B,num_valid_b,c,fm_b_h,fm_b_w
        proto_features_b = proto_features_b.sum(dim=(3,4)) / (mask_b_count.unsqueeze(2) + 0.001) # B,num_valid_b,c
        sampled_indices_b = torch.multinomial((mask_b_count >= FLAGS.min_per_img_embds)[:,valid_protos_b].movedim(0,1), FLAGS.num_embds_per_cluster) # num_valid,num_embds_per_clust
        features_clust_b = torch.gather(proto_features_b.movedim(0,1),1,sampled_indices_b) # num_valid_a,num_embds_per_clust
        sim_features_a = fm_a_crop.unsqueeze(1) * resized_mask_a[:,valid_protos_b].unsqueeze(2) # B,num_valid_a,c,fm_a_h,fm_a_w
        sim_features_a = sim_features_a.sum(dim=(3,4)) / (mask_b_count.unsqueeze(2) + 0.001) # B,num_valid_a,c
        features_sim_a = torch.gather(sim_features_a.movedim(0,1),1,sampled_indices_b) # num_valid_a,num_embds_per_clust


        return features_clust_a, features_clust_b, features_sim_a, features_sim_b

    

    def extract_feature_map(self, images):
        return self.model(images)[0]
