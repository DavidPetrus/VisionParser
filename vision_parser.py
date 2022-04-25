import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2

from utils import k_means

from absl import flags

FLAGS = flags.FLAGS


class VisionParser(nn.Module):

    def __init__(self):
        super(VisionParser, self).__init__()

        self.net = timm.create_model('resnest26d', features_only=True, pretrained=False, out_indices=(2,))


    def cluster_batch(self, main_features, crop_features, crop_dims, K):
        embds = main_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)
        cl_idxs, centroids, ce, num_points = k_means(embds, K) # (N,),(K,D),(K,),(K,)

        cl_idxs = cl_idxs.reshape(main_features.shape[0],main_features.shape[2],main_features.shape[3])
        crop_idxs = cl_idxs[:,round(crop_dims[1]*main_features.shape[2]):round((crop_dims[1]+crop_dims[2])*main_features.shape[2]), \
                              round(crop_dims[0]*main_features.shape[3]):round((crop_dims[0]+crop_dims[2])*main_features.shape[3])]
        
        crop_idxs = F.interpolate(crop_idxs.unsqueeze(1), size=(crop_features.shape[2],crop_features.shape[3]),mode='nearest',align_corners=False) # (B,1,H,W)
        crop_idxs = crop_idxs.reshape(-1)
        centroid_map = torch.index_select(centroids, 0, crop_idxs)

        crop_features = F.normalize(crop_features,dim=1).movedim(1,3).reshape(-1, FLAGS.embd_dim)
        sims = (crop_features * centroid_map).sum(dim=1) # (N,)

        return sims


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
