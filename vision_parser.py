import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2

from absl import flags

FLAGS = flags.FLAGS


class VisionParser(nn.Module):

    def __init__(self):
        super(VisionParser, self).__init__()

        self.net = timm.create_model('resnest26d', features_only=True, pretrained=False, out_indices=(2,))
        self.proj_head = nn.Conv2d(512,FLAGS.embd_dim,1,bias=False)


    def get_sims(self, crop_features, crop_dims, centroids, cl_idxs):
        
        sims = []
        pos_idxs = []
        for cr_dims,cr_features in zip(crop_dims,crop_features):
            cl_idxs = cl_idxs.reshape(cr_features.shape[0],cr_features.shape[2],cr_features.shape[3])

            crop_idxs = cl_idxs[:,round(cr_dims[1]*cr_features.shape[2]):round((cr_dims[1]+cr_dims[2])*cr_features.shape[2]), \
                                  round(cr_dims[0]*cr_features.shape[3]):round((cr_dims[0]+cr_dims[2])*cr_features.shape[3])]
            
            crop_idxs = F.interpolate(crop_idxs.unsqueeze(1).float(), size=(cr_features.shape[2],cr_features.shape[3]),mode='nearest') # (B,1,H,W)
            crop_idxs = crop_idxs.reshape(-1).long() # (N,)

            #centroid_map = torch.index_select(centroids, 0, crop_idxs)
            norm_features = F.normalize(cr_features,dim=1).movedim(1,3).reshape(-1, FLAGS.embd_dim)
            #sims.append((norm_features * centroid_map).sum(dim=1)) # (N,)

            sims.append(norm_features @ centroids.T) # (N,K)
            pos_idxs.append(crop_idxs)

        return sims, pos_idxs


    def extract_feature_map(self, images):
        return self.proj_head(self.net(images)[0])
