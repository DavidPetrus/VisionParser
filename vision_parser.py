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


    def mask_crop_feats(self, cr_features, cr_dims, cl_idxs):
        cl_idxs = cl_idxs.reshape(cr_features.shape[0],cr_features.shape[2],cr_features.shape[3])

        crop_idxs = cl_idxs[:,round(cr_dims[1]*cr_features.shape[2]):round((cr_dims[1]+cr_dims[2])*cr_features.shape[2]), \
                              round(cr_dims[0]*cr_features.shape[3]):round((cr_dims[0]+cr_dims[2])*cr_features.shape[3])]
        
        crop_idxs = F.interpolate(crop_idxs.unsqueeze(1).float(), size=(cr_features.shape[2],cr_features.shape[3]),mode='nearest') # (B,1,H,W)
        crop_idxs = crop_idxs.reshape(-1).long() # (N,)

        norm_features = cr_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)
        norm_features = norm_features[crop_idxs != -1]
        crop_idxs = crop_idxs[crop_idxs != -1]

        return norm_features, crop_idxs


    def extract_feature_map(self, images):
        return self.proj_head(self.net(images)[0])
