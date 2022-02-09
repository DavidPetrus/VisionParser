import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from utils import display_reconst_img, plot_embeddings, sinkhorn_knopp

import warnings

from absl import flags

FLAGS = flags.FLAGS


class VisionParser(nn.Module):

    def __init__(self):
        super(VisionParser, self).__init__()


    def build_model(self):

        self.seg_cnn = self.cnn()

        self.prototypes = nn.Linear(128, FLAGS.num_prototypes, bias=False)
        self.normalize_prototypes()

    def normalize_prototypes(self):
        #with torch.set_grad_enabled(not FLAGS.sg_cluster_assign):
        with torch.set_grad_enabled(False):
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

    def cnn(self):
        cnn_channels = [3,32,32,32,32,64,64,64,64,128,128,128,128]
        strides = [1,2,1,1,2,1,1,1,2,1,1,1,1]
        skips = [False,]
        cnn_layers = []
        cnn_layers.append(nn.Sequential(nn.Conv2d(cnn_channels[0],cnn_channels[1],kernel_size=5,stride=1,padding=2), \
                                        nn.InstanceNorm2d(cnn_channels[1], affine=True),nn.Hardswish(inplace=True)))
        for l in range(1,len(cnn_channels)-1):
            cnn_layers.append(nn.Sequential(nn.Conv2d(cnn_channels[l],cnn_channels[l+1],kernel_size=3,stride=strides[l],padding=1), \
                                            nn.InstanceNorm2d(cnn_channels[l+1], affine=True),nn.Hardswish(inplace=True)))

        cnn_layers.append(nn.Conv2d(cnn_channels[-1],cnn_channels[-1],kernel_size=3,stride=1,padding=1))

        return nn.ModuleList(cnn_layers)

    def cnn_forward(self, x):
        skips = [1,4,8,12]
        skip_connect = None
        for l_ix,l in enumerate(self.seg_cnn):
            if l_ix in skips and skip_connect is not None:
                x = x+skip_connect

            x = l(x)
            if l_ix in skips:
                skip_connect = x

        return x

    def swav_loss(self, full_img_embds, crop_img_embds):
        # full_img_embds (B,128)
        # crop_img_embds (B,128)
        B,c = full_img_embds.shape
        full_img_embds = F.normalize(full_img_embds,dim=1)
        crop_img_embds = F.normalize(crop_img_embds,dim=1)

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
