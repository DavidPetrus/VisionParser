import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
#import matplotlib.pyplot as plt
import time
from sklearn.cluster import AgglomerativeClustering

from absl import flags

FLAGS = flags.FLAGS

IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)

color = np.random.randint(0,256,[5120,3],dtype=np.uint8)

def resize_image(image, max_patch_size):
    height, width, _ = image.shape
    new_height = min(512, height - height%max_patch_size)
    new_width = min(512, width - width%max_patch_size)
    lu = (np.random.randint(0,max(1,image.shape[0]-new_height)), np.random.randint(0,max(1,image.shape[1]-new_width)))

    return image[lu[0]:lu[0]+new_height, lu[1]:lu[1]+new_width]

def normalize_image(image):
    image = image.float()
    image /= 255.

    return image


def random_crop_resize(image, crop_dims=None):
    c, img_h, img_w = image.shape

    if crop_dims is None:
        crop_size = np.random.uniform(FLAGS.min_crop,min(img_h/img_w,img_w/img_h))
        crop_size = int(max(img_h,img_w)*crop_size)
        crop_x,crop_y = np.random.randint(0,img_w-crop_size), np.random.randint(0,img_h-crop_size)
    else:
        crop_size = int(max(img_h,img_w)*crop_dims[2])
        crop_x, crop_y = int(crop_dims[0]*img_w), int(crop_dims[1]*img_h)

    crop = image[:,crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    #resize_frac = np.random.uniform(FLAGS.min_resize,FLAGS.max_resize,size=(1,))[0]
    #t_size = (int(c_h*resize_frac),int(c_w*resize_frac))
    #resized = F.interpolate(crop.unsqueeze(0),size=(t_size[0]*8,t_size[1]*8),mode='bilinear',align_corners=True)

    return crop

def color_distortion(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2):
    color_jitter = torchvision.transforms.ColorJitter(brightness,contrast,saturation,hue)
    return color_jitter


def sinkhorn_knopp(sims):
    Q = torch.exp(sims / FLAGS.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sum to 1
    sum_Q = torch.sum(Q)
    Q = Q/sum_Q

    for it in range(FLAGS.sinkhorn_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q = Q/sum_of_rows
        Q = Q/K

        # normalize each column: total weight per sample must be 1/B
        Q = Q/torch.sum(Q, dim=0, keepdim=True)
        Q = Q/B

    if FLAGS.round_q:
        max_proto_sim,_ = Q.max(dim=0)
        Q[Q != max_proto_sim] = 0.
        Q[Q == max_proto_sim] = 1.
    else:
        Q = Q*B # the columns must sum to 1 so that Q is an assignment

    return Q.t()



def find_clusters(log_dict, level_embds, prototypes):
    l_ix = 0
    embd_tensor = level_embds
    embds = embd_tensor.detach().movedim(1,3)
    _,l_h,l_w,_ = embds.shape
    embds = embds.reshape(l_h*l_w,-1)
    embds = F.normalize(embds,dim=1)
    sims = prototypes(embds)

    clust_sims,clusters = sims.max(dim=1)

    _,clust_counts = torch.unique(clusters, return_counts=True)
    sorted_counts,_ = torch.sort(clust_counts,descending=True)
    total_points = sorted_counts.sum()

    log_dict['n_clusters/mean_sim'] = clust_sims.mean()
    log_dict['n_clusters/std_sim'] = clust_sims.std()
    log_dict['n_clusters/num_clusts'] = sorted_counts.shape[0]
    log_dict['n_clusters/min_freq'] = sorted_counts[-1]/total_points

    if sorted_counts.shape[0] >= 3:
        log_dict['n_clusters/1_freq'] = sorted_counts[0]/total_points
        log_dict['n_clusters/2_freq'] = sorted_counts[1]/total_points
        log_dict['n_clusters/3_freq'] = sorted_counts[2]/total_points

    return log_dict

def plot_embeddings(level_embds,prototypes):
    global color

    resize = [8,8,8,8,16]
    segs = []

    for l_ix, embd_tensor in enumerate(level_embds):
        embds = embd_tensor.detach().movedim(1,3)
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)

        embds = F.normalize(embds,dim=1)
        sims = prototypes(embds)
        clusters = sims.argmax(dim=1)
        clusters = clusters.reshape(l_h,l_w).cpu().numpy()

        seg = np.zeros([clusters.shape[0],clusters.shape[1],3],dtype=np.uint8)
        for c in range(clusters.max()+1):
            seg[clusters==c] = color[c]
        seg = cv2.resize(seg, (seg.shape[1]*resize[l_ix],seg.shape[0]*resize[l_ix]))
        segs.append(seg)

    return segs