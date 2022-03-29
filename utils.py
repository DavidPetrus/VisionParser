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

    return image/255.


def random_crop(image, crop_dims=None):
    c, img_h, img_w = image.shape

    if crop_dims is None:
        crop_size = np.random.uniform(FLAGS.min_crop,min(img_h/img_w,img_w/img_h))
        crop_size = int(min(img_h,img_w)*crop_size)
        crop_x,crop_y = np.random.randint(0,img_w-crop_size), np.random.randint(0,img_h-crop_size)
    else:
        crop_size = int(max(img_h,img_w)*crop_dims[2])
        crop_x, crop_y = int(crop_dims[0]*img_w), int(crop_dims[1]*img_h)

    crop = image[:,crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    #resize_frac = np.random.uniform(FLAGS.min_resize,FLAGS.max_resize,size=(1,))[0]
    #t_size = (int(c_h*resize_frac),int(c_w*resize_frac))
    #resized = F.interpolate(crop.unsqueeze(0),size=(t_size[0]*8,t_size[1]*8),mode='bilinear',align_corners=True)

    return crop, [crop_x,crop_y,crop_size]


def color_distortion(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.15):
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

def calculate_iou(cluster_mask, annots):
    # cluster_mask B,num_proto,h/8,w/8
    # annots B,460,h,w

    annots = F.interpolate(annots.float(), size=(int(FLAGS.image_size/8),int(FLAGS.image_size/8)), mode='nearest').bool()
    cluster_mask = cluster_mask.bool()
    annotated_cats = annots.sum((0,2,3)) > 0
    max_clusters =  cluster_mask.sum((0,2,3)) > 0
    
    intersection = (cluster_mask[:8,max_clusters].unsqueeze(1) & annots[:8, annotated_cats].unsqueeze(2)).sum((3,4)) # B,460,num_proto
    union = (cluster_mask[:8,max_clusters].unsqueeze(1) | annots[:8, annotated_cats].unsqueeze(2)).sum((3,4)) # B,460,num_proto

    iou = (intersection / (union + 1e-6))
    iou,_ = iou.max(1) # B,num_protos

    mean_iou = iou[iou > 0.02].mean()
    num_ious = (iou > 0.02).sum()

    if mean_iou.isnan():
        return 0.,0.
    else:
        return mean_iou, num_ious / 8


def plot_clusters(sims):
    global color

    clusters = sims.argmax(dim=1)[0].cpu().numpy()

    seg = np.zeros([clusters.shape[0],clusters.shape[1],3],dtype=np.uint8)
    for c in range(clusters.max()+1):
        seg[clusters==c] = color[c]
    
    seg = cv2.resize(seg, (seg.shape[1]*8,seg.shape[0]*8))
    return seg