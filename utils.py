import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
#import matplotlib.pyplot as plt
import time
from kornia.filters import sobel
from pykeops.torch import LazyTensor

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


def random_crop(image, crop_dims=None, min_crop=None):
    c, img_h, img_w = image.shape

    if crop_dims is None:
        crop_size = np.random.uniform(min_crop,min(img_h/img_w,img_w/img_h))
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
    

def k_means(x, K):
    N = x.shape[0]
    x = F.normalize(x)
    step = int(N/K)
    c = x[::step, :][:K].clone()  # (K,D) Simplistic initialization for the centroids
    prev_c = c.clone()
    assert c.shape[0] == K

    x_i = LazyTensor(x.view(N, 1, FLAGS.embd_dim))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, FLAGS.embd_dim))  # (1, K, D) centroids

    for i in range(FLAGS.niter):

        # E step: assign points to the closest cluster -------------------------
        #D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        D_ij = 1 - (x_i | c_j) # (N, K)
        cl = D_ij.argmin(dim=1).long().view(-1)  # (N,) Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, FLAGS.embd_dim), x)

        # Divide by the number of points per cluster:
        #Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        #c /= Ncl  # in-place division to compute the average

        c[:] = F.normalize(c)

        if i > 20:
            diff = (c - prev_c).norm()/(K**0.5)
            if diff < FLAGS.diff_thresh:
                break

            prev_c = c.clone()

    ce, num_points = concentration_estimation(D_ij,cl)

    #unique,counts = torch.unique(cl,return_counts=True)
    #print(counts.sort()[0]/cl.shape[0])

    diff = (c - prev_c).norm()/(K**0.5)

    return cl, c, ce, num_points, diff


def concentration_estimation(dists,cl):
    centroid_dists = dists.min(dim=1)[:,0] # (N,)
    num_points_per_clust = torch.bincount(cl) # (K,)
    if dists.shape[1]-5 < num_points_per_clust.shape[0] < dists.shape[1]:
        print("Num Points", num_points_per_clust.shape[0], dists.shape[1])
        num_points_per_clust = torch.bincount(cl, minlength=dists.shape[1]) # (K,)

    assert num_points_per_clust.shape[0] == dists.shape[1]
    if FLAGS.ce_root:
        centroid_dists = torch.sqrt(torch.clamp(centroid_dists,min=0.))
    avg_dist = torch.scatter_add(torch.zeros(dists.shape[1]).to('cuda'), 0, cl, centroid_dists) / (num_points_per_clust + 1)
    ce = avg_dist / torch.log(num_points_per_clust + 10)

    return ce, num_points_per_clust


def sobel_filter(imgs):
    with torch.no_grad():
        sobel_mags = sobel(imgs).mean(1) # B,H,W
        if FLAGS.sobel_mag_thresh > 0:
            sobel_mags[sobel_mags < FLAGS.sobel_mag_thresh] = 0.
            sobel_mags[sobel_mags >= FLAGS.sobel_mag_thresh] = 1.

        sobel_mags = sobel_mags.reshape(FLAGS.batch_size,32,8,32,8).movedim(2,3).mean([3,4]).reshape(-1) # N
        sobel_mask = sobel_mags < FLAGS.sobel_pix_thresh

    return sobel_mask


def vic_reg(x):

    x = x.movedim(1,3).reshape(FLAGS.batch_size,-1,FLAGS.embd_dim)
    x = x - x.mean(dim=1, keepdim=True)
    std_x = torch.sqrt(x.var(dim=1) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x))

    cov_x = torch.matmul(x.movedim(1,2),x) / (x.shape[1] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(FLAGS.batch_size*FLAGS.embd_dim)

    return std_loss, cov_loss


def off_diagonal(x):
    b, n, m = x.shape
    assert n == m
    return x.reshape(b,-1)[:,:-1].reshape(b, n - 1, n + 1)[:,:,1:].reshape(b,-1)


def color_distortion():
    color_jitter = torchvision.transforms.ColorJitter(FLAGS.color_aug,FLAGS.color_aug,FLAGS.color_aug,FLAGS.color_aug*0.2)
    return color_jitter


def calculate_iou(cluster_mask, annots):
    # cluster_mask B,num_proto,h/8,w/8
    # annots B,460,h,w

    annots = F.one_hot(annots.long().squeeze(1), 460).movedim(3,1)
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


def plot_clusters(clust_idxs):
    global color

    seg = np.zeros([clust_idxs.shape[0],clust_idxs.shape[1],3],dtype=np.uint8)
    for c in range(clust_idxs.max()+1):
        seg[clust_idxs==c] = color[c]
    
    seg = cv2.resize(seg, (seg.shape[1]*8,seg.shape[0]*8))
    return seg