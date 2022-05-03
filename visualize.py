import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime
import random

from dataloader import PascalVOC
from vision_parser import VisionParser
from utils import plot_clusters, random_crop, sobel_filter

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('weights','test','')
flags.DEFINE_string('root_dir','/home/petrus/','')
flags.DEFINE_integer('image_size',256,'')
flags.DEFINE_integer('embd_dim',512,'')
flags.DEFINE_integer('batch_size',128,'')
flags.DEFINE_integer('num_crops',3,'')
flags.DEFINE_float('min_crop',0.55,'')
flags.DEFINE_float('max_crop',0.75,'')

flags.DEFINE_integer('num_prototypes',50,'')
flags.DEFINE_integer('min_pts',20,'')
flags.DEFINE_float('max_clust_size',0.1,'')
flags.DEFINE_string('selection_method','eom','')
flags.DEFINE_float('frac_per_img',0.1,'')
flags.DEFINE_string('cluster_metric','euc','')

flags.DEFINE_float('sobel_mag_thresh',0.14,'')
flags.DEFINE_float('sobel_pix_thresh',0.2,'')


def main(argv):

    all_images = glob.glob(FLAGS.root_dir+"VOCdevkit/VOC2012/trainval/trainval/*.mat")
    random.seed(7)
    random.shuffle(all_images)
    val_images = all_images[:-300]

    validation_set = PascalVOC(val_images)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True)

    model = VisionParser()
    model.to('cuda')
    state_dict = torch.load('weights/{}'.format(FLAGS.weights),map_location=torch.device('cuda'))
    model.net = state_dict['net']
    model.proj_head = state_dict['proj_head']
    model.prototypes = state_dict['prototypes']
    #model.eval()

    for frames_load in validation_generator:
        image_batch = frames_load[0][0].to('cuda')

        feature_maps = model.extract_feature_map(image_batch)
        embds = feature_maps.movedim(1,3).reshape(-1, FLAGS.embd_dim)
        sobel_mask = sobel_filter(image_batch[:FLAGS.batch_size]) # N
        embds = embds[sobel_mask]
        
        norm_embds = F.normalize(embds)
        proto_sims = norm_embds @ model.prototypes.T # (N, num_prototypes)
        cl_idxs = torch.argmax(proto_sims, dim=1) # (N,)
        cl_idxs = torch.scatter(-torch.ones(sobel_mask.shape[0],dtype=torch.long,device=torch.device('cuda')), 0, sobel_mask.nonzero().reshape(-1), cl_idxs) \
                        .reshape(FLAGS.batch_size,32,32).cpu().numpy()  # (B,32,32)

        for img,cl in zip(image_batch,cl_idxs):
            img_cv = (img.movedim(0,2) * 255).cpu().numpy().astype(np.uint8)
            seg_map = plot_clusters(cl)

            cv2.imshow('a', seg_map)
            cv2.imshow('b', img_cv[:,:,::-1])
            key = cv2.waitKey(0)
            if key == ord('q'):
                break




if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)