import numpy as np
import cv2
import torch
import glob
import datetime
import random

from vision_parser import VisionParser
from utils import normalize_image, plot_clusters

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('weights','test','')
flags.DEFINE_string('root_dir','/home/petrus/ADE20K/images/','')
flags.DEFINE_integer('num_prototypes',100,'')
flags.DEFINE_integer('embd_dim',512,'')


def main(argv):

    all_images = glob.glob(FLAGS.root_dir+"ADE/training/work_place/*/*.jpg")
    random.shuffle(all_images)

    model = VisionParser()
    model.to('cuda')
    model.load_state_dict(torch.load('weights/'+FLAGS.weights))
    model.eval()

    for img_path in all_images:

        img_cv = cv2.imread(img_path)[:,:,::-1]
        h,w,_ = img_cv.shape
        img = torch.from_numpy(np.ascontiguousarray(img_cv)).float()
        img = img.movedim(2,0)
        img = normalize_image(img).unsqueeze(0).to('cuda')

        print(img.shape)
        feature_map = model.extract_feature_map(img)
        sims = model.assign_nearest_clusters(feature_map, ret_sims=True)
        seg_map = plot_clusters(sims)

        cv2.imshow('a', seg_map)
        cv2.imshow('b', img_cv[:,:,::-1])
        cv2.waitKey(0)




if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)