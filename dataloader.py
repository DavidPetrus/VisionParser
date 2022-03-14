import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time

from utils import resize_image, normalize_image, mask_random_crop, random_crop_resize, color_distortion

from absl import flags

FLAGS = flags.FLAGS

class ADE20k_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_files, labels=None):

        self.image_files = image_files
        self.batch_size = FLAGS.batch_size

    def __len__(self):
        return len(self.image_files)//self.batch_size

    def __getitem__(self, index):
        img_batch = [[] for c in range(FLAGS.num_crops)]
        crop_size_a = np.random.uniform(FLAGS.min_crop,1.)
        crop_size_b = np.random.uniform(FLAGS.min_crop,1.)
        crop_dims = [(np.random.uniform(0.,1.-crop_size_a),np.random.uniform(0.,1.-crop_size_a),crop_size_a), \
                     (np.random.uniform(0.,1.-crop_size_b),np.random.uniform(0.,1.-crop_size_b),crop_size_b)]

        for i in range(self.batch_size):
            # Select sample
            img_file = self.image_files[self.batch_size*index + i]

            img = cv2.imread(img_file)[:,:,::-1]
            h,w,_ = img.shape
            '''aspect_ratio = h/w
            if max(h,w) > 512:
                if h > w:
                    img = cv2.resize(img, (int(512//aspect_ratio),512))
                else:
                    img = cv2.resize(img, (512,int(aspect_ratio*512)))'''

            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            img = img.movedim(2,0)

            main_crop, _ = random_crop_resize(img)
            normalized = normalize_image(main_crop)
            resized = F.interpolate(normalized.unsqueeze(0),size=(FLAGS.image_size,FLAGS.image_size),mode='bilinear',align_corners=True)
            img_batch[0].append(resized)

            for c in range(1,FLAGS.num_crops):
                cr = random_crop_resize(main_crop, crop_dims=crop_dims[c-1])
                cr = normalize_image(cr)
                cr = F.interpolate(cr.unsqueeze(0),size=(FLAGS.image_size,FLAGS.image_size),mode='bilinear',align_corners=True)
                img_batch[c].append(cr)

        return [torch.cat(cr,dim=0) for cr in img_batch], crop_dims
