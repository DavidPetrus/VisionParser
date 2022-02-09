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
        img_batch = []
        augs_batch = [[] for c in range(FLAGS.num_crops)]
        all_dims = [[] for c in range(FLAGS.num_crops)]
        for i in range(self.batch_size):
            # Select sample
            img_file = self.image_files[self.batch_size*index + i]

            img = cv2.imread(img_file)[:,:,::-1]
            h,w,_ = img.shape
            aspect_ratio = h/w
            if max(h,w) > 512:
                if h > w:
                    img = cv2.resize(img, (int(512//aspect_ratio),512))
                else:
                    img = cv2.resize(img, (512,int(aspect_ratio*512)))

            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            img = img.movedim(2,0)

            for c in range(FLAGS.num_crops):
                #aug, crop_dims = random_crop_resize(img, crop_props[c][0], crop_props[c][1])
                aug, crop_dims = random_crop_resize(img)
                aug = normalize_image(aug)
                augs_batch[c].append(aug)
                all_dims[c].append(crop_dims)
            
            img = normalize_image(img)
            img_batch.append(img.unsqueeze(0))

        return img_batch, augs_batch, all_dims
