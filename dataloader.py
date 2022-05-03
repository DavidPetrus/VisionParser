import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time
import scipy.io

from utils import normalize_image, random_crop

from absl import flags

FLAGS = flags.FLAGS

class PascalVOC(torch.utils.data.Dataset):

    def __init__(self, image_files, labels=None):

        self.image_files = image_files
        self.batch_size = FLAGS.batch_size
        self.load_ann = True

        #cv2.setNumThreads(0)
        self.num_crops = FLAGS.num_crops
        self.min_crop = FLAGS.min_crop
        self.max_crop = FLAGS.max_crop
        self.image_size = FLAGS.image_size

    def __len__(self):
        return len(self.image_files)//self.batch_size

    def __getitem__(self, index):
        img_batch = [[] for c in range(self.num_crops)]
        crop_dims = []
        for c in range(1,self.num_crops):
            crop_size = np.random.uniform(self.min_crop,self.max_crop)
            crop_dims.append((np.random.uniform(0.,1.-crop_size),np.random.uniform(0.,1.-crop_size),crop_size))

        image_files = np.random.choice(self.image_files,self.batch_size,replace=False)
        annots = []

        for img_file in image_files:

            img = cv2.imread(img_file.replace("trainval/trainval","JPEGImages").replace(".mat",".jpg"))[:,:,::-1]
            h,w,_ = img.shape
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            img = img.movedim(2,0)

            if self.load_ann:
                annot = scipy.io.loadmat(img_file)['LabelMap'].astype(np.float32)
                main_crop,dims_main = random_crop(img, min_crop=self.min_crop)
                annot = torch.from_numpy(annot[dims_main[1]:dims_main[1]+dims_main[2],dims_main[0]:dims_main[0]+dims_main[2]])
                annot = F.interpolate(annot.unsqueeze(0).unsqueeze(0),size=(self.image_size,self.image_size),mode='nearest')
                annots.append(annot)
            else:
                main_crop,_ = random_crop(img, min_crop=self.min_crop)

            normalized = normalize_image(main_crop)
            resized = F.interpolate(normalized.unsqueeze(0),size=(self.image_size,self.image_size),mode='bilinear',align_corners=False)
            img_batch[0].append(resized)

            for c in range(1,self.num_crops):
                cr,_ = random_crop(main_crop, crop_dims=crop_dims[c-1])
                cr = normalize_image(cr)
                cr = F.interpolate(cr.unsqueeze(0),size=(self.image_size,self.image_size),mode='bilinear',align_corners=False)
                img_batch[c].append(cr)

        if self.load_ann:
            return [torch.cat(cr,dim=0) for cr in img_batch], crop_dims, torch.cat(annots,dim=0)
        else:
            return [torch.cat(cr,dim=0) for cr in img_batch], crop_dims


class ADE20k_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_files, labels=None):

        self.image_files = image_files
        self.batch_size = FLAGS.batch_size

    def __len__(self):
        return len(self.image_files)//self.batch_size

    def __getitem__(self, index):
        img_batch = [[] for c in range(FLAGS.num_crops)]
        crop_size_a = np.random.uniform(FLAGS.min_crop,FLAGS.max_crop)
        crop_size_b = np.random.uniform(FLAGS.min_crop,FLAGS.max_crop)
        crop_dims = [(np.random.uniform(0.,1.-crop_size_a),np.random.uniform(0.,1.-crop_size_a),crop_size_a), \
                     (np.random.uniform(0.,1.-crop_size_b),np.random.uniform(0.,1.-crop_size_b),crop_size_b)]

        image_files = np.random.choice(self.image_files,self.batch_size,replace=False)

        for img_file in image_files:

            img = cv2.imread(img_file)[:,:,::-1]
            h,w,_ = img.shape

            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            img = img.movedim(2,0)

            main_crop,_ = random_crop(img)
            normalized = normalize_image(main_crop)
            resized = F.interpolate(normalized.unsqueeze(0),size=(FLAGS.image_size,FLAGS.image_size),mode='bilinear',align_corners=False)
            img_batch[0].append(resized)

            for c in range(1,FLAGS.num_crops):
                cr,_ = random_crop(main_crop, crop_dims=crop_dims[c-1])
                cr = normalize_image(cr)
                cr = F.interpolate(cr.unsqueeze(0),size=(FLAGS.image_size,FLAGS.image_size),mode='bilinear',align_corners=False)
                img_batch[c].append(cr)

        return [torch.cat(cr,dim=0) for cr in img_batch], crop_dims
