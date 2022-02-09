import numpy as np
import cv2
import torch
import glob
import datetime
import random

from vision_parser import VisionParser
from dataloader import ADE20k_Dataset
from utils import color_distortion

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')

torch.multiprocessing.set_sharing_strategy('file_system')


def main(argv):
    start = datetime.datetime.now()

    wandb.init(project="VisionParser",name=FLAGS.exp)
    wandb.save("train.py")
    wandb.save("vision_parser.py")
    wandb.save("dataloader.py")
    wandb.save("utils.py")
    wandb.config.update(flags.FLAGS)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    all_images = glob.glob(FLAGS.root_dir+"/ADE20K/work_place/*/*.jpg")
    random.shuffle(all_images)
    train_images = all_images[:-16]
    val_images = all_images[-16:]

    print("Num train images:",len(train_images))
    print("Num val images:",len(val_images))

    training_set = ADE20k_Dataset(train_images)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

    validation_set = ADE20k_Dataset(val_images)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)
    
    if FLAGS.plot:
        model.load_state_dict(torch.load('weights/22Sep1.pt'))
        FLAGS.lr = 0.

    model.to('cuda')

   
    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    for epoch in range(10000):
        model.train()
        # Set optimzer gradients to zero
        optimizer.zero_grad()
        for frames_load in training_generator:
            #frames = [frame.to('cuda') for frame in frames_load]
            images = [color_aug(img.to('cuda')) for img in frames_load[0]]
        

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)