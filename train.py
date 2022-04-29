import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime
import random

from vision_parser import VisionParser
from dataloader import ADE20k_Dataset, PascalVOC
from utils import color_distortion, calculate_iou, k_means

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('dataset','pascal','pascal,ade')
flags.DEFINE_string('root_dir','/home/petrus/','')
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_integer('image_size',256,'')
flags.DEFINE_integer('embd_dim',512,'')
flags.DEFINE_integer('num_crops',3,'')
flags.DEFINE_float('min_crop',0.55,'')
flags.DEFINE_float('max_crop',0.75,'')
flags.DEFINE_float('color_aug',0.5,'')
flags.DEFINE_bool('main_aug',True,'')

flags.DEFINE_integer('niter',20,'')
flags.DEFINE_bool('ce_root',True,'')

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

    if FLAGS.dataset == 'ade':
        all_images = glob.glob(FLAGS.root_dir+"ADE20K/images/ADE/training/work_place/*/*.jpg")
        random.seed(7)
        random.shuffle(all_images)
        train_images = all_images[:-300]
        val_images = all_images[-300:]

        training_set = ADE20k_Dataset(train_images)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = ADE20k_Dataset(val_images)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

    elif FLAGS.dataset == 'pascal':
        all_images = glob.glob(FLAGS.root_dir+"VOCdevkit/VOC2012/trainval/trainval/*.mat")
        random.seed(7)
        random.shuffle(all_images)
        train_images = all_images[:-300]
        val_images = all_images[-300:]

        training_set = PascalVOC(train_images)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = PascalVOC(val_images)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)


    print("Num train images:",len(train_images))
    print("Num val images:",len(val_images))

    color_aug = color_distortion()

    model = VisionParser()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=FLAGS.lr)

    model.to('cuda')
   
    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    for epoch in range(50):
        model.train()
        # Set optimzer gradients to zero
        optimizer.zero_grad()
        for frames_load in training_generator:
            #with torch.autograd.detect_anomaly():
            if FLAGS.main_aug:
                image_batch = [color_aug(img.to('cuda')) for img in frames_load[0]]
            else:
                image_batch = [frames_load[0][0].to('cuda')] + [color_aug(img.to('cuda')) for img in frames_load[0][1:]]

            crop_dims = frames_load[1]
            image_batch = torch.cat(image_batch, dim=0)

            feature_maps = model.extract_feature_map(image_batch)
            main_features,crop_features_a,crop_features_b = feature_maps[:FLAGS.batch_size],feature_maps[FLAGS.batch_size:2*FLAGS.batch_size],feature_maps[2*FLAGS.batch_size:]
            embds = main_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)
            
            losses = []
            min_pts,max_pts = [],[]
            diffs = []
            for K in [25,50,100,200]:
                #print(K,datetime.datetime.now())
                with torch.no_grad():
                    cl_idxs, centroids, ce, num_points, diff = k_means(embds, K) # (N,),(K,D),(K,),(K,)

                #print(datetime.datetime.now())

                sims,pos_idxs = model.get_sims([crop_features_a,crop_features_b],crop_dims,centroids,cl_idxs)

                loss = 0.
                for cr_sims,p_idxs in zip(sims,pos_idxs):
                    loss += F.cross_entropy(cr_sims/(ce[None,:] + 0.001), p_idxs) / K

                losses.append(loss)
                min_pts.append(num_points.min()/cl_idxs.shape[0])
                max_pts.append(num_points.max()/cl_idxs.shape[0])
                diffs.append(diff)
            
            final_loss = sum(losses)

            final_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.net.parameters(), 10000.)

            optimizer.step()
            optimizer.zero_grad()

            log_dict = {"Epoch":epoch, "Iter":train_iter, "Loss": final_loss, "Grad Norm": grad_norm}

            k = 0
            for min_k,max_k,loss_k,diff_k in zip(min_pts,max_pts,losses,diffs):
                log_dict["Loss_K{}".format(k)] = loss_k
                log_dict["MinPts_K{}".format(k)] = min_k
                log_dict["MaxPts_K{}".format(k)] = max_k
                log_dict["Diff_K{}".format(k)] = diff_k

                k += 1


            '''if train_iter % 20 == 0:
                with torch.no_grad():
                    #num_embds_per_cluster = max_cluster_mask.sum((0,2,3))

                    annots = frames_load[2]
                    miou, num_ious = calculate_iou(max_cluster_mask.to('cpu'), annots)

                    log_dict['MIOU'] = miou
                    log_dict['Num IOUS'] = num_ious'''

            
            train_iter += 1

            if train_iter % 10 == 0 or train_iter < 20:
                print(log_dict)

            wandb.log(log_dict)

            #if train_iter == 300 or train_iter == 800:
            #    for g in optimizer.param_groups:
            #        g['lr'] /= 10

        with torch.no_grad():
            val_iter = 0
            total_val_loss = 0.
            losses_k = [0.,0.,0.,0.]
            total_miou = 0
            total_num_ious = 0
            for frames_load in validation_generator:
                if FLAGS.main_aug:
                    image_batch = [color_aug(img.to('cuda')) for img in frames_load[0]]
                else:
                    image_batch = [frames_load[0][0].to('cuda')] + [color_aug(img.to('cuda')) for img in frames_load[0][1:]]

                crop_dims = frames_load[1]
                image_batch = torch.cat(image_batch, dim=0)

                feature_maps = model.extract_feature_map(image_batch)
                main_features,crop_features_a,crop_features_b = feature_maps[:FLAGS.batch_size],feature_maps[FLAGS.batch_size:2*FLAGS.batch_size],feature_maps[2*FLAGS.batch_size:]
                embds = main_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)
                
                losses = []
                for k_ix,K in enumerate([25,50,100,200]):
                    cl_idxs, centroids, ce, num_points, diff = k_means(embds, K) # (N,),(K,D),(K,),(K,)

                    sims,pos_idxs = model.get_sims([crop_features_a,crop_features_b],crop_dims,centroids,cl_idxs)

                    loss = 0.
                    for cr_sims,p_idxs in zip(sims,pos_idxs):
                        loss += F.cross_entropy(cr_sims/(ce[None,:] + 0.001), p_idxs) / K

                    losses.append(loss)
                    losses_k[k_ix] += loss
                
                total_val_loss += sum(losses)

                val_iter += 1

                '''with torch.no_grad():
                    annots = frames_load[2]
                    miou, num_ious = calculate_iou(max_cluster_mask.to('cpu'), annots)
                    total_miou += miou
                    total_num_ious += num_ious'''


            avg_val_loss = total_val_loss/val_iter
            log_dict = {"Epoch":epoch, "Val Loss": avg_val_loss, \
                        "Val Loss_K0":losses_k[0]/val_iter, "Val Loss_K1":losses_k[1]/val_iter, \
                        "Val Loss_K2":losses_k[2]/val_iter, "Val Loss_K3":losses_k[3]/val_iter}
            
            print(log_dict)

            wandb.log(log_dict)

            if avg_val_loss < min_loss:
                torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                min_loss = avg_val_loss
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)