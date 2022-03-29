import numpy as np
import cv2
import torch
import glob
import datetime
import random

from vision_parser import VisionParser
from dataloader import ADE20k_Dataset, PascalVOC
from utils import color_distortion, calculate_iou

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('dataset','pascal','pascal,ade')
flags.DEFINE_string('root_dir','/home/petrus/','')
flags.DEFINE_integer('num_workers',0,'')
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_integer('image_size',256,'')
flags.DEFINE_integer('embd_dim',512,'')
flags.DEFINE_integer('num_crops',3,'')
flags.DEFINE_float('min_crop',0.55,'')
flags.DEFINE_float('max_crop',0.75,'')
flags.DEFINE_float('assign_thresh',0.7,'')

flags.DEFINE_integer('num_prototypes',100,'')
flags.DEFINE_integer('min_per_img_embds',5,'')
flags.DEFINE_integer('num_embds_per_cluster',5,'')
flags.DEFINE_integer('min_valid_clusts',5,'')
flags.DEFINE_bool('sg_cluster_assign',True,'')
flags.DEFINE_integer('sinkhorn_iters',3,'')
flags.DEFINE_bool('round_q',True,'')
flags.DEFINE_float('epsilon',0.05,'')

flags.DEFINE_float('cl_temp',0.1,'')

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
        train_images = all_images[:-600]
        val_images = all_images[-600:]

        training_set = ADE20k_Dataset(train_images)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = ADE20k_Dataset(val_images)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)
    elif FLAGS.dataset == 'pascal':
        all_images = glob.glob(FLAGS.root_dir+"VOCdevkit/VOC2012/trainval/trainval/*")
        random.seed(7)
        random.shuffle(all_images)
        train_images = all_images[:-600]
        val_images = all_images[-600:]

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
            image_batch = [color_aug(img.to('cuda')) for img in frames_load[0]]

            crop_dims = frames_load[1]
            image_batch = torch.cat(image_batch, dim=0)

            feature_maps = model.extract_feature_map(image_batch)
            fm_full,fm_a,fm_b = feature_maps[:FLAGS.batch_size],feature_maps[FLAGS.batch_size:2*FLAGS.batch_size],feature_maps[2*FLAGS.batch_size:]

            max_cluster_mask = model.assign_nearest_clusters(fm_full)
            features_clust_a, features_clust_b, features_sim_a, features_sim_b = model.spatial_map_cluster_assign([fm_a,fm_b],max_cluster_mask, crop_dims)

            loss_1 = model.swav_loss(features_clust_a, features_sim_b)
            loss_2 = model.swav_loss(features_clust_b, features_sim_a)
            loss = loss_1 + loss_2

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.net.parameters(), 10000.)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                num_embds_per_cluster = max_cluster_mask.sum((0,2,3))

                annots = frames_load[2]
                miou, num_ious = calculate_iou(max_cluster_mask.to('cpu'), annots)

            train_iter += 1
            log_dict = {"Epoch":epoch, "Iter":train_iter, "Loss": loss, "Grad Norm": grad_norm, \
                        "Num A": features_clust_a.shape[0]/FLAGS.num_embds_per_cluster, \
                        "MIOU": miou, "Num IOUS": num_ious, \
                        "C0": num_embds_per_cluster[0], "C1": num_embds_per_cluster[1], "C2": num_embds_per_cluster[2]}
            
            if train_iter % 10 == 0:
                print(log_dict)

            wandb.log(log_dict)

            if train_iter % 300 == 0:
                for g in optimizer.param_groups:
                    g['lr'] /= 10

        with torch.no_grad():
            val_iter = 0
            total_val_loss = 0.
            total_miou = 0
            total_num_ious = 0
            for frames_load in validation_generator:
                
                image_batch = [color_aug(img.to('cuda')) for img in frames_load[0]]

                crop_dims = frames_load[1]
                image_batch = torch.cat(image_batch, dim=0)

                feature_maps = model.extract_feature_map(image_batch)
                fm_full,fm_a,fm_b = feature_maps[:FLAGS.batch_size],feature_maps[FLAGS.batch_size:2*FLAGS.batch_size],feature_maps[2*FLAGS.batch_size:]

                max_cluster_mask = model.assign_nearest_clusters(fm_full)
                features_clust_a, features_clust_b, features_sim_a, features_sim_b = model.spatial_map_cluster_assign([fm_a,fm_b],max_cluster_mask, crop_dims)

                loss_1 = model.swav_loss(features_clust_a, features_sim_b)
                loss_2 = model.swav_loss(features_clust_b, features_sim_a)
                loss = loss_1 + loss_2

                total_val_loss += loss
                val_iter += 1

                with torch.no_grad():
                    annots = frames_load[2]
                    miou, num_ious = calculate_iou(max_cluster_mask.to('cpu'), annots)
                    total_miou += miou
                    total_num_ious += num_ious


            avg_val_loss = total_val_loss/val_iter
            log_dict = {"Epoch":epoch, "Val Loss": avg_val_loss, \
                        "Val MIOU": total_miou/val_iter, "Val Num IOUS": total_num_ious/val_iter, \
                        "Val Num Valid A": features_clust_a.shape[0]/FLAGS.num_embds_per_cluster}
            
            if val_iter % 10 == 0:
                print(log_dict)

            wandb.log(log_dict)

            if avg_val_loss < min_loss:
                torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                min_loss = avg_val_loss
        

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)



'''overlap_dims = (max(crop_dims[0][0],crop_dims[1][0]), max(crop_dims[0][1],crop_dims[1][1]), \
                        min(crop_dims[0][0]+crop_dims[0][2],crop_dims[1][0]+crop_dims[1][2]), min(crop_dims[0][1]+crop_dims[0][2],crop_dims[1][1]+crop_dims[1][2]))

            fm_a_crop = image_batch[1][:,:, \
                        round((overlap_dims[1]-crop_dims[0][1])*32/crop_dims[0][2])*8: \
                        round((overlap_dims[3]-crop_dims[0][1])*32/crop_dims[0][2])*8, \
                        round((overlap_dims[0]-crop_dims[0][0])*32/crop_dims[0][2])*8: \
                        round((overlap_dims[2]-crop_dims[0][0])*32/crop_dims[0][2])*8] # B,c,fm_a_h,fm_a_w

            fm_b_crop = image_batch[2][:,:, \
                        round((overlap_dims[1]-crop_dims[1][1])*32/crop_dims[1][2])*8: \
                        round((overlap_dims[3]-crop_dims[1][1])*32/crop_dims[1][2])*8, \
                        round((overlap_dims[0]-crop_dims[1][0])*32/crop_dims[1][2])*8: \
                        round((overlap_dims[2]-crop_dims[1][0])*32/crop_dims[1][2])*8] # B,c,fm_b_h,fm_b_w

            for j,imgs in enumerate([fm_a_crop,fm_b_crop]):
                np_imgs = imgs.movedim(1,3).to('cpu').numpy() * 255
                print(np_imgs.mean())
                np_imgs = np_imgs.astype(np.uint8)
                for i,img in enumerate(np_imgs):
                    cv2.imshow("{}_{}".format(j,i),img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()'''