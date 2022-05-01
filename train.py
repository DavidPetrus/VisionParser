import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime
import random
import ast

from vision_parser import VisionParser
from dataloader import ADE20k_Dataset, PascalVOC
from utils import color_distortion, calculate_iou, k_means, sobel_filter

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('dataset','pascal','pascal,ade')
flags.DEFINE_string('root_dir','/home/petrus/','')
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_integer('batch_size',128,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_integer('image_size',256,'')
flags.DEFINE_integer('embd_dim',512,'')
flags.DEFINE_integer('num_crops',3,'')
flags.DEFINE_float('min_crop',0.55,'')
flags.DEFINE_float('max_crop',0.75,'')
flags.DEFINE_float('color_aug',0.8,'')
flags.DEFINE_bool('main_aug',False,'')

flags.DEFINE_float('epsilon',0.001,'')
flags.DEFINE_float('temperature',0.1,'')
flags.DEFINE_float('margin',0.,'')

flags.DEFINE_string("K","20,40,80",'')
flags.DEFINE_integer('niter',40,'')
flags.DEFINE_float('diff_thresh',0.01,'')
flags.DEFINE_bool('ce_root',True,'')
flags.DEFINE_bool('use_ce',True,'')

flags.DEFINE_float('sobel_mag_thresh',0.1,'')
flags.DEFINE_float('sobel_pix_thresh',0.2,'')

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
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    model.to('cuda')

    all_Ks = ast.literal_eval(FLAGS.K)
   
    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    for epoch in range(12):
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

            with torch.no_grad():
                main_features = model.extract_feature_map(image_batch[0])
                embds = main_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)

                sobel_mask = sobel_filter(image_batch[0]) # N
                embds = embds[sobel_mask]
                
                min_pts,max_pts = [],[]
                diffs = []
                ces = []
                centr_sims = []
                centroids_K = []
                cl_idxs_K = []
                for K in all_Ks:
                    with torch.no_grad():
                        cl_idxs, centroids, ce, num_points, diff = k_means(embds, K) # (N,),(K,D),(K,),(K,)
                        cl_idxs = torch.scatter(-torch.ones(sobel_mask.shape[0],dtype=torch.long).to('cuda'), 0, sobel_mask.nonzero().reshape(-1), cl_idxs)
                        centr_sims.append(((centroids @ centroids.T).sum() - K)/(K*(K-1)))

                        min_pts.append(num_points.min()/cl_idxs.shape[0])
                        max_pts.append(num_points.max()/cl_idxs.shape[0])
                        diffs.append(diff)
                        ces.append(ce)
                        centroids_K.append(centroids)
                        cl_idxs_K.append(cl_idxs)

            
            for cr_img,cr_dims in zip(image_batch[1:],crop_dims):
                crop_features = model.extract_feature_map(cr_img)
                crop_features = F.normalize(crop_features)
                losses = []
                for K,centroids,cl_idxs,ce in zip(all_Ks,centroids_K,cl_idxs_K,ces):
                    cr_features,p_idxs = model.mask_crop_feats(crop_features,cr_dims,cl_idxs) # (N,D), (N,)
                    cr_sims = cr_features @ centroids.T # (N,K)

                    pos_mask = torch.scatter(torch.zeros(cr_sims.shape[0],cr_sims.shape[1]).to('cuda'),1,p_idxs[:,None],1.)
                    arc_sims = torch.where(pos_mask==1., torch.cos(torch.acos(cr_sims.clamp(min=-0.999)-0.001)+FLAGS.margin), cr_sims)
                    if not FLAGS.use_ce:
                        loss = F.cross_entropy(arc_sims/FLAGS.temperature, p_idxs) / K
                    else:
                        loss = F.cross_entropy(arc_sims/(FLAGS.temperature*ce[None,:]/ce.mean() + FLAGS.epsilon), p_idxs) / K

                    losses.append(loss)
                    
                final_loss = sum(losses)

                final_loss.backward()

                log_dict = {"Epoch":epoch, "Iter":train_iter, "Loss": final_loss, \
                        "Frac Masked": 1-sobel_mask.sum()/sobel_mask.shape[0], "Crop Frac Masked": 1-p_idxs.shape[0]/sobel_mask.shape[0]}

                k = 0
                for min_k,max_k,loss_k,diff_k,ce_k,c_sim in zip(min_pts,max_pts,losses,diffs,ces,centr_sims):
                    log_dict["Loss_K{}".format(k)] = loss_k
                    log_dict["MinPts_K{}".format(k)] = min_k
                    log_dict["MaxPts_K{}".format(k)] = max_k
                    log_dict["Diff_K{}".format(k)] = diff_k
                    log_dict["CE_K{}".format(k)] = ce_k.mean()
                    log_dict["CentSim_K{}".format(k)] = c_sim

                    k += 1
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.)
            log_dict["Grad Norm"] = grad_norm

            optimizer.step()
            optimizer.zero_grad()

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

            if train_iter == 1000:
                for g in optimizer.param_groups:
                    g['lr'] /= 10

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
                main_features = model.extract_feature_map(image_batch[0])
                embds = main_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)

                sobel_mask = sobel_filter(image_batch[0]) # N
                embds = embds[sobel_mask]
                
                ces = []
                centr_sims = []
                centroids_K = []
                cl_idxs_K = []
                for K in all_Ks:
                    with torch.no_grad():
                        cl_idxs, centroids, ce, num_points, diff = k_means(embds, K) # (N,),(K,D),(K,),(K,)
                        cl_idxs = torch.scatter(-torch.ones(sobel_mask.shape[0],dtype=torch.long).to('cuda'), 0, sobel_mask.nonzero().reshape(-1), cl_idxs)
                        centr_sims.append(((centroids @ centroids.T).sum() - K)/(K*(K-1)))

                        ces.append(ce)
                        centroids_K.append(centroids)
                        cl_idxs_K.append(cl_idxs)

            
                losses = []
                for cr_img,cr_dims in zip(image_batch[1:],crop_dims):
                    crop_features = model.extract_feature_map(cr_img)
                    crop_features = F.normalize(crop_features)
                    
                    for K,centroids,cl_idxs,ce in zip(all_Ks,centroids_K,cl_idxs_K,ces):
                        cr_features,p_idxs = model.mask_crop_feats(crop_features,cr_dims,cl_idxs) # (N,D), (N,)
                        cr_sims = cr_features @ centroids.T # (N,K)
                        if not FLAGS.use_ce:
                            loss = F.cross_entropy(cr_sims/FLAGS.temperature, p_idxs) / K
                        else:
                            loss = F.cross_entropy(cr_sims/(FLAGS.temperature*ce[None,:]/ce.mean() + FLAGS.epsilon), p_idxs) / K

                        losses.append(loss)
                
                total_val_loss += sum(losses)/len(crop_dims)

                val_iter += 1

                '''with torch.no_grad():
                    annots = frames_load[2]
                    miou, num_ious = calculate_iou(max_cluster_mask.to('cpu'), annots)
                    total_miou += miou
                    total_num_ious += num_ious'''


            avg_val_loss = total_val_loss/val_iter
            log_dict = {"Epoch":epoch, "Val Loss": avg_val_loss}
            #            "Val Loss_K0":losses_k[0]/val_iter, "Val Loss_K1":losses_k[1]/val_iter, \
            #            "Val Loss_K2":losses_k[2]/val_iter}
            
            print(log_dict)

            wandb.log(log_dict)

            if avg_val_loss < min_loss:
                torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                min_loss = avg_val_loss
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)