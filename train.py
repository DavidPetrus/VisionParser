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

from scipy.optimize import linear_sum_assignment

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('dataset','pascal','pascal,ade')
flags.DEFINE_string('root_dir','/home/petrus/','')
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_integer('batch_size',128,'')
flags.DEFINE_float('lr',0.003,'')
flags.DEFINE_integer('image_size',256,'')
flags.DEFINE_integer('embd_dim',512,'')
flags.DEFINE_integer('num_crops',3,'')
flags.DEFINE_float('min_crop',0.75,'')
flags.DEFINE_float('max_crop',0.95,'')
flags.DEFINE_float('color_aug',0.8,'')
flags.DEFINE_bool('main_aug',False,'')

flags.DEFINE_float('epsilon',0.001,'')
flags.DEFINE_float('temperature',0.1,'')
flags.DEFINE_float('cl_margin',0.4,'')
flags.DEFINE_float('cr_margin',0.,'')

flags.DEFINE_float('sobel_mag_thresh',0.14,'')
flags.DEFINE_float('sobel_pix_thresh',0.2,'')

flags.DEFINE_integer('num_prototypes',100,'')
flags.DEFINE_integer('min_pts',10,'')
flags.DEFINE_float('max_clust_size',0.2,'')
flags.DEFINE_string('selection_method','eom','')
flags.DEFINE_float('frac_per_img',0.1,'')
flags.DEFINE_string('cluster_metric','euc','')
flags.DEFINE_bool('update_b4_crop',False,'')

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
   
    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    for epoch in range(10):
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

            main_features = model.extract_feature_map(image_batch[0])
            embds = main_features.movedim(1,3).reshape(-1, FLAGS.embd_dim)

            sampled_indices = torch.randint(32*32,(FLAGS.batch_size, int(FLAGS.frac_per_img*32*32)),device=torch.device('cuda'))
            sampled_mask = torch.scatter(torch.zeros(FLAGS.batch_size,32*32,dtype=torch.long,device=torch.device('cuda')), 1, sampled_indices, 1).reshape(-1).bool() # (N,)

            sobel_mask = sobel_filter(image_batch[0]) # N
            final_mask = sobel_mask & sampled_mask
            embds = embds[final_mask]
            norm_embds = F.normalize(embds)
            with torch.no_grad():
                if FLAGS.cluster_metric == 'euc':
                    dists = torch.cdist(embds.unsqueeze(0),embds.unsqueeze(0))[0]**2
                elif FLAGS.cluster_metric == 'cosine':
                    dists = 1 - norm_embds @ norm_embds.T

                cl_labels, cl_probs = model.cluster_features(dists.cpu().numpy().astype(np.float64)) # (M/N,) (M/N,)
                cl,clust_freqs = np.unique(cl_labels, return_counts=True) # (K+1,), (K+1,)
                cl_mask = torch.tensor(cl_labels != -1, device=torch.device('cuda'))
                assert cl[0] == -1

            cl_labels = torch.tensor(cl_labels,device=torch.device('cuda'))
            proto_sims = norm_embds[cl_mask] @ model.prototypes.T # (M/N, num_prototypes)
            mean_sims = torch.scatter_add(torch.zeros(len(cl)-1,FLAGS.num_prototypes,device=torch.device('cuda')),0, \
                                          cl_labels[cl_labels != -1][:,None].tile(1,FLAGS.num_prototypes),proto_sims) / \
                                          torch.tensor(clust_freqs[1:,None],device=torch.device('cuda')) # (K,num_prototypes)

            row_ind,col_ind = linear_sum_assignment(-mean_sims.detach().cpu().numpy())
            if FLAGS.cl_margin > 0.:
                pos_mask = torch.scatter(torch.zeros(mean_sims.shape[0],mean_sims.shape[1],device=torch.device('cuda')),1,torch.tensor(col_ind,device=torch.device('cuda'))[:,None],1.)
                arc_sims = torch.where(pos_mask==1., torch.cos(torch.acos(mean_sims.clamp(min=-0.999)-0.001)+FLAGS.cl_margin), mean_sims)
                cl_loss = F.cross_entropy(arc_sims/FLAGS.temperature, torch.tensor(col_ind,device=torch.device('cuda')))
            else:
                cl_loss = F.cross_entropy(mean_sims/FLAGS.temperature, torch.tensor(col_ind,device=torch.device('cuda')))
            cl_loss.backward()

            if FLAGS.update_b4_crop:
                optimizer.step()
                optimizer.zero_grad()

            cl_idxs = torch.scatter(-torch.ones(final_mask.shape[0],dtype=torch.long,device=torch.device('cuda')), 0, final_mask.nonzero().reshape(-1), cl_labels) # (N,)
            noise_fracs = (cl_idxs.reshape(FLAGS.batch_size,32*32) == -1).sum(1) / (32*32)
            
            for cr_img,cr_dims in zip(image_batch[1:],crop_dims):
                crop_features = model.extract_feature_map(cr_img)
                crop_features = F.normalize(crop_features)
                
                cr_features,p_idxs = model.mask_crop_feats(crop_features,cr_dims,cl_idxs) # (M'/N,D), (M'/N,)
                cr_sims = cr_features @ model.prototypes.T # (M'/N,K)

                if FLAGS.cr_margin > 0.:
                    pos_mask = torch.scatter(torch.zeros(cr_sims.shape[0],cr_sims.shape[1],device=torch.device('cuda')),1,p_idxs[:,None],1.)
                    arc_sims = torch.where(pos_mask==1., torch.cos(torch.acos(cr_sims.clamp(min=-0.999)-0.001)+FLAGS.cr_margin), cr_sims)
                    crop_loss = F.cross_entropy(arc_sims/FLAGS.temperature, p_idxs)
                else:
                    crop_loss = F.cross_entropy(cr_sims/FLAGS.temperature, p_idxs)

                crop_loss.backward()

            log_dict = {"Epoch":epoch, "Iter":train_iter, "CL Loss": cl_loss, "Crop Loss": crop_loss, \
                    "Frac Masked": 1-sobel_mask.sum()/sobel_mask.shape[0], "Crop Frac Masked": 1-p_idxs.shape[0]/sobel_mask.shape[0], \
                    "Total Masked": 1-final_mask.sum()/final_mask.shape[0], "Avg Cluster Probability": cl_probs[cl_probs != 0].mean(), \
                    "Frac Noise Pts": clust_freqs[0]/cl_probs.shape[0], "Num Clusters": len(clust_freqs)-1, \
                    "Min Clust": clust_freqs[1:].min()/clust_freqs[1:].sum(), "Max Clust": clust_freqs[1:].max()/clust_freqs[1:].sum(), \
                    "Min Noise Pts": noise_fracs.min(), "Max Noise Pts": noise_fracs.max(), "Num Noise Images": (noise_fracs == 1.).sum()}
            
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

            if train_iter == 400:
                for g in optimizer.param_groups:
                    g['lr'] /= 10

        with torch.no_grad():
            val_iter = 0
            total_cl_loss = 0.
            total_cr_loss = 0.
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

                sampled_indices = torch.randint(32*32,(FLAGS.batch_size, int(FLAGS.frac_per_img*32*32)),device=torch.device('cuda'))
                sampled_mask = torch.scatter(torch.zeros(FLAGS.batch_size,32*32,dtype=torch.long,device=torch.device('cuda')), 1, sampled_indices, 1).reshape(-1).bool() # (N,)

                sobel_mask = sobel_filter(image_batch[0]) # N
                final_mask = sobel_mask & sampled_mask
                embds = embds[final_mask]
                norm_embds = F.normalize(embds)
                
                if FLAGS.cluster_metric == 'euc':
                    dists = torch.cdist(embds.unsqueeze(0),embds.unsqueeze(0))[0]**2
                elif FLAGS.cluster_metric == 'cosine':
                    dists = 1 - norm_embds @ norm_embds.T

                cl_labels, cl_probs = model.cluster_features(dists.cpu().numpy().astype(np.float64)) # (M/N,) (M/N,)
                cl,clust_freqs = np.unique(cl_labels, return_counts=True) # (K+1,), (K+1,)
                cl_mask = torch.tensor(cl_labels != -1, device=torch.device('cuda'))
                assert cl[0] == -1

                cl_labels = torch.tensor(cl_labels,device=torch.device('cuda'))
                proto_sims = norm_embds[cl_mask] @ model.prototypes.T # (M/N, num_prototypes)
                mean_sims = torch.scatter_add(torch.zeros(len(cl)-1,FLAGS.num_prototypes,device=torch.device('cuda')),0, \
                                              cl_labels[cl_labels != -1][:,None].tile(1,FLAGS.num_prototypes),proto_sims) / \
                                              torch.tensor(clust_freqs[1:,None],device=torch.device('cuda')) # (K,num_prototypes)

                row_ind,col_ind = linear_sum_assignment(-mean_sims.detach().cpu().numpy())
                cl_loss = F.cross_entropy(mean_sims/FLAGS.temperature, torch.tensor(col_ind,device=torch.device('cuda')))

                cl_idxs = torch.scatter(-torch.ones(final_mask.shape[0],dtype=torch.long,device=torch.device('cuda')), 0, final_mask.nonzero().reshape(-1), cl_labels) # (N,)
                
                for cr_img,cr_dims in zip(image_batch[1:],crop_dims):
                    crop_features = model.extract_feature_map(cr_img)
                    crop_features = F.normalize(crop_features)
                    
                    cr_features,p_idxs = model.mask_crop_feats(crop_features,cr_dims,cl_idxs) # (M'/N,D), (M'/N,)
                    cr_sims = cr_features @ model.prototypes.T # (M'/N,K)

                    crop_loss = F.cross_entropy(cr_sims/FLAGS.temperature, p_idxs)
                
                total_cl_loss += cl_loss
                total_cr_loss += crop_loss

                val_iter += 1

                '''annots = frames_load[2]
                miou, num_ious = calculate_iou(max_cluster_mask.to('cpu'), annots)
                total_miou += miou
                total_num_ious += num_ious'''


            avg_cl_loss = total_cl_loss/val_iter
            avg_cr_loss = total_cr_loss/val_iter
            log_dict = {"Epoch":epoch, "Val CL Loss": avg_cl_loss, "Val CR Loss": avg_cr_loss}
            
            print(log_dict)

            wandb.log(log_dict)

            if avg_cl_loss+avg_cr_loss < min_loss:
                torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                torch.save({'net':model.net,'proj_head':model.proj_head,'prototypes':model.prototypes},'weights/{}.pt'.format(FLAGS.exp))
                min_loss = avg_cl_loss+avg_cr_loss
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)