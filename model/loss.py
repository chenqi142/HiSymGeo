# -*- coding:utf8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import bbox_iou, xyxy2xywh

def adjust_learning_rate(args, optimizer, i_iter):
    
    lr = args.lr*((0.1)**(i_iter//10))
        
    print(("lr", lr))
    for param_idx, param in enumerate(optimizer.param_groups):
        param['lr'] = lr

# the shape of the target is (batch_size, anchor_count, 5, grid_wh, grid_wh)
def yolo_loss(predictions, gt_bboxes, anchors_full, best_anchor_gi_gj, image_wh):
    batch_size, grid_stride = predictions.shape[0], image_wh // predictions.shape[3]
    best_anchor, gi, gj = best_anchor_gi_gj[:, 0], best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2]
    scaled_anchors = anchors_full / grid_stride
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss_confidence = torch.nn.CrossEntropyLoss(size_average=True)
    #celoss_cls = torch.nn.CrossEntropyLoss(size_average=True)

    selected_predictions = predictions[range(batch_size), best_anchor, :, gj, gi]

    #---bbox loss---
    pred_bboxes = torch.zeros_like(gt_bboxes)
    pred_bboxes[:, 0:2] = selected_predictions[:, 0:2].sigmoid()
    pred_bboxes[:, 2:4] = selected_predictions[:, 2:4]
    
    loss_x = mseloss(pred_bboxes[:,0], gt_bboxes[:,0])
    loss_y = mseloss(pred_bboxes[:,1], gt_bboxes[:,1])
    loss_w = mseloss(pred_bboxes[:,2], gt_bboxes[:,2])
    loss_h = mseloss(pred_bboxes[:,3], gt_bboxes[:,3])

    loss_bbox = loss_x + loss_y + loss_w + loss_h

    #---confidence loss---
    pred_confidences = predictions[:,:,4,:,:]
    gt_confidences = torch.zeros_like(pred_confidences)
    gt_confidences[range(batch_size), best_anchor, gj, gi] = 1
    pred_confidences, gt_confidences = pred_confidences.reshape(batch_size, -1), \
                    gt_confidences.reshape(batch_size, -1)
    loss_confidence = celoss_confidence(pred_confidences, gt_confidences.max(1)[1])

    return loss_bbox, loss_confidence

#target_coord:batch_size, 5
def build_target(ori_gt_bboxes, anchors_full, image_wh, grid_wh):
    #the default value of coord_dim is 5
    batch_size, coord_dim, grid_stride, anchor_count = ori_gt_bboxes.shape[0], ori_gt_bboxes.shape[1], image_wh//grid_wh, anchors_full.shape[0]
    
    gt_bboxes = xyxy2xywh(ori_gt_bboxes)
    gt_bboxes = (gt_bboxes/image_wh) * grid_wh
    scaled_anchors = anchors_full/grid_stride

    gxy = gt_bboxes[:, 0:2]
    gwh = gt_bboxes[:, 2:4]
    gij = gxy.long()

    #get the best anchor for each target bbox
    gt_bboxes_tmp, scaled_anchors_tmp = torch.zeros_like(gt_bboxes), torch.zeros((anchor_count, coord_dim), device=gt_bboxes.device)
    gt_bboxes_tmp[:, 2:4] = gwh
    gt_bboxes_tmp = gt_bboxes_tmp.unsqueeze(1).repeat(1, anchor_count, 1).view(-1, coord_dim)
    scaled_anchors_tmp[:, 2:4] = scaled_anchors
    scaled_anchors_tmp = scaled_anchors_tmp.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, coord_dim)
    anchor_ious = bbox_iou(gt_bboxes_tmp, scaled_anchors_tmp).view(batch_size, -1)
    best_anchor=torch.argmax(anchor_ious, dim=1)
    
    twh = torch.log(gwh / scaled_anchors[best_anchor] + 1e-16)
    #print((gxy.dtype, gij.dtype, twh.dtype, gwh.dtype, scaled_anchors.dtype, 'inner'))
    #print((gxy.shape, gij.shape, twh.shape, gwh.shape), flush=True)
    #print(('gxy,gij,twh', gxy, gij, twh), flush=True)
    return torch.cat((gxy - gij, twh), 1), torch.cat((best_anchor.unsqueeze(1), gij), 1)

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=image_features1.device)

        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    #pdb.set_trace()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss