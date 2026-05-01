# -*- coding: utf8 -*-

import os
import sys
import argparse
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2
import math

from dataset.data_loader_vigor import VigorBuildingGroundDataset
from model.HisymGeo import HisymGeo
from model.loss import  yolo_loss, build_target, adjust_learning_rate, bbox_iou
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


from utils.utils import AverageMeter, eval_iou_acc
from utils.checkpoint import save_checkpoint, load_pretrain




def main():
    parser = argparse.ArgumentParser(
        description='cross-view object geo-localization')
    parser.add_argument('--gpu', default='2,3', help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers for data loading')

    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding dimensions')
    parser.add_argument('--img_size', default=640, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='./data', help='path to the root folder of all dataset')
    parser.add_argument('--data_name', default='CVOGL_DroneAerial', type=str, help='CVOGL_DroneAerial/CVOGL_SVI')
    
    parser.add_argument('--train_pth', required=True, type=str)
    parser.add_argument('--val_pth', required=True, type=str)
    parser.add_argument('--test_pth', required=True, type=str)
    
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')
    
    parser.add_argument('--RMSprop_or_AdamW', default=False)

    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
    

    # ---- 新增：task1/task2 是否使用双尺度 click-token query ----
    parser.add_argument('--use_dual_scale_query', default=True, type=lambda x: str(x).lower() in ['true','1','yes'],
                    help='task1/task2 使用 token16+32 gate 融合 (True) 或仅用 token32 (False)')



    

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    

    eps=1e-10
    
    # 锚点配置
    anchors = '137,82, 144,164, 479,243, 255,537, 73,202, 242,117, 175,359, 259,260, 74,108'
    args.anchors = anchors

    ## save logs
    if args.savename=='default':
        args.savename = '%s_batch%d' % (args.data_name, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    


    
    # 创建数据集
    train_dataset = VigorBuildingGroundDataset(
        data_root=args.data_root,
        split_pth=args.train_pth,
        img_size=args.img_size,
        augment=True,
    )
    val_dataset = VigorBuildingGroundDataset(
        data_root=args.data_root,
        split_pth=args.val_pth,
        img_size=args.img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    ## Model
    torch.cuda.empty_cache()
    model = HisymGeo(emb_size=args.emb_size)

    model = torch.nn.DataParallel(model).cuda()


    

    if args.pretrain:
        model = load_pretrain(model, args, logging)
    
    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    optimizer = torch.optim.AdamW([{'params': model.parameters()},], lr=args.lr, weight_decay=0.0005)
    
    ## training and testing
    best_acc = -float('Inf')  # 三视角任务的最佳准确率
    
   

    if args.test:
        # 测试模式
        test_dataset = VigorBuildingGroundDataset(
        data_root=args.data_root,
        split_pth=args.test_pth,
        img_size=args.img_size,
        augment=False,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        
        # 测试所有数据集

        svi_acc = test_single_epoch_ground_only(test_loader, model, args)
        

        
        print(f"测试结果:")
        print(f"  地面+卫星任务准确率: {svi_acc:.4f}")
        
        
        logging.info(f"测试结果: 地面+卫星={svi_acc:.4f}")
        
    elif args.val:
        # 验证模式
        
        svi_acc = test_single_epoch_ground_only(val_loader, model, args)
        
        print(f"验证结果:")
        
        print(f"  地面+卫星任务准确率: {svi_acc:.4f}")
        
        
        logging.info(f"验证结果: 地面+卫星={svi_acc:.4f}")
        
        return svi_acc  # 用于模型选择
    else:
        # 训练模式
        for epoch in range(args.max_epoch):
            
            adjust_learning_rate(args, optimizer, epoch)
            gc.collect()

         
            # 训练
            task_losses = train_epoch(
                train_loader, model, optimizer, epoch, args,
            )
            

            # 验证
            svi_acc = test_single_epoch_ground_only(val_loader, model, args)
            

            # 保存最佳模型 (基于三视角任务准确率和地面+卫星任务准确率)
            is_best = False
            if svi_acc > best_acc:
                is_best = True
                best_acc = svi_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accu': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, args, filename=args.savename)
            
            print(f"Epoch {epoch}结果:")
            # print(f"  三视角任务准确率: {triplet_acc:.4f} (最佳: {best_triplet_acc:.4f})")
            print(f"  地面+卫星任务准确率: {svi_acc:.4f}")
            
            
            logging.info(f"Epoch {epoch}: 地面+卫星={svi_acc:.4f}")






def train_epoch(train_loader, model, optimizer, epoch, args):
    
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    avg_accu_center = AverageMeter()

    model.train()
    end = time.time()

    # scaler = amp.GradScaler()  # 混合精度梯度缩放器

    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()
    

    for batch_idx, batch_data in enumerate(train_loader):

        # 解析数据
        ground_imgs, satellite_imgs, click_map, bbox, _ = batch_data
        ground_imgs = ground_imgs.cuda()
        satellite_imgs = satellite_imgs.cuda()
        click_map = click_map.cuda()
        bbox = bbox.cuda()
        ori_gt_bbox = torch.clamp(bbox, min=0, max=args.img_size - 1)


        # 随机丢弃视角
        batch_size = ground_imgs.size(0)
        B = ground_imgs.size(0)
        
        # 前向传播
        pred_task1, _, _, _, _, _ = model(
            query_imgs=ground_imgs,
            reference_imgs=satellite_imgs,
            mat_clickptns=click_map,
            training=True
        )

        
        # 计算任务损失
        loss1 = calculate_task_loss(
            pred_task1, ori_gt_bbox, anchors_full, args.img_size, args
        )
        
        
        optimizer.zero_grad()
        

        # 1. 正常反向传播
        loss1.backward()


        # # 梯度裁剪防止爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 记录损失
        
        avg_loss1.update(loss1.item(), batch_size)
        
        
        pred_task1 = pred_task1.view(pred_task1.shape[0], 9, 5, pred_task1.shape[2], pred_task1.shape[3])
        new_gt_bbox_1, best_anchor_gi_gj_1 = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_task1.shape[3])
        accu_list1, accu_center, iou1, _, _, _ = eval_iou_acc(pred_task1, ori_gt_bbox, anchors_full, best_anchor_gi_gj_1[:, 1], best_anchor_gi_gj_1[:, 2], args.img_size, iou_threshold_list=[0.5, 0.25])
        
        ## metrics
        avg_iou.update(iou1, batch_size)
        avg_accu50.update(accu_list1[0], batch_size)
        avg_accu25.update(accu_list1[1], batch_size)

        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:

            print_str = (
                f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                f'Time {batch_time.avg:.3f}\t'
                f'Loss {avg_loss1.avg:.4f}\t'
                f'Acc1@50 {(avg_accu50.avg):.4f}|Accu25 {avg_accu25.val:.4f}|IoU1 {avg_iou.avg:.4f}\t'
                
            )
            # *** 修改结束 ***
            print(print_str)
            logging.info(print_str)

    # 返回epoch平均损失
    return avg_loss1.avg
    


def calculate_task_loss(pred, target_bbox, anchors_full, image_wh, args):
    """计算单个任务的损失"""
    
    pred = pred.view(pred.shape[0], 9, 5, pred.shape[2], pred.shape[3])
    new_gt_bbox, best_anchor_gi_gj = build_target(
        target_bbox, anchors_full, image_wh, pred.shape[3]
    )
    loss_geo, loss_cls = yolo_loss(pred, new_gt_bbox, anchors_full, best_anchor_gi_gj, image_wh)
    base_loss = loss_cls + loss_geo * args.beta
    
    return base_loss



@torch.no_grad()
def test_single_epoch_ground_only(data_loader, model, args):
    batch_time = AverageMeter()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    avg_accu_center = AverageMeter()

    model.eval()
    device = next(model.parameters()).device
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()
    end = time.time()

    for batch_idx, batch_data in enumerate(data_loader):
        ground_imgs, satellite_imgs, click_map, bbox, _ = batch_data
        ground_imgs = ground_imgs.to(device, non_blocking=True)
        satellite_imgs = satellite_imgs.to(device, non_blocking=True)
        click_map = click_map.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        bbox = torch.clamp(bbox, min=0, max=args.img_size - 1)

        pred_anchor, _, _, _, _, _ = model(
            query_imgs=ground_imgs,
            reference_imgs=satellite_imgs,
            mat_clickptns=click_map,
            training=False
        )
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])

        _, best_anchor_gi_gj = build_target(bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(
            pred_anchor, bbox, anchors_full,
            best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2],
            args.img_size, iou_threshold_list=[0.5, 0.25]
        )

        bs = bbox.size(0)
        avg_accu50.update(accu_list[0], bs)
        avg_accu25.update(accu_list[1], bs)
        avg_iou.update(iou, bs)
        avg_accu_center.update(accu_center, bs)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = (
                '[{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Accu50 {accu50.val:.4f} ({accu50.avg:.4f})\t'
                'Accu25 {accu25.val:.4f} ({accu25.avg:.4f})\t'
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t'
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})'
            ).format(
                batch_idx, len(data_loader), batch_time=batch_time,
                accu50=avg_accu50, accu25=avg_accu25, miou=avg_iou, accu_c=avg_accu_center
            )
            print(print_str)
            logging.info(print_str)

    print(avg_accu50.avg, avg_accu25.avg, avg_iou.avg, avg_accu_center.avg)
    logging.info('%f, %f, %f, %f' % (avg_accu50.avg, avg_accu25.avg, float(avg_iou.avg), avg_accu_center.avg))
    return avg_accu50.avg




if __name__ == "__main__":
    main()
