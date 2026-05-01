# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
# from shapely.geometry import Polygon
import math
from torchvision.transforms import Compose, ToTensor, Normalize
import xml.etree.ElementTree as ET

cv2.setNumThreads(0)
    
class DatasetNotFoundError(Exception):
    pass

class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
                albumentations.Blur(p=0.01),
                albumentations.MedianBlur(p=0.01),
                albumentations.ToGray(p=0.01),
                albumentations.CLAHE(p=0.01),
                albumentations.RandomBrightnessContrast(p=0.0),
                albumentations.RandomGamma(p=0.0),
                albumentations.ImageCompression(quality_lower=75, p=0.0)])
    
    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh,imgw, _ = img.shape
        x, y, w, h = (bbox[0]+bbox[2])/2/imgw, (bbox[1]+bbox[3])/2/imgh, (bbox[2]-bbox[0])/imgw, (bbox[3]-bbox[1])/imgh
        img = self.transform(image=img)['image']
        #self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1-y
            
        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1-x
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh==imgh, new_imgw==imgw
        x, y, w, h = x*imgw, y*imgh, w*imgw, h*imgh

        # Crop image
        iscropped=False
        if random.random() < 0.5:
            left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
            if left >= new_imgw/2:
                start_cropped_x = random.randint(0, int(0.15*new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw/2:
                start_cropped_x = random.randint(int(0.85*new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh/2:
                start_cropped_y = random.randint(0, int(0.15*new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh/2:
                start_cropped_y = random.randint(int(0.85*new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left/cropped_imgw, top/cropped_imgh, right/cropped_imgw, bottom/cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left*new_cropped_imgw, top*new_cropped_imgh, right*new_cropped_imgw, bottom*new_cropped_imgh 
            x, y, w, h = (left+right)/2, (top+bottom)/2, right-left, bottom-top
            iscropped=True
        #if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        #print(type(img))
        #draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        #print(('draw_bbox', iscropped, draw_bbox), flush=True)
        #img_new=draw_rectangle(img, draw_bbox)
        #cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x-w/2), y-h/2, x+w/2, y+h/2]
        #print(bbox)
        #print(new_bbox)
        #print('---end---')
        return img, np.array(new_bbox, dtype=int)

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

#可视化
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img






IMAGENET_NORM = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class VigorBuildingGroundDataset(Dataset):
    def __init__(self, data_root, split_pth, img_size=640, ground_size=(512, 256), transform=None, augment=False):
        self.data_root = data_root
        self.data_list = torch.load(split_pth)
        self.sat_size = img_size
        self.ground_w, self.ground_h = ground_size
        self.transform = transform if transform is not None else IMAGENET_NORM
        self.augment = augment

        self.rs_transform = albumentations.Compose([
            albumentations.RandomSizedBBoxSafeCrop(width=img_size, height=img_size, erosion_rate=0.2, p=0.2),
            albumentations.RandomRotate90(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.OneOf([
                albumentations.Blur(p=0.4),
                albumentations.MedianBlur(p=0.3),
            ], p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(p=0.4),
                albumentations.CLAHE(p=0.3),
            ], p=0.5),
            albumentations.ToGray(p=0.2),
            albumentations.RandomGamma(p=0.3),
        ], bbox_params=albumentations.BboxParams(format='pascal_voc'),
        
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        city, g_img_name, g_xml_name, s_img_name, s_xml_name, target_idx = self.data_list[idx]

        g_img_path = os.path.join(self.data_root, city, 'query', 'images', g_img_name)
        g_xml_path = os.path.join(self.data_root, city, 'query', 'labels', g_xml_name)
        s_img_path = os.path.join(self.data_root, city, 'satellite', 'images', s_img_name)
        s_xml_path = os.path.join(self.data_root, city, 'satellite', 'labels', s_xml_name)

        g_img = self._read_rgb(g_img_path)
        s_img = self._read_rgb(s_img_path)

        g_ann = self._parse_voc_xml(g_xml_path)   # 原始应为 2048x1024
        s_ann = self._parse_voc_xml(s_xml_path)   # 原始应为 640x640

        target_name = self._resolve_target_name(target_idx, g_ann['objects'], s_ann['objects'])
        g_box = self._find_box_by_name(g_ann['objects'], target_name)
        s_box = self._find_box_by_name(s_ann['objects'], target_name)

        # resize image
        g_img_rs = cv2.resize(g_img, (self.ground_w, self.ground_h), interpolation=cv2.INTER_LINEAR)
        s_img_rs = cv2.resize(s_img, (self.sat_size, self.sat_size), interpolation=cv2.INTER_LINEAR)

        # scale bbox
        g_box_rs = self._scale_box(g_box, g_ann['width'], g_ann['height'], self.ground_w, self.ground_h)
        s_box_rs = self._scale_box(s_box, s_ann['width'], s_ann['height'], self.sat_size, self.sat_size)

        # 数据增强
        if self.augment:
            # 卫星图像增强
            try:
                rs_transformed = self.rs_transform(
                    image=s_img_rs, 
                    bboxes=[s_box_rs.tolist()],
                )
                s_img_rs = rs_transformed['image']

                s_box_rs = np.array(rs_transformed['bboxes'][0][0:4], dtype=np.float32)
            except Exception as e:
                print(f"卫星图像增强失败: {e}")

        # click from resized ground bbox center
        cx = (g_box_rs[0] + g_box_rs[2]) * 0.5
        cy = (g_box_rs[1] + g_box_rs[3]) * 0.5
        click_map = self._create_click_map((cx, cy), (self.ground_h, self.ground_w))

        g_img_ts = self.transform(g_img_rs.copy())
        s_img_ts = self.transform(s_img_rs.copy())
        click_map_ts = torch.tensor(click_map, dtype=torch.float32)
        s_box_ts = torch.tensor(s_box_rs, dtype=torch.float32)

        return g_img_ts, s_img_ts, click_map_ts, s_box_ts, idx

    @staticmethod
    def _read_rgb(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _parse_voc_xml(xml_path):
        root = ET.parse(xml_path).getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
        return {'width': width, 'height': height, 'objects': objects}

    @staticmethod
    def _resolve_target_name(target_idx, ground_objects, sat_objects):
        target_name = str(target_idx)
        ground_names = {obj['name'] for obj in ground_objects}
        sat_names = {obj['name'] for obj in sat_objects}
        common = ground_names & sat_names

        if target_name in common:
            return target_name

        if isinstance(target_idx, (int, np.integer)):
            if 1 <= int(target_idx) <= len(ground_objects):
                name_1b = ground_objects[int(target_idx) - 1]['name']
                if name_1b in common:
                    return name_1b
            if 0 <= int(target_idx) < len(ground_objects):
                name_0b = ground_objects[int(target_idx)]['name']
                if name_0b in common:
                    return name_0b

        raise KeyError(f'Cannot resolve target_idx={target_idx}')

    @staticmethod
    def _find_box_by_name(objects, target_name):
        for obj in objects:
            if obj['name'] == str(target_name):
                return np.array(obj['bbox'], dtype=np.float32)
        raise KeyError(f'Object name {target_name} not found.')

    @staticmethod
    def _scale_box(box, src_w, src_h, dst_w, dst_h):
        x1, y1, x2, y2 = box.astype(np.float32)
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)

    @staticmethod
    def _create_click_map(click_xy, hw):
        h, w = hw
        cx, cy = click_xy
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
        diag = np.sqrt(w * w + h * h) + 1e-6
        mat = 1.0 - dist / diag
        mat = np.clip(mat, 0.0, 1.0)
        return (mat * mat).astype(np.float32)