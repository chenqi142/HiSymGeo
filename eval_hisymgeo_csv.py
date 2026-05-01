import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize

# =========================================================
# TODO: 改成你工程真实路径
from model.HisymGeo import HisymGeo
# =========================================================


def load_pretrain(model, pretrain_path):
    if os.path.isfile(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}".format(pretrain_path))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(pretrain_path)))
    return model


def generate_encoding(query_featuremap_hw, click_hw, sigma=1.0):
    """生成 click 位置编码矩阵"""
    mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
    click_h = [pow((one - click_hw[0]) / sigma, 2) for one in range(query_featuremap_hw[0])]
    click_w = [pow((one - click_hw[1]) / sigma, 2) for one in range(query_featuremap_hw[1])]
    norm_hw = pow(query_featuremap_hw[0]**2 + query_featuremap_hw[1]**2, 0.5)
    for i in range(query_featuremap_hw[0]):
        for j in range(query_featuremap_hw[1]):
            tmp_val = 1 - (pow(click_h[i] + click_w[j], 0.5) / norm_hw)
            mat_clickhw[i, j] = tmp_val * tmp_val
    return mat_clickhw


@torch.no_grad()
def decode_pred_bbox(pred_anchor_reshaped, anchors_full, image_wh):
    """
    解码预测框
    pred_anchor_reshaped: [1, 9, 5, S, S]
    anchors_full: [9, 2] pixel尺度(w,h)
    image_wh: 卫星图尺寸（默认1024）
    """
    assert pred_anchor_reshaped.dim() == 5 and pred_anchor_reshaped.shape[0] == 1
    _, A, _, S, _ = pred_anchor_reshaped.shape
    grid_stride = image_wh // S

    pred_conf_logit = pred_anchor_reshaped[0, :, 4, :, :]  # [A, S, S]
    idx = int(torch.argmax(pred_conf_logit.reshape(-1)).item())

    ss = S * S
    best_n = idx // ss
    rem = idx % ss
    gj = rem // S
    gi = rem % S

    tx = pred_anchor_reshaped[0, best_n, 0, gj, gi]
    ty = pred_anchor_reshaped[0, best_n, 1, gj, gi]
    tw = pred_anchor_reshaped[0, best_n, 2, gj, gi]
    th = pred_anchor_reshaped[0, best_n, 3, gj, gi]
    tobj = pred_anchor_reshaped[0, best_n, 4, gj, gi]

    bx = (tx.sigmoid() + gi) * grid_stride
    by = (ty.sigmoid() + gj) * grid_stride
    bw = torch.exp(tw) * anchors_full[best_n, 0]
    bh = torch.exp(th) * anchors_full[best_n, 1]

    x1 = float((bx - 0.5 * bw).clamp(0, image_wh - 1).item())
    y1 = float((by - 0.5 * bh).clamp(0, image_wh - 1).item())
    x2 = float((bx + 0.5 * bw).clamp(0, image_wh - 1).item())
    y2 = float((by + 0.5 * bh).clamp(0, image_wh - 1).item())

    conf = float(tobj.sigmoid().item())
    return [x1, y1, x2, y2], conf, int(best_n), int(gi), int(gj), S


def compute_iou(box1, box2):
    """
    计算两个框的IoU
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def main():
    parser = argparse.ArgumentParser(description="HiSymGeo 评估脚本 (CSV数据源)")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--query_root", type=str, required=True, help="Query 图片目录")
    parser.add_argument("--sat_root", type=str, required=True, help="Satellite 图片目录")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--dp_ids", type=str, default="2,3")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--img_size", type=int, default=1024, help="Satellite 图像尺寸")
    parser.add_argument("--query_size", type=int, default=256, help="Query 图像尺寸（用于 click 编码）")
    parser.add_argument("--anchors", type=str,
                        default="37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573",
                        help="预设锚点")
    parser.add_argument("--iou_thresh", type=str, default="0.50,0.25",
                        help="IoU阈值列表，逗号分隔")
    parser.add_argument("--save_results", type=str, default=None,
                        help="可选：保存每条结果的CSV路径")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=> using device: {device}")

    # ---- 解析锚点 ----
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(",")], dtype=np.float32)
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).to(device)

    # ---- 解析IoU阈值 ----
    iou_thresholds = [float(x.strip()) for x in args.iou_thresh.split(",")]
    print(f"=> IoU thresholds: {iou_thresholds}")

    # ---- 加载模型 ----
    model = HisymGeo().to(device)
    if args.dp:
        ids = [int(x) for x in args.dp_ids.split(",")]
        model = torch.nn.DataParallel(model, device_ids=ids)
    model.eval()
    model = load_pretrain(model, args.ckpt)
    print("=> model loaded")

    # ---- 图像预处理 ----
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- 读取CSV ----
    df = pd.read_csv(args.csv_path)
    print(f"=> loaded {len(df)} samples from {args.csv_path}")

    # ---- 评估 ----
    results = []
    total_iou = 0.0
    success_counts = {thresh: 0 for thresh in iou_thresholds}

    for idx, row in df.iterrows():
        query_name = row['query_name']
        sat_name = row['satellite_name']

        # 解析 click_point: "x,y" -> (x, y)
        click_str = str(row['click_point']).strip()
        click_x, click_y = map(int, click_str.split(','))

        # 解析 gt_bbox: "x1,y1,x2,y2"
        gt_str = str(row['gt_bbox']).strip()
        gt_bbox = list(map(float, gt_str.split(',')))

        q_path = os.path.join(args.query_root, query_name + ".png")
        s_path = os.path.join(args.sat_root, sat_name + ".png")

        # 检查文件
        if not os.path.exists(q_path):
            print(f"[WARN] query not found: {q_path}, skip")
            continue
        if not os.path.exists(s_path):
            print(f"[WARN] satellite not found: {s_path}, skip")
            continue

        # 读取图片
        q_bgr = cv2.imread(q_path)
        s_bgr = cv2.imread(s_path)
        if q_bgr is None or s_bgr is None:
            print(f"[WARN] failed to load images for {query_name}, skip")
            continue

        q_rgb = cv2.cvtColor(q_bgr, cv2.COLOR_BGR2RGB)
        s_rgb = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)

        # ---- 生成 click 编码 ----
        query_featuremap_hw = (args.query_size, args.query_size)
        click_point = (click_y, click_x)  # generate_encoding 用 (h, w) 即 (y, x)
        mat_clickptns = generate_encoding(query_featuremap_hw, click_point)
        mat_clickptns = torch.tensor(mat_clickptns).unsqueeze(0).to(device)  # [1, H, W]

        # ---- 准备输入张量 ----
        q_t = transform(q_rgb.copy()).unsqueeze(0).to(device)
        s_t = transform(s_rgb.copy()).unsqueeze(0).to(device)

        # ---- 前向推理 ----
        with torch.inference_mode():
            outbox, attn_score, query_g, ref1, ref2, ref3 = model(q_t, s_t, mat_clickptns, False)

        # outbox: [1, 45, S, S] -> [1, 9, 5, S, S]
        pred_anchor = outbox.view(outbox.shape[0], 9, 5, outbox.shape[2], outbox.shape[3])

        # ---- 解码预测框 ----
        pred_xyxy, conf, best_n, gi, gj, S = decode_pred_bbox(pred_anchor, anchors_full, args.img_size)

        # ---- 计算IoU ----
        iou = compute_iou(pred_xyxy, gt_bbox)
        total_iou += iou

        # 检查各阈值
        for thresh in iou_thresholds:
            if iou >= thresh:
                success_counts[thresh] += 1

        result = {
            'id': row.get('id', idx),
            'query_name': query_name,
            'satellite_name': sat_name,
            'pred_bbox': f"{pred_xyxy[0]:.1f},{pred_xyxy[1]:.1f},{pred_xyxy[2]:.1f},{pred_xyxy[3]:.1f}",
            'gt_bbox': gt_str,
            'iou': round(iou, 4),
            'conf': round(conf, 4),
        }
        results.append(result)

        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"  processed {idx+1}/{len(df)} | IoU={iou:.4f}")

    # ---- 计算最终指标 ----
    n = len(results)
    if n == 0:
        print("[ERROR] no valid samples evaluated!")
        return

    miou = total_iou / n

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total samples: {n}")
    print(f"mIoU: {miou:.4f}")
    for thresh in iou_thresholds:
        acc = success_counts[thresh] / n
        print(f"acc@{thresh:.2f}: {acc:.4f} ({success_counts[thresh]}/{n})")
    print("=" * 50)

    # ---- 可选保存详细结果 ----
    if args.save_results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.save_results, index=False)
        print(f"=> detailed results saved to {args.save_results}")


if __name__ == "__main__":
    main()
