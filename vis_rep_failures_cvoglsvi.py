import logging
import os
import math
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        print("=> loaded pretrain model at {}"
              .format(pretrain_path))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(pretrain_path)))
    return model

# -------------------------
# 1) 生成 mat_clickhw（与 __getitem__ 同公式，向量化加速）
# -------------------------
def build_mat_clickhw(click_xy_img, q_h, q_w, query_featuremap_hw, sigma=1.0):
    """
    click_xy_img: (x,y) in query ORIGINAL pixels
    q_h,q_w: query image shape
    query_featuremap_hw: (Hf,Wf) == mat_clickhw shape, 这里我们设为 (q_h,q_w)
                         因为 combine_clickptns 需要 mat 与 query_imgs 同尺寸。
    return: np.float32 [Hf,Wf]
    """
    Hf, Wf = query_featuremap_hw
    x_img, y_img = float(click_xy_img[0]), float(click_xy_img[1])
    x_img = float(np.clip(x_img, 0, q_w - 1))
    y_img = float(np.clip(y_img, 0, q_h - 1))

    # 将原图像素 click 映射到 mat 坐标系（如果 Hf,Wf==q_h,q_w 则等价）
    x = x_img * (Wf / float(q_w))
    y = y_img * (Hf / float(q_h))
    x = int(np.clip(int(round(x)), 0, Wf - 1))
    y = int(np.clip(int(round(y)), 0, Hf - 1))

    norm_hw = math.sqrt(Hf * Hf + Wf * Wf)

    ih = (np.arange(Hf, dtype=np.float32) - y) / sigma
    jw = (np.arange(Wf, dtype=np.float32) - x) / sigma
    click_h = ih * ih
    click_w = jw * jw

    dist = np.sqrt(click_h[:, None] + click_w[None, :])
    tmp = 1.0 - dist / (norm_hw + 1e-12)
    mat = tmp * tmp
    return mat.astype(np.float32)

# -------------------------
# 2) 与 eval_iou_acc 完全一致的解码：全局 argmax(conf logit) + 同公式
# -------------------------
@torch.no_grad()
def decode_pred_bbox_evalstyle(pred_anchor_reshaped, anchors_full, image_wh):
    """
    pred_anchor_reshaped: [1, 9, 5, S, S]
    anchors_full: [9,2] pixel尺度(w,h)，按 test_epoch reshape(-1,2)[::-1]
    image_wh: args.img_size（1024）
    return:
      pred_xyxy(list[4]) in [0,image_wh)
      conf_sigmoid(float)
      best_n, gi, gj(int)
    """
    assert pred_anchor_reshaped.dim() == 5 and pred_anchor_reshaped.shape[0] == 1
    _, A, _, S, _ = pred_anchor_reshaped.shape
    grid_stride = image_wh // S

    pred_conf_logit = pred_anchor_reshaped[0, :, 4, :, :]     # [A,S,S]
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

def box_iou_xyxy(b1, b2, eps=1e-6):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2]-b1[0]) * max(0.0, b1[3]-b1[1])
    a2 = max(0.0, b2[2]-b2[0]) * max(0.0, b2[3]-b2[1])
    return inter / (a1 + a2 - inter + eps)

def center_err_px(b1, b2):
    cx1 = 0.5*(b1[0]+b1[2]); cy1 = 0.5*(b1[1]+b1[3])
    cx2 = 0.5*(b2[0]+b2[2]); cy2 = 0.5*(b2[1]+b2[3])
    return math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

# -------------------------
# 3) 可视化：Query红点；Satellite Pred绿框 / GT红框
# -------------------------
def save_query_only(out_path, q_rgb, click_xy_img, radius=8):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(q_rgb)
    ax.axis("off")
    if click_xy_img is not None:
        ax.add_patch(patches.Circle((click_xy_img[0], click_xy_img[1]), radius=radius, color="red"))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_satellite_only(out_path, s_rgb, pred_xyxy, gt_xyxy):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(s_rgb)
    ax.axis("off")

    # Pred green
    px1, py1, px2, py2 = pred_xyxy
    ax.add_patch(patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                   fill=False, linewidth=2, edgecolor="lime"))

    # GT red
    gx1, gy1, gx2, gy2 = gt_xyxy
    ax.add_patch(patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                   fill=False, linewidth=2, edgecolor="red"))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# -------------------------
# 4) 5桶 * 2：在线选“代表性”样例
# -------------------------
BUCKETS = ["highconf_wrong", "scale_aspect", "near_miss", "small_object", "center_far"]

class BucketTopK:
    def __init__(self, k=2):
        self.k = k
        self.store = {b: [] for b in BUCKETS}  # list[(score, meta)]

    def full(self):
        return all(len(v) >= self.k for v in self.store.values())

    def score(self, bucket, iou, conf, c_err, area, img_size):
        if bucket == "highconf_wrong":
            return conf*2.0 + (0.1 - iou)*10.0
        if bucket == "scale_aspect":
            return (0.05*img_size - c_err) + (0.25 - iou)*10.0
        if bucket == "near_miss":
            return -abs(iou - 0.5)
        if bucket == "small_object":
            return (0.02*img_size*img_size - area) + (0.5 - iou)*5.0
        if bucket == "center_far":
            return c_err + (0.25 - iou)*2.0
        return 0.0

    def push(self, bucket, score, meta):
        lst = self.store[bucket]
        lst.append((score, meta))
        lst.sort(key=lambda x: x[0], reverse=True)
        if len(lst) > self.k:
            lst.pop()

def generate_encoding(query_featuremap_hw, click_hw):
    mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
    sigma = 1.0
    click_h = [pow((one-click_hw[0]) / sigma,2) for one in range(query_featuremap_hw[0])]
    click_w = [pow((one-click_hw[1]) / sigma,2) for one in range(query_featuremap_hw[1])]
    norm_hw = pow(query_featuremap_hw[0]*query_featuremap_hw[0] + query_featuremap_hw[1]*query_featuremap_hw[1], 0.5)
    for i in range(query_featuremap_hw[0]):
        for j in range(query_featuremap_hw[1]):
            tmp_val = 1 - (pow(click_h[i]+click_w[j], 0.5)/norm_hw)
            mat_clickhw[i, j] = tmp_val * tmp_val

    return mat_clickhw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data0/chenqi_data/data/CVOGL_DroneAerial/CVOGL_DroneAerial_test.pth")
    parser.add_argument("--query_root", type=str, default="/data0/chenqi_data/data/CVOGL_DroneAerial/query")
    parser.add_argument("--sat_root", type=str, default="/data0/chenqi_data/data/CVOGL_DroneAerial/satellite")
    parser.add_argument("--ckpt", type=str, default="saved_models_20251124/model_droneaerial_model_best.pth.tar")
    parser.add_argument("--save_dir", type=str, default="vis_rep_failures_DroneAerial")

    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--dp_ids", type=str, default="2,3")

    parser.add_argument("--anchors", type=str, default="37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573")
    parser.add_argument("--img_size", type=int, default=1024)  # satellite is 1024x1024
    parser.add_argument("--per_bucket", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--conf_thr_highconf", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)

    # anchors parse (same as your test_epoch)
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(",")], dtype=np.float32)
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).to(device)

    # model
    model = HisymGeo().to(device)
    if args.dp:
        ids = [int(x) for x in args.dp_ids.split(",")]
        model = torch.nn.DataParallel(model, device_ids=ids)
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[2, 3])

    model = load_pretrain(model, "saved_models_20251124/model_droneaerial_model_best.pth.tar")

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    data = torch.load(args.data_path)
    topk = BucketTopK(k=args.per_bucket)

    for i in range(len(data)):
        if args.max_samples is not None and i >= args.max_samples:
            break
        if topk.full():
            break

        # format: _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name
        _, qname, sname, _, click_xy, bbox, _, cls_name = data[i]

        q_path = os.path.join(args.query_root, qname)
        s_path = os.path.join(args.sat_root, sname)
        if not (os.path.exists(q_path) and os.path.exists(s_path)):
            continue

        # read original images (cv2 BGR->RGB)
        q_bgr = cv2.imread(q_path)
        s_bgr = cv2.imread(s_path)
        if q_bgr is None or s_bgr is None:
            continue
        q_rgb = cv2.cvtColor(q_bgr, cv2.COLOR_BGR2RGB)
        s_rgb = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)

        q_h, q_w = q_rgb.shape[0], q_rgb.shape[1]  # ground: 256x512, drone: 256x256
        s_h, s_w = s_rgb.shape[0], s_rgb.shape[1]  # 1024x1024

        # # click_xy is ORIGINAL pixel coord (x,y)
        # click_xy = np.array(click_xy, dtype=np.float32).reshape(-1).tolist()
        # click_img_xy = (float(np.clip(click_xy[0], 0, q_w-1)),
        #                 float(np.clip(click_xy[1], 0, q_h-1)))

        # # mat_clickhw must match query image spatial size for fusion module
        # mat_click = build_mat_clickhw(click_img_xy, q_h, q_w, query_featuremap_hw=(q_h, q_w), sigma=1.0)
        # mat_click_t = torch.from_numpy(mat_click).unsqueeze(0).to(device)  # [1,H,W]
        query_featuremap_hw = (256, 256)
        click_point = (int(click_xy[0]), int(click_xy[1]))  # (x, y)
        mat_clickptns = generate_encoding(query_featuremap_hw, click_point)
        mat_clickptns = torch.tensor(mat_clickptns)
        mat_clickptns = mat_clickptns.unsqueeze(0).to(device)

        # image tensors (same Normalize)
        q_t = transform(q_rgb.copy()).unsqueeze(0).to(device)
        s_t = transform(s_rgb.copy()).unsqueeze(0).to(device)

        # GT bbox in satellite pixels (xyxy)
        gt_xyxy = [float(v) for v in np.array(bbox, dtype=np.float32).tolist()]
        # clamp
        gt_xyxy = [
            float(np.clip(gt_xyxy[0], 0, args.img_size-1)),
            float(np.clip(gt_xyxy[1], 0, args.img_size-1)),
            float(np.clip(gt_xyxy[2], 0, args.img_size-1)),
            float(np.clip(gt_xyxy[3], 0, args.img_size-1)),
        ]

        # forward (training=False)
        with torch.inference_mode():
            outbox, attn_score, query_g, ref1, ref2, ref3 = model(q_t, s_t, mat_clickptns, False)

        # outbox: [1, 45, S, S] -> [1,9,5,S,S]
        pred_anchor = outbox.view(outbox.shape[0], 9, 5, outbox.shape[2], outbox.shape[3])

        # decode (eval_iou_acc style)
        pred_xyxy, conf, best_n, gi, gj, S = decode_pred_bbox_evalstyle(pred_anchor, anchors_full, args.img_size)

        # metrics
        iou = box_iou_xyxy(pred_xyxy, gt_xyxy)
        c_err = center_err_px(pred_xyxy, gt_xyxy)
        area = max(0.0, (gt_xyxy[2]-gt_xyxy[0])) * max(0.0, (gt_xyxy[3]-gt_xyxy[1]))

        # 5 buckets
        buckets = []
        if (iou < 0.10) and (conf > args.conf_thr_highconf):
            buckets.append("highconf_wrong")
        if (c_err < 0.05 * args.img_size) and (iou < 0.25):
            buckets.append("scale_aspect")
        if 0.25 <= iou < 0.50:
            buckets.append("near_miss")
        if (area < 0.02 * args.img_size * args.img_size) and (iou < 0.50):
            buckets.append("small_object")
        if (c_err > 0.15 * args.img_size) and (iou < 0.25):
            buckets.append("center_far")

        if not buckets:
            continue

        cname = cls_name if isinstance(cls_name, str) else str(cls_name)
        title = f"cls={cname} | IoU={iou:.3f} | conf={conf:.3f} | c_err={c_err:.1f}px | best_n={best_n} gi={gi} gj={gj}"

        meta = {
            "q_path": q_path,
            "s_path": s_path,
            "qname": os.path.basename(qname),
            "sname": os.path.basename(sname),
            "title": title,
            "click_img_xy": click_point,
            "pred_xyxy": pred_xyxy,
            "gt_xyxy": gt_xyxy,
            "iou": iou, "conf": conf, "c_err": c_err, "area": area
        }

        for b in buckets:
            sc = topk.score(b, iou, conf, c_err, area, args.img_size)
            topk.push(b, sc, meta)

    # save top2 per bucket (query & satellite separately, no text)
    for bucket, items in topk.store.items():
        out_dir = os.path.join(args.save_dir, bucket)
        os.makedirs(out_dir, exist_ok=True)

        if len(items) == 0:
            print(f"[WARN] {bucket}: 0")
            continue

        for rank, (sc, m) in enumerate(items, start=1):
            q_bgr = cv2.imread(m["q_path"])
            s_bgr = cv2.imread(m["s_path"])
            q_rgb = cv2.cvtColor(q_bgr, cv2.COLOR_BGR2RGB)
            s_rgb = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)

            stem = f"{rank:02d}_{m['qname']}__{m['sname']}"
            out_q = os.path.join(out_dir, f"{stem}__query.png")
            out_s = os.path.join(out_dir, f"{stem}__satellite.png")

            save_query_only(out_q, q_rgb, m["click_img_xy"], radius=8)
            save_satellite_only(out_s, s_rgb, m["pred_xyxy"], m["gt_xyxy"])

        print(bucket, "saved:", len(items), "pairs (query+satellite)")


if __name__ == "__main__":
    main()
