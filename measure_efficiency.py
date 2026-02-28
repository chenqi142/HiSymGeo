import logging
import torch
import torch.nn as nn
from thop import profile  # pip install thop
import time
import argparse

# 请替换为您的模型定义路径
from model.HisymGeo import HisymGeo
from utils.checkpoint import load_pretrain  # 例如 from model import HiSymGeo

# logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
#                         format="%(asctime)-15s %(levelname)-8s %(message)s")
# logging.info(str(sys.argv))

def measure_efficiency(model, input_size_query, input_size_reference, device='cuda', repetitions=100):
    model.eval()
    model.to(device)

    # 构造示例输入（Batch size = 1）
    q = torch.randn(1, *input_size_query).to(device)  # [B, C, H, W]
    r = torch.randn(1, *input_size_reference).to(device)
    click_point = torch.tensor([[128, 256]]).to(device)  # 示例点击点，形状 [B, 2]

    # 1. 测量 #Params 和 FLOPs
    macs, params = profile(model, inputs=(q, r, click_point), verbose=False)
    flops = macs * 2  # MACs ≈ 0.5 FLOPs，但常规报告中常将 MACs ×2 视为 FLOPs
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs: {flops / 1e9:.2f} G")

    # 2. 预热 GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model(q, r, click_point)

    # 3. 测量推理时间（FPS）
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(q, r, click_point)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_sample = total_time / repetitions
    fps = 1.0 / avg_time_per_sample

    print(f"Average inference time: {avg_time_per_sample * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")

    return params / 1e6, flops / 1e9, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='saved_models_ResConvFus_LeakyReLU/model_droneaerial_checkpoint.pth.tar', help='Path to model checkpoint')
    args = parser.parse_args()

    # 初始化模型（请根据您的构造函数调整）
    model = HisymGeo()  # 若需传参，请按实际修改

    # 加载权重（可选，若未训练可跳过，仅测结构）
    if args.pretrain:
        model = load_pretrain(model, args, logging)

    # 输入尺寸（根据论文 Section IV-A）：
    # Ground: 256x512 → 经 ResNet-18 后特征图通道为 512，但输入为 RGB
    # Satellite: 1024x1024 → 输入为 RGB
    input_size_query = (3, 256, 512)      # Ground or Drone (统一用最大尺寸)
    input_size_reference = (3, 1024, 1024)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device.upper()}")

    params, flops, fps = measure_efficiency(
        model,
        input_size_query,
        input_size_reference,
        device=device,
        repetitions=100
    )

    # 可选：保存结果
    with open('efficiency_results.txt', 'w') as f:
        f.write(f"Params (M): {params:.2f}\n")
        f.write(f"FLOPs (G): {flops:.2f}\n")
        f.write(f"FPS: {fps:.2f}\n")