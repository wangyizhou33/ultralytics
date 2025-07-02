import copy
from pathlib import Path
from typing import Dict, Any, Union

import torch
from ultralytics.models.yolo.detect import DetectionValidator

# 在导入 AutoBackend 前先注册自定义后端
import benchmark.custom_backend

from ultralytics.nn.autobackend import AutoBackend  # noqa: E402
from ultralytics.utils import LOGGER


# ------------------- 评测函数 ------------------- #

def run_validation(model: torch.nn.Module, args: Dict[str, Any]) -> Dict[str, Any]:
    """使用 DetectionValidator 跑一次验证并返回统计结果"""
    validator = DetectionValidator(args=args)
    # 若传入的 model 已经是 AutoBackend；则取其内部真实模型以免在 DetectionValidator 内再次封装
    if isinstance(model, AutoBackend):
        model_to_use = model.model  # 底层 DetectionModel，具有 fuse() 等接口
    else:
        model_to_use = model

    stats = validator(model=model_to_use)
    return stats


def compare_with_official(
    official_weights: Union[str, Path] = "/root/work/runs/yolo11s.pt",
    custom_weights: Union[str, Path] = "/root/work/runs/yolo11s.pt",  # 与 official 相同文件，但用 custom: 前缀走自定义后端
    data_yaml: Union[str, Path] = "ultralytics/cfg/datasets/coco128.yaml",
    imgsz: int = 640,
    batch: int = 8,
    device: Union[str, int, torch.device] = "cpu",
):
    """比较官方 AutoBackend 与自定义后端在相同数据集上的验证速度/精度"""
    base_args: Dict[str, Any] = dict(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.7,
        device=device,
        verbose=False,
        save_json=False,
        plots=False,
    )

    # 1️⃣ 官方 AutoBackend
    LOGGER.info("\n>>> Running official AutoBackend validation …")
    # official_model = AutoBackend(str(official_weights), device=device, fp16=False)
    off_stats = run_validation(str(official_weights), copy.deepcopy(base_args))

    # 2️⃣ 自定义后端 (通过 custom: 前缀触发)
    LOGGER.info("\n>>> Running custom backend validation …")
    custom_path = f"custom:{custom_weights}" if not str(custom_weights).startswith("custom:") else str(custom_weights)
    # custom_model = AutoBackend(custom_path, device=device, fp16=False)
    my_stats = run_validation(custom_path, copy.deepcopy(base_args))

    # 3️⃣ 打印对比
    LOGGER.info("\n========= Result Comparison =========")
    for k in sorted(off_stats.keys()):
        off_v = off_stats[k]
        my_v = my_stats.get(k, None)
        diff = None if my_v is None else my_v - off_v
        LOGGER.info(f"{k:>12}: official = {off_v:.4f},  custom = {my_v:.4f}  (Δ={diff:+.4f})")

    # print(f'custom_path = {custom_path}')
    # from ultralytics import YOLO
    # model = YOLO(custom_path)
    # # Perform object detection on an image
    # results = model("/root/work/datasets/coco8/images/val/000000000036.jpg")  # Predict on an image
    # # results[0].show()  # Display results
    # results[0].save(filename="result.jpg")  # save to disk

    return off_stats, my_stats


if __name__ == "__main__":
    compare_with_official() 