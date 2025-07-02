import copy
from pathlib import Path
from typing import Dict, Any, Union

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER

# 导入自己的推理类
from torch_model import YOLOv8Detector


class MyBackend(torch.nn.Module):
    """将自定义 YOLOv8Detector 封装成与 AutoBackend 行为一致的推理后端。"""

    def __init__(self, weights_path: str = "/root/work/runs/yolo11s.pt", device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        # 内部 detector 负责真正的模型加载与推理
        self.detector = YOLOv8Detector(weights_path=weights_path)
        # names/stride 等属性，DetectionValidator 会用到
        self.names = {i: n for i, n in enumerate(self.detector.classes)}
        # DetectionModel.stride 通常是 list/tuple/torch.Tensor，取 max 方便后续计算
        stride = self.detector.model.stride
        if isinstance(stride, torch.Tensor):
            self.stride = stride
        elif isinstance(stride, (list, tuple)):
            self.stride = torch.tensor(stride)
        else:
            self.stride = torch.tensor([int(stride)])

        # 下面这些布尔标记是 AutoBackend 的常用属性，在 validator 判断逻辑里会访问
        self.pt, self.jit, self.engine, self.dynamic = True, False, False, False
        self.fp16 = False
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.detector.model.to(self.device)

        # AutoBackend 依赖于 .yaml['channels'] 来确定输入通道数
        self.yaml = {"channels": 3}

    @torch.no_grad()
    def forward(
        self,
        imgs: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed=None,
        **kwargs,
    ):  # type: ignore[override]
        """DetectionValidator 会调用此函数进行前向推理。忽略 AutoBackend 可能传入的其他关键字。"""
        return self.detector.forward(imgs)

    # DetectionValidator 在推理模式下会先 warmup()；我们空实现即可
    def warmup(self, *args, **kwargs):
        pass

    # AutoBackend 在 nn_module 模式下会尝试调用 .fuse()，此处直接返回自身即可
    def fuse(self, verbose: bool = False):  # noqa: D401
        """兼容接口：保持与 Ultralytics DetectionModel 一致，直接返回 self"""
        return self


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
    weights_path: str = "/root/work/runs/yolo11s.pt",
    data_yaml: Union[str, Path] = "ultralytics/cfg/datasets/coco128.yaml",
    imgsz: int = 640,
    batch: int = 8,
    device: Union[str, int, torch.device] = "cpu",
):
    """比较官方 AutoBackend 与自定义 MyBackend 在相同数据集上的 mAP/FPS 差异"""
    base_args: Dict[str, Any] = dict(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.7,
        device=device,
        verbose=False,
        save_json=False,  # 关闭保存以加快速度
        plots=False,      # 关闭绘图
    )

    # 1. 官方 AutoBackend
    LOGGER.info("\n>>> Running official AutoBackend validation …")
    official_model = AutoBackend(weights_path, device=device, fp16=False)
    off_stats = run_validation(official_model, copy.deepcopy(base_args))

    # 2. 自定义 Backend
    LOGGER.info("\n>>> Running custom MyBackend validation …")
    my_model = MyBackend(weights_path, device=device)
    my_stats = run_validation(my_model, copy.deepcopy(base_args))

    # 3. 打印并比对
    LOGGER.info("\n========= Result Comparison =========")
    keys = sorted(off_stats.keys())
    for k in keys:
        off_v = off_stats[k]
        my_v = my_stats.get(k, None)
        diff = None if my_v is None else my_v - off_v
        LOGGER.info(f"{k:>12}: official = {off_v:.4f},  custom = {my_v:.4f}  (Δ={diff:+.4f})")

    return off_stats, my_stats


if __name__ == "__main__":
    # 直接运行当前脚本可进行一次对比评测
    compare_with_official() 