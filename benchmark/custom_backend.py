# Ultralytics custom backend plugin
# 通过 monkey-patch 向 ultralytics.nn.autobackend.AutoBackend 注入"自定义加载逻辑"。
# 识别权重路径形如  'custom:/path/to/xxx.pt'  或  'custom:xxx.pt'。
# 其余流程保持官方 AutoBackend 不变，实现完全无侵入扩展。
# -----------------------------------------------------------------------------
from __future__ import annotations

import types
from pathlib import Path
from typing import List, Union

import torch
from ultralytics.utils import LOGGER

from torch_model import YOLOv8Detector  # 自定义 Detector

import ultralytics.nn.autobackend as ab  # 原始 AutoBackend

CUSTOM_PREFIX = "custom:"

# 保存官方实现，稍后回调 -----------------------------------------
_orig_init = ab.AutoBackend.__init__


def _custom_init(self: ab.AutoBackend, weights: Union[str, List[str], torch.nn.Module], *args, **kwargs):
    """扩展 AutoBackend.__init__：当权重字符串以 'custom:' 前缀开头时，
    使用 YOLOv8Detector 做加载；否则走原流程。
    """

    device = kwargs.get("device", torch.device("cpu"))
    if args:
        device = args[0]

    # 判定是否采用自定义后端
    if isinstance(weights, (str, Path, list)):
        # 把 list[str] 归一化成 str 以便判断
        w0 = weights[0] if isinstance(weights, list) else weights
        w0 = str(w0)
    else:
        w0 = ""

    if isinstance(w0, str) and w0.startswith(CUSTOM_PREFIX):
        real_path = w0[len(CUSTOM_PREFIX) :]
        LOGGER.info(f"AutoBackend: using custom backend for PT weights → {real_path}")

        # -------------------------------- 自定义加载 ------------------------------
        torch.nn.Module.__init__(self)  # 初始化 nn.Module

        detector = YOLOv8Detector(weights_path=real_path)
        detector.model.to(device)

        # 对齐 AutoBackend 预期属性
        self.model = detector.model
        self.detector = detector
        # Ultralytics 期望 names 映射的 value 为 **str**。
        if isinstance(detector.classes, dict):
            # 已是 {id: name}
            self.names = {int(k): str(v) for k, v in detector.classes.items()}
        else:  # list 或 tuple
            self.names = {i: str(n) for i, n in enumerate(detector.classes)}

        # stride 应当是一个 **标量整数**，否则后续 build_yolo_dataset(int(stride)) 会报
        # "only one element tensors can be converted to Python scalars"。
        _s = detector.model.stride  # 可能是 list、tuple 或 Tensor
        if isinstance(_s, torch.Tensor):
            _s = int(_s.max())
        elif isinstance(_s, (list, tuple)):
            _s = int(max(_s))
        else:
            _s = int(_s)

        self.stride = _s  # 保证为 python int

        # 标记位（仅保留 pt=True 方便后续逻辑）
        self.pt = True
        for flag in [
            "jit",
            "onnx",
            "xml",
            "engine",
            "coreml",
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
            "paddle",
            "mnn",
            "ncnn",
            "imx",
            "rknn",
            "triton",
        ]:
            setattr(self, flag, False)

        self.dynamic = False
        self.fp16 = False
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        # channels
        ch = detector.model.yaml.get("channels", 3) if hasattr(detector.model, "yaml") else 3
        self.yaml = {"channels": ch}
        self.ch = ch  # Predictor.warmup 需要

        # 前向 & 兼容函数 --------------------------------------------------------
        @torch.no_grad()
        def _forward(self, imgs: torch.Tensor, *f_args, **f_kwargs):  # type: ignore[override]
            """AutoBackend.forward：调用内部 self.detector 进行推理"""
            imgs = imgs.to(self.device, dtype=torch.float32)
            print(
                "Hahahahahahahahahahahahahahahahaha =====================123 (custom forward called)"
            )
            return self.detector.forward(imgs)

        # 绑定为实例方法
        self.forward = types.MethodType(_forward, self)
        self.warmup = lambda *a, **k: None  # noqa: E731
        self.fuse = lambda verbose=False: self  # noqa: E731
        return  # 结束，自定义流程已完成

    # ------------------------ 非 custom: 前缀，回到官方实现 ----------------------
    _orig_init(self, weights, *args, **kwargs)


ab.AutoBackend.__init__ = _custom_init  # 覆写回类

# 可选：修补 _model_type 让日志更友好 ------------------------------
_orig_model_type = ab.AutoBackend._model_type


@staticmethod
def _model_type_patched(p: str = "path/to/model.pt") -> List[bool]:  # type: ignore[override]
    if isinstance(p, str) and p.startswith(CUSTOM_PREFIX):
        p = p[len(CUSTOM_PREFIX) :]
    return _orig_model_type(p)


ab.AutoBackend._model_type = _model_type_patched

LOGGER.info("✅  Custom PT backend registered (prefix 'custom:').")

# ------------------- 防止自动下载 -------------------------------------------
# Ultralytics 内部在 AutoBackend.__init__ 前会调用 ultralytics.utils.downloads
# 的 attempt_download_asset() 判断/下载权重。若给出 'custom:' 前缀，它会被
# 误判为远程 URL。这里通过 monkey-patch 让该函数直接返回本地真实路径。

import ultralytics.utils.downloads as _dl  # 延迟导入以保证模块已就绪

_orig_attempt_download = _dl.attempt_download_asset  # 备份原函数


def _attempt_download_hook(path: Union[str, Path], *args, **kwargs):  # noqa: D401
    """Hook for ultralytics.utils.downloads.attempt_download_asset.

    若 path 以 'custom:' 开头，则视为本地文件，直接返回去掉前缀后的路径；
    否则调用原始实现。
    """

    if isinstance(path, (str, Path)) and str(path).startswith(CUSTOM_PREFIX):
        return str(path)[len(CUSTOM_PREFIX) :]
    return _orig_attempt_download(path, *args, **kwargs)


# 覆写 utils.downloads
_dl.attempt_download_asset = _attempt_download_hook  # type: ignore[attr-defined]

# 同时替换 AutoBackend 模块作用域内先前导入的引用
setattr(ab, "attempt_download_asset", _attempt_download_hook)

# ------------------- 让 YOLO 保留 custom: 字符串 -------------------------------
import ultralytics.engine.model as _mdl

_orig_load_model = _mdl.Model._load  # 备份

def _load_patched(self: _mdl.Model, weights, task=None):  # type: ignore[override]
    """替换 Model._load：遇到 custom: 前缀时不解析而直接保留字符串"""
    if isinstance(weights, (str, Path)) and str(weights).startswith(CUSTOM_PREFIX):
        # 仅填充最基本的属性，等待后续 AutoBackend 处理
        self.model = weights          # 关键：保持字符串形态
        self.ckpt = None
        self.task = task or "detect"  # 默认任务，可按需改变
        self.overrides = getattr(self, "overrides", {})
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights
        return                        # 提前返回 → 不走官方解析
    # 非 custom: 仍走官方逻辑
    return _orig_load_model(self, weights, task)

_mdl.Model._load = _load_patched      # 完成覆写 