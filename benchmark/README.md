# Yolo 的基本封装模式 ———— 通过backend兼容不同格式的权重和推理实现

在 Ultralytics 中，YOLO 类被设计成“一层统一封装 + 多个后端实现”的架构。
不同格式的模型（PyTorch、TorchScript、ONNX、TensorRT …）在调用 YOLO(path) 时，核心差异体现在 “加载与推理后端” 两点，其他诸如 predict()、val()、export() 等高层接口保持一致。
以predict()为例，YOLO类定义了yolo.detect.DetectionPredictor，通过调用model = YOLO("xxx.onnx")： 
    当传入字符串路径时，Model._load() 先把 self.model 设成 纯路径字符串（非 .pt 情况）。
    直到第一次调用 predict()/val()，BasePredictor.setup_model() 才真正把这个字符串交给ultralytics.nn.autobackend.AutoBackend
    也就是：
        BasePredictor.__call__ ---> stream_inference() ---> setup_model() ---> self.model = AutoBackend

## setup_model() 
```python
    def setup_model(self, model, verbose: bool = True):
        """
        Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module, optional): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        self.model.eval()
```

```python
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        self.setup_model(model)

        # Preprocess
        with profilers[0]:
            im = self.preprocess(im0s)

        # Inference
        with profilers[1]:
            preds = self.inference(im, *args, **kwargs)

        # Postprocess
        with profilers[2]:
            self.results = self.postprocess(preds, im, im0s)
```

# 方式一：自定义 backend 并调用 ———— 自定义部分简洁，但需要引入DetectionValidator、ClassificationPredictor等
    - 实现新的后端类（假设叫 MyBackend）
    - 文件位置：ultralytics/engine/exports/backends/my_backend.py
    - 继承 BaseBackend，实现以下最少方法：
        ** __init__(self, weights, device, **kwargs) 负责把权重解析成可调用对象
        ** _forward(self, im, *args, **kwargs)   真正的前向推理
        ** warmup()（可选）           做线程 / Tensor 预热
        ** from_export()（可选）        若支持 export() 反向构建

## 使用方式：
    ```python
        python -m benchmark.benchmark_ed1
    ```

# 方式二：采用monkey-patch方法复写yolo相关部分 ———— 可直接使用yolo的推理和评测，但需要patch很多处
## 使用方式：
    ```python
        python -m benchmark.benchmark_ed2
    ```

