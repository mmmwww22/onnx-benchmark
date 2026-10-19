# ONNX CNN CPU Benchmark

一个用于比较 PyTorch CPU 与 ONNX Runtime CPU 推理性能的小型实验项目。脚本会自动完成模型加载、ONNX 导出、模型校验、数值一致性验证和推理测速，并将结果保存为 CSV。

## 功能

- 导出 PyTorch 预训练模型为 ONNX；
- 使用 `onnx.checker` 验证导出的模型；
- 比较 PyTorch 与 ONNX Runtime 输出是否一致；
- 在 CPU 上分别测量 PyTorch 和 ONNX Runtime 的延迟与吞吐量；
- 支持设置模型、batch size、预热次数和正式测速次数；
- 将实验结果自动保存为 `inference_benchmark_results.csv`。

默认实验组：

| 模型 | 角色 |
| --- | --- |
| ResNet18 | 常规 CNN 基线 |
| MobileNetV2 | 轻量化 CNN |
| ShuffleNetV2 x1.0 | 轻量化 CNN |

## 工作流程

```text
加载 PyTorch 预训练模型
        ↓
导出 ONNX 并检查模型合法性
        ↓
分别运行 PyTorch 和 ONNX Runtime
        ↓
比较两者输出是否一致
        ↓
按不同 batch size 测量延迟和吞吐量
        ↓
保存 CSV 结果
```

## 环境要求

- Windows 10/11（脚本在 Windows 上开发与验证）
- Python 3.14
- CPU 版 PyTorch、TorchVision、ONNX、ONNX Runtime
- 首次运行时可访问网络，用于下载 TorchVision 的预训练权重

已验证的主要依赖版本写在 [`requirements.txt`](requirements.txt) 中。

## 安装

克隆仓库：

```bash
git clone https://github.com/mmmwww22/onnx-benchmark.git
cd onnx-benchmark
```

创建并启用虚拟环境：

```cmd
python -m venv .venv
.venv\Scripts\activate
```

安装依赖：

```cmd
pip install -r requirements.txt
```

确认脚本参数：

```cmd
python onnx_benchmark.py --help
```

## 使用方法

### 完整实验

默认运行 3 个模型，在 batch size 为 1、8、32 时分别测试 PyTorch 与 ONNX Runtime。每个配置预热 10 次、正式测速 100 次：

```cmd
python onnx_benchmark.py
```

这会生成 18 条结果：`3 个模型 × 3 个 batch size × 2 个框架`。

### 快速验证

首次确认环境是否可用时，建议先运行一个很小的测试：

```cmd
python onnx_benchmark.py --models mobilenet_v2 --batches 1 --warmup 1 --repeats 3
```

若控制台出现以下内容，表示 ONNX 导出和数值校验成功：

```text
输出是否基本一致：True
PyTorch | Batch= 1 | Latency=... | Throughput=...
ONNX Runtime | Batch= 1 | Latency=... | Throughput=...
```

### 自定义实验配置

只测试一个模型：

```cmd
python onnx_benchmark.py --models shufflenet_v2_x1_0
```

选择多个模型、batch size 和测速次数：

```cmd
python onnx_benchmark.py --models resnet18 mobilenet_v2 --batches 1 8 --warmup 20 --repeats 200
```

可选模型名称：

```text
resnet18
mobilenet_v2
shufflenet_v2_x1_0
```

## 结果说明

每次运行会在项目根目录生成或覆盖：

- `*.onnx`：本次导出的 ONNX 模型；
- `inference_benchmark_results.csv`：本次实验的测速结果。

CSV 示例：

```csv
Model,Framework,Batch Size,Latency (ms),Throughput (img/s)
mobilenet_v2,PyTorch,1,4.250,235.29
mobilenet_v2,ONNX Runtime,1,2.800,357.14
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `Model` | 模型名称 |
| `Framework` | 推理框架：`PyTorch` 或 `ONNX Runtime` |
| `Batch Size` | 每次同时推理的图像数量 |
| `Latency (ms)` | 一次前向推理的平均延迟，越小越好 |
| `Throughput (img/s)` | 每秒可处理的图像数，越大越好 |

请在模型、batch size、预热次数和测速次数相同的条件下比较两种框架。

## 测速建议

CPU 性能会受电源模式、后台程序、温度和线程调度影响。为获得更有参考价值的结果：

1. 插电并使用高性能电源模式；
2. 关闭浏览器、下载器、游戏等占用 CPU 的程序；
3. 对同一配置至少运行 3 次；
4. 使用多次结果的中位数或平均值判断趋势；
5. 将结果视为本机环境下的对比，而非通用硬件性能结论。

## 项目结构

```text
onnx-benchmark/
├── onnx_benchmark.py            # 主脚本
├── requirements.txt             # Python 依赖
├── inference_benchmark_results.csv  # 运行后生成
└── README.md                    # 项目说明
```

## 限制

- 当前仅支持 README 列出的三个模型，用于保持实验对比的可解释性；
- 首次运行会下载预训练权重，离线使用前需要预先准备权重缓存；
- 脚本会重新生成 ONNX 文件并覆盖同名结果 CSV；
- 当前仅测量模型前向推理，不包含图片读取、预处理和后处理时间。

## License

本项目目前未附带许可证。若准备公开发布或允许他人复用，建议补充一个许可证文件，例如 MIT License。
