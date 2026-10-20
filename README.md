# ONNX CNN CPU Benchmark

比较 ResNet18、MobileNetV2、ShuffleNetV2 x1.0 在 PyTorch CPU 与 ONNX Runtime CPU 上的推理性能。一次命令完成预训练模型加载、ONNX 导出与结构检查、各 batch 输出校验、双框架测速和结果保存。

## 安装

当前验证环境为 Windows、Python 3.14；依赖版本见 `requirements.txt`。首次使用需要下载 TorchVision 预训练权重，后续可复用本机缓存。

```cmd
git clone https://github.com/mmmwww22/onnx-benchmark.git
cd onnx-benchmark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python onnx_benchmark.py --help
```

## 运行

默认运行三个模型，batch size 为 1、8、32；每组预热 10 次、计时 100 次，测试 1 轮：

```cmd
python onnx_benchmark.py
```

完整成功时生成 18 条测速记录（3 个模型 × 3 个 batch × 2 个框架），另有 9 条输出校验记录和 9 条双框架对比记录。

快速确认流程是否能运行：

```cmd
python onnx_benchmark.py --models mobilenet_v2 --batches 1 --warmup 1 --repeats 3
```

正式记录实验时，可以固定线程数并运行多轮：

```cmd
python onnx_benchmark.py --models resnet18 mobilenet_v2 shufflenet_v2_x1_0 --batches 1 8 32 --threads 4 --seed 42 --warmup 20 --repeats 100 --rounds 3
```

该配置完整成功时有 54 条测速记录。每轮单独保留；不同 batch 和相邻轮次交替测试 PyTorch/ORT 的顺序。增加轮数会增加总运行时间。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--models` | 三个模型 | 可指定一个或多个固定模型组中的名称 |
| `--batches` | `1 8 32` | 一个或多个正整数 batch size |
| `--warmup` | `10` | 每框架、每 batch、每轮的预热次数，允许 0 |
| `--repeats` | `100` | 每组正式计时次数，必须大于 0 |
| `--threads` | `min(4, 逻辑 CPU 数)` | 两框架均显式设置的算子内线程数 |
| `--rounds` | `1` | 每种模型/batch 的测试轮数 |
| `--seed` | `42` | 随机种子，固定输入以便比较 |
| `--output-dir` | 脚本所在目录的 `results/` | 输出根目录。指定相对路径时，相对于运行命令的当前目录 |

模型名称为 `resnet18`、`mobilenet_v2`、`shufflenet_v2_x1_0`。重复指定的模型或 batch 会去重。原有模型、batch、预热和重复次数参数仍然可用。

在原学习目录运行时，主脚本名称为 `onnx_pipline_update.py`，将以上命令的脚本名替换即可。

## 输出文件

每次运行创建一个包含时间和随机后缀的独立子目录，保留旧实验结果：

```text
results/
└── 20261020_160000_123456_ab12cd34/
    ├── experiment.json
    ├── inference_benchmark_results.csv
    ├── comparisons.csv
    ├── validation_results.csv
    ├── raw_timings.csv
    ├── run.log
    └── models/
        ├── resnet18.onnx
        ├── mobilenet_v2.onnx
        └── shufflenet_v2_x1_0.onnx
```

- `experiment.json`：实验参数、CPU/系统/Python/依赖版本、线程设置、实际测试顺序、权重名称、脚本和 ONNX 文件哈希、运行状态及预期/实际记录数。CPU 信息取自系统接口，可能只显示架构，报告时可手动补充具体型号。
- `inference_benchmark_results.csv`：每框架、每 batch、每轮的均值、中位数、P95、标准差、吞吐量及测试配置。前五列保留原来的列名。
- `comparisons.csv`：同模型、同 batch、同轮的完整两框架对比。加速比 = PyTorch 延迟 ÷ ORT 延迟，大于 1 表示 ORT 在这一组更快；均值与中位数加速比分别计算。缺少任一框架结果时不生成该组加速比。
- `validation_results.csv`：各 batch 的输出形状、最大/平均绝对误差、数值接近情况和预测类别是否相同。
- `raw_timings.csv`：每次正式推理的原始计时（毫秒），不含预热。
- `run.log`：运行进度、警告和失败的异常堆栈。

每次完成校验或一组框架测速后保存进度。CSV 使用 UTF-8 BOM 编码，方便 Excel 打开；文件先写入临时文件再替换，减少写入途中留下半个文件的风险。多个文件并非整体事务，应结合 JSON 状态、记录数和各文件检查完整性。

原项目根目录下已有的 CSV/ONNX 文件不再作为新结果的写入目标。每轮导出到独立目录会占用磁盘空间，可按需归档已完成的实验。

## 结果如何理解

| 字段 | 含义 |
| --- | --- |
| `Latency (ms)` | 单批前向推理的平均耗时，不是整份脚本的运行时间 |
| `Median (ms)` | 本组单次耗时的中位数，便于观察典型耗时 |
| `P95 (ms)` | 本组耗时的第 95 百分位，使用线性插值计算 |
| `Std (ms)` | 单次耗时的总体标准差，用于查看组内波动 |
| `Throughput (img/s)` | 总处理图片数 ÷ 正式计时总秒数 |
| `Round` | 独立记录的测试轮次 |

CPU 计时使用 `perf_counter_ns`；包含框架前向调用及返回输出的开销，不包含权重加载、导出、输入构造、校验、图片读取、预处理、后处理或写文件。PyTorch 和 ORT 在同一模型/batch 下使用同一输入数据和同一组权重，均为 CPU、FP32 输入。

两侧算子内线程数由 `--threads` 控制，算子间线程数为 1；ORT 使用顺序图执行、开启图优化，并关闭算子内线程池忙等，减轻空闲 ORT 会话对 PyTorch 测试的干扰。相同线程配置不代表两框架内部执行方式完全相同，也不能消除温度、功耗及后台任务造成的波动。旧版使用默认线程配置，历史数值应连同当时条件一并保留，避免直接归因比较。

相关配置依据：[PyTorch 线程配置](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)、[ORT 线程管理](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)。

## 校验与失败处理

每个指定 batch 均执行 PyTorch/ORT 输出比对：形状必须一致，数值必须有限，并满足 `rtol=1e-3, atol=1e-3`。预测类别是否一致额外记录；数值接近但类别不同可能出现在接近并列的输出，应结合误差检查。本脚本不测分类准确率。

校验失败后停止该模型后续测速，记录错误并继续处理其他模型。该模型之前已完成的记录会保留，因此看到 CSV 并不代表整次实验成功。

运行状态见 `experiment.json`：

- `completed`：全部预期测速记录完成且没有模型失败，退出码 0。
- `failed`：有模型失败或记录不完整，退出码 1。
- `interrupted`：通过 Ctrl+C 中断，退出码 130，保留已完成组；当前未完成组的样本不写入结果。
- `running`：尚未正常收尾；强制结束进程、断电等情况可能保留此状态。

## 实验建议与范围

正式汇报时记录硬件、电源模式和运行配置，尽量避免后台高负载，并观察多轮结果。固定随机种子固定的是输入构造条件，不能保证延迟数值相同。只运行 3 次适合验证功能，不足以稳定估计 P95 或加速幅度。

当前仅支持列出的三个预训练 CNN、CPU FP32 推理，以及模型部署流程验证。随机输入用于一致性校验和测速，不代表真实图片的分类效果。为保持当前依赖与导出行为，继续使用显式 `dynamo=False` 的导出方式；新版 PyTorch 可能提示弃用警告。

本项目目前未附带许可证。
