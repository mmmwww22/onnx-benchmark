"""PyTorch -> ONNX -> ONNX Runtime 的可配置端到端推理脚本。"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision


PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
DEFAULT_BATCH_SIZES = [1, 8, 32]
DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100

# 这是当前作业的固定对比模型组；--models 只允许从此列表中选择。
MODELS = {
    "resnet18": (
        torchvision.models.resnet18,
        torchvision.models.ResNet18_Weights.DEFAULT,
    ),
    "mobilenet_v2": (
        torchvision.models.mobilenet_v2,
        torchvision.models.MobileNet_V2_Weights.DEFAULT,
    ),
    "shufflenet_v2_x1_0": (
        torchvision.models.shufflenet_v2_x1_0,
        torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
    ),
}


def positive_int(value: str) -> int:
    """供 argparse 使用：只接受大于 0 的整数。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def non_negative_int(value: str) -> int:
    """供 argparse 使用：只接受大于等于 0 的整数。"""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出指定 CNN 模型为 ONNX，并使用 ONNX Runtime 校验和测速。"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="要运行的模型，可选多个；默认运行全部模型。",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=positive_int,
        default=DEFAULT_BATCH_SIZES,
        help="要测试的 batch size；默认：1 8 32。",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=DEFAULT_WARMUP,
        help="每种 batch 的预热次数；默认：10。",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=DEFAULT_REPEATS,
        help="每种 batch 的正式测速次数；默认：100。",
    )
    return parser.parse_args()


def load_model(model_fn, weights):
    model = model_fn(weights=weights)
    model.eval()
    return model.to(DEVICE)


def export_onnx(model: torch.nn.Module, name: str) -> Path:
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    onnx_path = PROJECT_DIR / f"{name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"{name} ONNX 导出并检查成功")
    return onnx_path


def create_session(onnx_path: Path):
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print("ONNX 输入名称：", input_info.name)
    print("ONNX 输入形状：", input_info.shape)
    print("ONNX 输出名称：", output_info.name)
    return session, input_info.name, output_info.name


def compare_outputs(
    model: torch.nn.Module,
    input_name: str,
    output_name: str,
    session: ort.InferenceSession,
    model_name: str,
) -> None:
    test_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.inference_mode():
        pytorch_output = model(test_input).cpu().numpy()

    onnx_output = session.run(
        [output_name], {input_name: test_input.cpu().numpy()}
    )[0]
    max_error = np.max(np.abs(onnx_output - pytorch_output))
    is_close = np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-3)

    print(f"{model_name} PyTorch 输出形状：{pytorch_output.shape}")
    print(f"{model_name} ORT 输出形状：{onnx_output.shape}")
    print(f"最大绝对误差：{max_error:.8f}")
    print(f"输出是否基本一致：{is_close}")
    if not is_close:
        raise ValueError(f"{model_name} 的 PyTorch 与 ORT 输出不一致")
    print("输出验证通过\n")


def benchmark_ort(
    batch: int,
    session: ort.InferenceSession,
    input_name: str,
    model_name: str,
    warmup: int,
    repeats: int,
) -> dict:
    dummy_input = torch.randn(batch, 3, 224, 224).numpy()
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - start_time)

    average_time = statistics.mean(times)
    latency = average_time * 1000
    throughput = batch / average_time
    print(
        f"ONNX Runtime | Batch={batch:2d} | Latency={latency:.3f} ms | "
        f"Throughput={throughput:.2f} img/s"
    )
    return {
        "Model": model_name,
        "Framework": "ONNX Runtime",
        "Batch Size": batch,
        "Latency (ms)": round(latency, 3),
        "Throughput (img/s)": round(throughput, 2),
    }


def benchmark_pytorch(
    batch: int,
    model: torch.nn.Module,
    model_name: str,
    warmup: int,
    repeats: int,
) -> dict:
    """在 CPU 上测量 PyTorch 模型的纯前向推理时间。"""
    dummy_input = torch.randn(batch, 3, 224, 224, device=DEVICE)
    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy_input)

        times = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            model(dummy_input)
            times.append(time.perf_counter() - start_time)

    average_time = statistics.mean(times)
    latency = average_time * 1000
    throughput = batch / average_time
    print(
        f"PyTorch | Batch={batch:2d} | Latency={latency:.3f} ms | "
        f"Throughput={throughput:.2f} img/s"
    )
    return {
        "Model": model_name,
        "Framework": "PyTorch",
        "Batch Size": batch,
        "Latency (ms)": round(latency, 3),
        "Throughput (img/s)": round(throughput, 2),
    }


def save_results(results: list[dict]) -> Path:
    csv_path = PROJECT_DIR / "inference_benchmark_results.csv"
    fieldnames = [
        "Model",
        "Framework",
        "Batch Size",
        "Latency (ms)",
        "Throughput (img/s)",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n测速结果保存至：{csv_path}")
    return csv_path


def main() -> int:
    args = parse_args()
    print("运行配置：")
    print(f"  模型：{', '.join(args.models)}")
    print(f"  Batch Size：{args.batches}")
    print(f"  预热次数：{args.warmup}")
    print(f"  正式测速次数：{args.repeats}\n")

    results = []
    failed_models = []
    for model_name in args.models:
        model_fn, weights = MODELS[model_name]
        try:
            model = load_model(model_fn, weights)
            onnx_path = export_onnx(model, model_name)
            session, input_name, output_name = create_session(onnx_path)
            compare_outputs(model, input_name, output_name, session, model_name)
            for batch in args.batches:
                results.append(
                    benchmark_pytorch(
                        batch,
                        model,
                        model_name,
                        args.warmup,
                        args.repeats,
                    )
                )
                results.append(
                    benchmark_ort(
                        batch,
                        session,
                        input_name,
                        model_name,
                        args.warmup,
                        args.repeats,
                    )
                )
        except Exception as error:
            failed_models.append(model_name)
            print(f"{model_name} 处理失败：{error}")

    save_results(results)
    print(f"成功记录的测速条数：{len(results)}")
    if failed_models:
        print(f"失败模型：{', '.join(failed_models)}")
        return 1

    print("全部模型处理流程结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
