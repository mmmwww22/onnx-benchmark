"""可复现配置的 PyTorch -> ONNX -> ONNX Runtime CPU 推理实验。"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import uuid

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision


PROJECT_DIR = Path(__file__).resolve().parent
MODELS = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT),
    "mobilenet_v2": (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.DEFAULT),
    "shufflenet_v2_x1_0": (
        torchvision.models.shufflenet_v2_x1_0,
        torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
    ),
}
RESULT_FIELDS = [
    "Model", "Framework", "Batch Size", "Latency (ms)", "Throughput (img/s)",
    "Round", "Median (ms)", "P95 (ms)", "Std (ms)", "Threads", "Warmup", "Repeats",
]
RAW_FIELDS = ["Model", "Framework", "Batch Size", "Round", "Iteration", "Latency (ms)"]
VALIDATION_FIELDS = [
    "Model", "Batch Size", "PyTorch Shape", "ORT Shape", "Max Abs Error",
    "Mean Abs Error", "All Close", "Predictions Equal", "Finite", "Passed",
]
COMPARISON_FIELDS = [
    "Model", "Batch Size", "Round", "PyTorch Mean (ms)", "ORT Mean (ms)",
    "ORT Speedup (mean)", "ORT Speedup (median)",
]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def non_negative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--batches", nargs="+", type=positive_int, default=[1, 8, 32])
    parser.add_argument("--warmup", type=non_negative_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=100)
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1),
                        help="两框架的算子内线程数；默认最多 4，算子间线程数固定为 1")
    parser.add_argument("--seed", type=non_negative_int, default=42)
    parser.add_argument("--rounds", type=positive_int, default=1,
                        help="每种模型/batch 的测试轮数，每轮交替测试框架顺序")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results",
                        help="结果根目录，每次运行自动创建独立子目录")
    args = parser.parse_args(argv)
    args.models = list(dict.fromkeys(args.models))
    args.batches = list(dict.fromkeys(args.batches))
    return args


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 63))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_comparisons(results):
    grouped = {}
    for row in results:
        key = (row["Model"], row["Batch Size"], row["Round"])
        grouped.setdefault(key, {})[row["Framework"]] = row
    comparisons = []
    for (name, batch, round_number), pair in grouped.items():
        if "PyTorch" not in pair or "ONNX Runtime" not in pair:
            continue
        pytorch, ort_row = pair["PyTorch"], pair["ONNX Runtime"]
        comparisons.append({
            "Model": name, "Batch Size": batch, "Round": round_number,
            "PyTorch Mean (ms)": pytorch["Latency (ms)"],
            "ORT Mean (ms)": ort_row["Latency (ms)"],
            "ORT Speedup (mean)": pytorch["Latency (ms)"] / ort_row["Latency (ms)"],
            "ORT Speedup (median)": pytorch["Median (ms)"] / ort_row["Median (ms)"],
        })
    return comparisons


def save_progress(run_dir, metadata, results, validations, raw_times):
    atomic_csv(run_dir / "inference_benchmark_results.csv", RESULT_FIELDS, results)
    atomic_csv(run_dir / "validation_results.csv", VALIDATION_FIELDS, validations)
    atomic_csv(run_dir / "raw_timings.csv", RAW_FIELDS, raw_times)
    atomic_csv(run_dir / "comparisons.csv", COMPARISON_FIELDS, make_comparisons(results))
    metadata["recorded_results"] = len(results)
    metadata["updated_at"] = now()
    atomic_json(run_dir / "experiment.json", metadata)


def export_onnx(model, name, model_dir):
    path = model_dir / f"{name}.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), str(path),
        input_names=["input"], output_names=["output"], opset_version=18,
        dynamo=False,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    onnx.checker.check_model(onnx.load(str(path)))
    logging.info("%s ONNX 导出并检查成功", name)
    return path


def create_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT 空闲时不忙等，减少其线程池对随后 PyTorch 测速的干扰。
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("未使用预期的 CPUExecutionProvider")
    return session


def compare_outputs(model, session, x, name, batch):
    with torch.inference_mode():
        pytorch_output = model(x).numpy()
    ort_output = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    shape_matches = pytorch_output.shape == ort_output.shape
    finite = bool(np.isfinite(pytorch_output).all() and np.isfinite(ort_output).all())
    max_error = mean_error = None
    is_close = predictions_equal = False
    if shape_matches and finite:
        diff = np.abs(pytorch_output - ort_output)
        max_error, mean_error = float(diff.max()), float(diff.mean())
        is_close = bool(np.allclose(pytorch_output, ort_output, rtol=1e-3, atol=1e-3))
        predictions_equal = bool((pytorch_output.argmax(1) == ort_output.argmax(1)).all())
    passed = shape_matches and finite and is_close
    logging.info("%s batch=%d 输出验证：%s | max_abs_error=%s | 预测一致=%s",
                 name, batch, passed, max_error, predictions_equal)
    return {
        "Model": name, "Batch Size": batch,
        "PyTorch Shape": str(list(pytorch_output.shape)), "ORT Shape": str(list(ort_output.shape)),
        "Max Abs Error": max_error, "Mean Abs Error": mean_error,
        "All Close": is_close, "Predictions Equal": predictions_equal,
        "Finite": finite, "Passed": passed,
    }


def benchmark(run, warmup, repeats):
    # 输入生成、格式转换、校验和写文件均在计时区间之外。
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        times_ms = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            run()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            if elapsed <= 0 or not np.isfinite(elapsed):
                raise RuntimeError("无效计时样本")
            times_ms.append(elapsed)
    return times_ms


def summarize(times_ms, name, framework, batch, round_number, args):
    mean_ms = statistics.mean(times_ms)
    return {
        "Model": name, "Framework": framework, "Batch Size": batch,
        "Latency (ms)": mean_ms,
        "Throughput (img/s)": batch * len(times_ms) * 1000 / sum(times_ms),
        "Round": round_number, "Median (ms)": statistics.median(times_ms),
        "P95 (ms)": float(np.percentile(times_ms, 95, method="linear")),
        "Std (ms)": statistics.pstdev(times_ms),
        "Threads": args.threads, "Warmup": args.warmup, "Repeats": args.repeats,
    }


def main(argv=None):
    args = parse_args(argv)
    # 必须在模型执行前设置。每次 CLI 运行使用独立进程。
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    args.output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8])
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
    )
    logging.captureWarnings(True)
    metadata = {
        "schema_version": 1, "run_id": run_dir.name, "status": "running",
        "started_at": now(), "finished_at": None,
        "configuration": {**vars(args), "output_dir": str(args.output_dir)},
        "environment": {
            "python": sys.version, "python_executable": sys.executable,
            "os": platform.platform(), "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "pytorch_intra_op_threads": torch.get_num_threads(),
            "pytorch_inter_op_threads": torch.get_num_interop_threads(),
        },
        "protocol": {
            "device": "cpu", "dtype": "float32", "input_shape": ["batch", 3, 224, 224],
            "timing": "CPU forward wall-clock time; excludes export, validation, input creation and file IO",
            "rtol": 1e-3, "atol": 1e-3, "std": "population", "p95": "linear interpolation",
            "ort_intra_op_threads": args.threads, "ort_inter_op_threads": 1,
            "ort_execution_mode": "ORT_SEQUENTIAL", "ort_graph_optimization": "ORT_ENABLE_ALL",
            "ort_intra_op_allow_spinning": False,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_results": len(args.models) * len(args.batches) * args.rounds * 2,
        "models": {}, "execution_order": [], "failures": [],
    }
    results, validations, raw_times = [], [], []
    save = lambda: save_progress(run_dir, metadata, results, validations, raw_times)
    save()
    logging.info("结果目录：%s", run_dir)
    logging.info("模型=%s | batch=%s | 线程=%d | 预热=%d | 测速=%d | 轮数=%d | seed=%d",
                 args.models, args.batches, args.threads, args.warmup, args.repeats, args.rounds, args.seed)
    exit_code = 0
    model = session = None
    try:
        for name in args.models:
            model_state = metadata["models"][name] = {"status": "running", "stage": "load"}
            try:
                # 模型单独选择或调整排列时，也保持该模型的输入随机序列一致。
                model_seed = args.seed + list(MODELS).index(name)
                set_seed(model_seed)
                model_fn, weights = MODELS[name]
                model_state.update({"seed": model_seed, "weights": str(weights)})
                model = model_fn(weights=weights).eval().cpu()
                model_state["stage"] = "export"
                path = export_onnx(model, name, model_dir)
                model_state["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                model_state["stage"] = "session"
                session = create_session(path, args.threads)
                model_state["providers"] = session.get_providers()
                for batch_index, batch in enumerate(args.batches):
                    generator = torch.Generator().manual_seed((model_seed + batch) % (2 ** 63))
                    x = torch.randn(batch, 3, 224, 224, generator=generator)
                    model_state.update({"stage": "validation", "batch": batch})
                    validation = compare_outputs(model, session, x, name, batch)
                    validations.append(validation)
                    save()
                    if not validation["Passed"]:
                        raise ValueError(f"{name} batch={batch} 输出校验失败，跳过该模型剩余测速")
                    feed = {session.get_inputs()[0].name: x.numpy()}
                    runners = {"PyTorch": lambda: model(x),
                               "ONNX Runtime": lambda: session.run(None, feed)}
                    for round_number in range(1, args.rounds + 1):
                        order = list(runners)
                        if (batch_index + round_number - 1) % 2:
                            order.reverse()
                        metadata["execution_order"].append({
                            "model": name, "batch": batch, "round": round_number, "frameworks": order,
                        })
                        for framework in order:
                            model_state.update({"stage": "benchmark", "round": round_number, "framework": framework})
                            save()
                            times_ms = benchmark(runners[framework], args.warmup, args.repeats)
                            row = summarize(times_ms, name, framework, batch, round_number, args)
                            results.append(row)
                            raw_times.extend({
                                "Model": name, "Framework": framework, "Batch Size": batch,
                                "Round": round_number, "Iteration": i, "Latency (ms)": elapsed,
                            } for i, elapsed in enumerate(times_ms, 1))
                            save()
                            logging.info("%s | %s | batch=%d round=%d | mean=%.3f ms median=%.3f ms P95=%.3f ms | %.2f img/s",
                                         name, framework, batch, round_number, row["Latency (ms)"],
                                         row["Median (ms)"], row["P95 (ms)"], row["Throughput (img/s)"])
                model_state["status"] = "completed"
            except Exception as error:
                model_state.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
                metadata["failures"].append({"model": name, "stage": model_state["stage"], "error": str(error)})
                logging.exception("%s 处理失败，保留已完成结果，继续其他模型", name)
            finally:
                model = session = None
                save()
        if metadata["failures"] or len(results) != metadata["expected_results"]:
            metadata["status"], exit_code = "failed", 1
        else:
            metadata["status"] = "completed"
    except KeyboardInterrupt:
        metadata["status"], exit_code = "interrupted", 130
        for state in metadata["models"].values():
            if state["status"] == "running":
                state["status"] = "interrupted"
        logging.warning("用户中断；已完成的配置已保存，当前未完成的计时组不计入结果")
    except Exception as error:
        metadata["status"], exit_code = "failed", 1
        metadata["failures"].append({"model": None, "stage": "run", "error": str(error)})
        logging.exception("实验失败")
    finally:
        metadata["finished_at"] = now()
        save()
        logging.info("实验状态=%s | 记录=%d/%d | 结果目录=%s",
                     metadata["status"], len(results), metadata["expected_results"], run_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
