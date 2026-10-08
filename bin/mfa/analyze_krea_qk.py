#!/usr/bin/env python3
"""Compare captured production Q/K quantizers against original-basis FP64 QK."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def hadamard(x):
    x = x.astype(np.float32, copy=True)
    block = min(x.shape[-1] & -x.shape[-1], 256)
    y = x.reshape(*x.shape[:-1], -1, block)
    width = 1
    while width < block:
        view = y.reshape(*y.shape[:-1], -1, 2 * width)
        a = view[..., :width].copy()
        b = view[..., width:].copy()
        view[..., :width] = a + b
        view[..., width:] = a - b
        width *= 2
    return x


def read_buffers(meta_path, meta):
    source = np.memmap(meta_path.with_suffix(".bin"), dtype=np.uint8, mode="r")
    tensors = {}
    for name, info in meta["buffers"].items():
        data = source[info["offset"]:info["offset"] + info["bytes"]]
        if name.endswith("scale"):
            heads = meta["Hq"] if "q_scale" in name else meta["Hk"]
            length = meta["R"] if "q_scale" in name else meta["C"]
            tile = meta["q_tile"] if "q_scale" in name else meta["k_tile"]
            value = data.view("<f4").reshape(meta["B"], heads, (length + tile - 1) // tile)
        else:
            is_q = name == "q" or name.endswith("_q")
            heads, length = (meta["Hq"], meta["R"]) if is_q else (meta["Hk"], meta["C"])
            if name in ("q", "k"):
                if meta["precision"] == "bf16":
                    value = (data.view("<u2").astype(np.uint32) << 16).view(np.float32)
                else:
                    value = data.view("<f2" if meta["precision"] == "fp16" else "<f4")
            else:
                value = data.view(np.int8)
            value = value.reshape(meta["B"], length, heads, meta["D"])
        tensors[name] = value
    return tensors


def check_quantizer(x, quant, scales, tile, rotated):
    y = hadamard(x) if rotated else x.astype(np.float32)
    normalization = np.float32(1 / np.sqrt(min(x.shape[-1] & -x.shape[-1], 256))) if rotated else np.float32(1)
    max_quant_error = 0
    max_scale_error = 0.0
    for index, start in enumerate(range(0, len(y), tile)):
        block = y[start:start + tile]
        maximum = np.max(np.abs(block))
        inv_scale = np.float32(127) / maximum if maximum else np.float32(127)
        expected_scale = maximum * (normalization / np.float32(127)) if maximum else np.float32(1 / 127)
        expected = np.clip(np.rint(block * inv_scale), -127, 127).astype(np.int16)
        max_quant_error = max(max_quant_error, int(np.max(np.abs(expected - quant[start:start + tile].astype(np.int16)))))
        max_scale_error = max(max_scale_error, float(abs(scales[index] - expected_scale) / expected_scale))
    return max_quant_error, max_scale_error


def softmax(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    logp = shifted - logsum
    return np.exp(logp), logp


def score_metrics(reference, scores, reference_p, reference_logp):
    error = scores - reference
    centered = error - error.mean(axis=1, keepdims=True)
    p, logp = softmax(scores)
    ref_centered = reference - reference.mean(axis=1, keepdims=True)
    return {
        "count": int(error.size),
        "sse": float(np.square(error).sum()),
        "reference_sse": float(np.square(reference).sum()),
        "centered_sse": float(np.square(centered).sum()),
        "reference_centered_sse": float(np.square(ref_centered).sum()),
        "abs_sum": float(np.abs(error).sum()),
        "max_abs": float(np.max(np.abs(error))),
        "p95_abs": float(np.quantile(np.abs(error), 0.95)),
        "p99_abs": float(np.quantile(np.abs(error), 0.99)),
        "kl_sum": float((reference_p * (reference_logp - logp)).sum()),
        "tv_sum": float(np.abs(p - reference_p).sum() * 0.5),
        "max_probability_error": float(np.max(np.abs(p - reference_p))),
        "top1_changed": int(np.sum(np.argmax(scores, axis=1) != np.argmax(reference, axis=1))),
        "rows": int(len(error)),
    }


def summarize(records):
    counts = sum(r["count"] for r in records)
    rows = sum(r["rows"] for r in records)
    sse = sum(r["sse"] for r in records)
    centered_sse = sum(r["centered_sse"] for r in records)
    return {
        "scores": counts,
        "rows": rows,
        "rmse": float(np.sqrt(sse / counts)),
        "relative_l2": float(np.sqrt(sse / sum(r["reference_sse"] for r in records))),
        "centered_rmse": float(np.sqrt(centered_sse / counts)),
        "centered_relative_l2": float(np.sqrt(centered_sse / sum(r["reference_centered_sse"] for r in records))),
        "mae": sum(r["abs_sum"] for r in records) / counts,
        "max_abs": max(r["max_abs"] for r in records),
        "mean_kl": sum(r["kl_sum"] for r in records) / rows,
        "mean_tv": sum(r["tv_sum"] for r in records) / rows,
        "top1_changed_fraction": sum(r["top1_changed"] for r in records) / rows,
        "max_probability_error": max(r["max_probability_error"] for r in records),
    }


def quantize_reference(x, tile, rotated):
    y = hadamard(x) if rotated else x.astype(np.float32)
    normalization = np.float32(1 / np.sqrt(min(x.shape[-1] & -x.shape[-1], 256))) if rotated else np.float32(1)
    quant = np.empty(y.shape, dtype=np.int8)
    scales = []
    for start in range(0, len(y), tile):
        block = y[start:start + tile]
        maximum = np.max(np.abs(block))
        inv_scale = np.float32(127) / maximum if maximum else np.float32(127)
        scales.append(maximum * (normalization / np.float32(127)) if maximum else np.float32(1 / 127))
        quant[start:start + tile] = np.clip(np.rint(block * inv_scale), -127, 127).astype(np.int8)
    return quant, np.asarray(scales, dtype=np.float32)


def diagnose_first_layer(paths, query_count, output):
    records, heads, validations = [], [], []
    for path in paths:
        meta = json.loads(path.read_text())
        if meta["layer"] != 0:
            continue
        tensors = read_buffers(path, meta)
        rows = np.unique(np.linspace(0, meta["R"] - 1, min(query_count, meta["R"]), dtype=int))
        cols = np.arange(meta["C"])
        for batch in range(meta["B"]):
            centered_keys = {}
            for kh in range(meta["Hk"]):
                k = tensors["k"][batch, :, kh].astype(np.float64)
                centered = (k - k.mean(axis=0)).astype(np.float32)
                for variant in ("plain", "had"):
                    centered_keys[kh, variant] = quantize_reference(centered, meta["k_tile"], variant == "had")
            for head in range(meta["Hq"]):
                kh = head // (meta["Hq"] // meta["Hk"])
                q = tensors["q"][batch, rows, head].astype(np.float64)
                k = tensors["k"][batch, :, kh].astype(np.float64)
                mean = k.mean(axis=0)
                reference = (q @ k.T) * meta["scale"]
                centered_reference = (q @ (k - mean).T) * meta["scale"]
                p, logp = softmax(reference)
                pc, _ = softmax(centered_reference)
                dominant = int(np.argmax(np.abs(mean)))
                identity = {"call": meta["call"], "step": meta["step"], "layer": 0, "batch": batch, "head": head}
                diagnostic = {**identity,
                    "k_dominant_channel": dominant, "k_dominant_mean": float(mean[dominant]),
                    "k_dominant_std": float(k[:, dominant].std()),
                    "k_mean_energy_fraction": float(np.square(mean).sum() / np.square(k).mean(axis=0).sum()),
                    "reference_entropy": float(-(p * logp).sum(axis=1).mean()),
                    "centering_softmax_max_diff": float(np.max(np.abs(pc - p)))}
                for variant in ("plain", "had"):
                    q8 = tensors[variant + "_q"][batch, rows, head].astype(np.float64)
                    sq = tensors[variant + "_q_scale"][batch, head, rows // meta["q_tile"]].astype(np.float64)
                    for centered in (False, True):
                        label = "center_k_" + variant if centered else variant
                        if centered:
                            quant, scales = centered_keys[kh, variant]
                        else:
                            quant = tensors[variant + "_k"][batch, :, kh]
                            scales = tensors[variant + "_k_scale"][batch, kh]
                        sk = scales[cols // meta["k_tile"]].astype(np.float64)
                        scores = (q8 @ quant.astype(np.float64).T) * (sq[:, None] * sk[None, :] * meta["scale"])
                        metrics = score_metrics(centered_reference if centered else reference, scores, p, logp)
                        pp, ll = softmax(scores)
                        records.append({**identity, "variant": label, **metrics})
                        diagnostic[label + "_tv"] = metrics["tv_sum"] / metrics["rows"]
                        diagnostic[label + "_kl"] = metrics["kl_sum"] / metrics["rows"]
                        diagnostic[label + "_entropy"] = float(-(pp * ll).sum(axis=1).mean())
                    operands = [("q", head, meta["q_tile"])]
                    if head % (meta["Hq"] // meta["Hk"]) == 0:
                        operands.append(("k", kh, meta["k_tile"]))
                    for operand, oh, tile in operands:
                        qerr, serr = check_quantizer(tensors[operand][batch, :, oh],
                            tensors[variant + "_" + operand][batch, :, oh],
                            tensors[variant + "_" + operand + "_scale"][batch, oh], tile, variant == "had")
                        validations.append({**identity, "operand": operand, "variant": variant,
                            "max_int8_diff": qerr, "max_scale_relative_diff": serr})
                heads.append(diagnostic)
        print(f"First-layer diagnostics step={meta['step']} complete", flush=True)
    if not heads:
        raise ValueError("No first-layer captures found")
    checks = {"max_int8_diff": max(r["max_int8_diff"] for r in validations),
              "max_scale_relative_diff": max(r["max_scale_relative_diff"] for r in validations)}
    checks["passed"] = checks["max_int8_diff"] <= 1 and checks["max_scale_relative_diff"] < 1e-5
    summary = {"query_rows_requested": query_count, "full_key_softmax": True,
        "centering_softmax_max_diff": max(r["centering_softmax_max_diff"] for r in heads),
        "quantizer_checks": checks,
        "centered_variants": "Offline FP32 K centering and CPU quantization; Q uses captured GPU quantization",
        "aggregate": {v: summarize([r for r in records if r["variant"] == v])
                      for v in ("plain", "had", "center_k_plain", "center_k_had")}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "first-layer.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, data in (("first-layer-heads.csv", heads), ("first-layer-quantizer-checks.csv", validations)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(summary, indent=2))
    if not checks["passed"]:
        raise SystemExit("First-layer GPU quantizer checks failed")


def analyze_capture(meta_path, query_count, key_stride, centered_replay=None):
    meta = json.loads(meta_path.read_text())
    tensors = read_buffers(meta_path, meta)
    if centered_replay:
        for name in ("had_q", "had_k", "had_q_scale", "had_k_scale"):
            original = tensors[name]
            tensors[name] = np.fromfile(centered_replay / (meta_path.stem + "-" + name + ".bin"),
                                       dtype=original.dtype).reshape(original.shape)
        tensors["k_mean"] = np.fromfile(centered_replay / (meta_path.stem + "-k_mean.bin"),
                                       dtype="<f4").reshape(meta["B"], meta["Hk"], meta["D"])
        meta["k_centered"] = True
        meta["k_mean_max_absolute_error"] = float(np.max(np.abs(
            tensors["k_mean"] - tensors["k"].mean(axis=1, dtype=np.float64))))
    if not meta["live_hadamard"]:
        raise ValueError("Capture requires the live Hadamard production quantizer")
    rows = np.unique(np.linspace(0, meta["R"] - 1, min(query_count, meta["R"]), dtype=int))
    cols = np.arange(0, meta["C"], key_stride)
    # With full keys this includes the exact full softmax for each sampled query.
    scores = []
    validations = []
    quantization = []
    for batch in range(meta["B"]):
        for head in range(meta["Hq"]):
            kh = head // (meta["Hq"] // meta["Hk"])
            q = tensors["q"][batch, rows, head].astype(np.float64)
            k = tensors["k"][batch, cols, kh].astype(np.float64)
            reference = (q @ k.T) * meta["scale"]
            p, logp = softmax(reference)
            result = {"call": meta["call"], "step": meta["step"], "layer": meta["layer"], "batch": batch, "head": head}
            for variant in ("plain", "had"):
                q8 = tensors[variant + "_q"][batch, rows, head].astype(np.float64)
                k8 = tensors[variant + "_k"][batch, cols, kh].astype(np.float64)
                sq = tensors[variant + "_q_scale"][batch, head, rows // meta["q_tile"]].astype(np.float64)
                sk = tensors[variant + "_k_scale"][batch, kh, cols // meta["k_tile"]].astype(np.float64)
                quant_scores = (q8 @ k8.T) * (sq[:, None] * sk[None, :] * meta["scale"])
                if centered_replay and variant == "had":
                    # Restore in FP64 for comparable raw-logit metrics only, not GPU softmax.
                    quant_scores += (q @ tensors["k_mean"][batch, kh].astype(np.float64))[:, None] * meta["scale"]
                scores.append({**result, "variant": variant, **score_metrics(reference, quant_scores, p, logp)})
            # CPU checks cover full tiles, including partial sequence tails.
            if head in (0, meta["Hq"] // 2, meta["Hq"] - 1, 30, 40, 41):
                for operand, oh, length, tile in (("q", head, meta["R"], meta["q_tile"]), ("k", kh, meta["C"], meta["k_tile"])):
                    raw = tensors[operand][batch, :, oh].astype(np.float32)
                    base_energy = float(np.square(raw.astype(np.float64)).sum())
                    for variant in ("plain", "had"):
                        quant = tensors[variant + "_" + operand][batch, :, oh]
                        scales = tensors[variant + "_" + operand + "_scale"][batch, oh]
                        quant_input = raw
                        if centered_replay and variant == "had" and operand == "k":
                            quant_input = raw - tensors["k_mean"][batch, oh]
                        qerr, serr = check_quantizer(quant_input, quant, scales, tile, variant == "had")
                        validations.append({**result, "operand": operand, "variant": variant, "max_int8_diff": qerr, "max_scale_relative_diff": serr})
                        reconstructed = quant.astype(np.float32) * np.repeat(scales, tile)[:length, None]
                        if variant == "had":
                            reconstructed = hadamard(reconstructed) / np.float32(np.sqrt(min(meta["D"] & -meta["D"], 256)))
                            if centered_replay and operand == "k":
                                reconstructed += tensors["k_mean"][batch, oh]
                        error = reconstructed.astype(np.float64) - raw
                        quantization.append({**result, "operand": operand, "variant": variant,
                            "relative_l2": float(np.sqrt(np.square(error).sum() / base_energy)),
                            "abs_max": float(np.max(np.abs(raw))),
                            "mean_scale": float(scales.mean())})
    return scores, validations, quantization, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-rows", type=int, default=32)
    parser.add_argument("--key-stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--diagnose-first-layer", action="store_true")
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--centered-replay", type=Path)
    replay.add_argument("--replay-executable", type=Path)
    args = parser.parse_args()
    if args.query_rows < 1 or args.key_stride < 1:
        parser.error("Query count and key stride must be positive")
    paths = sorted(args.capture_directory.glob("capture-*.json"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("No captures found")
    if args.diagnose_first_layer:
        if args.key_stride != 1:
            parser.error("First-layer diagnostics require full keys")
        diagnose_first_layer(paths, args.query_rows, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records, validations, quantization, captures = [], [], [], []
    for path in paths:
        if args.replay_executable:
            metadata = json.loads(path.read_text())
            if metadata["B"] != 1 or metadata["precision"] != "fp16":
                parser.error("GPU replay currently requires batch=1 FP16 captures")
            with tempfile.TemporaryDirectory(prefix="kcenter-replay-") as directory:
                destination = Path(directory)
                subprocess.run([str(args.replay_executable.resolve()), str(path.with_suffix(".bin")),
                                *[str(metadata[k]) for k in ("R", "C", "Hq", "Hk", "D")],
                                str(destination / path.stem)], check=True)
                score, validation, quant, meta = analyze_capture(path, args.query_rows, args.key_stride, destination)
        else:
            score, validation, quant, meta = analyze_capture(path, args.query_rows, args.key_stride, args.centered_replay)
        records.extend(score)
        validations.extend(validation)
        quantization.extend(quant)
        captures.append({k: v for k, v in meta.items() if k != "buffers"})
        plain = summarize([r for r in score if r["variant"] == "plain"])
        rotated = summarize([r for r in score if r["variant"] == "had"])
        print(f"call={meta['call']} step={meta['step']} layer={meta['layer']} "
              f"RMSE {plain['rmse']:.6g} -> {rotated['rmse']:.6g} "
              f"ratio={rotated['rmse']/plain['rmse']:.4f} "
              f"TV {plain['mean_tv']:.6g} -> {rotated['mean_tv']:.6g}", flush=True)
    aggregates = {variant: summarize([r for r in records if r["variant"] == variant]) for variant in ("plain", "had")}
    per_step = {}
    for step in sorted({r["step"] for r in records}):
        per_step[step] = {variant: summarize([r for r in records if r["step"] == step and r["variant"] == variant]) for variant in ("plain", "had")}
    per_layer = {}
    for layer in sorted({r["layer"] for r in records}):
        per_layer[layer] = {variant: summarize([r for r in records if r["layer"] == layer and r["variant"] == variant]) for variant in ("plain", "had")}
    paired = list(zip(records[::2], records[1::2]))
    checks = {"max_int8_diff": max(r["max_int8_diff"] for r in validations),
              "max_scale_relative_diff": max(r["max_scale_relative_diff"] for r in validations)}
    checks["passed"] = checks["max_int8_diff"] <= 1 and checks["max_scale_relative_diff"] < 1e-5
    summary = {"captures": captures, "query_rows_requested": args.query_rows, "key_stride": args.key_stride,
               "reference": "FP64 accumulation of the captured original-basis Q/K, scaled by the model alpha",
               "centered_replay": bool(args.centered_replay or args.replay_executable),
               "replay_logit_metrics": "Row offsets restored in FP64 for reporting only" if args.centered_replay or args.replay_executable else None,
               "full_key_softmax": args.key_stride == 1, "aggregate": aggregates, "per_step": per_step,
               "per_layer": per_layer, "quantizer_checks": checks,
               "head_capture_rmse_improved_fraction": sum(h["sse"] < p["sse"] for p, h in paired) / len(paired)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("head_scores.csv", records), ("quantizer_checks.csv", validations), ("quantization.csv", quantization)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"aggregate": aggregates, "quantizer_checks": checks,
                      "head_capture_rmse_improved_fraction": summary["head_capture_rmse_improved_fraction"]}, indent=2))
    if not checks["passed"]:
        raise SystemExit("GPU quantizer checks failed")


if __name__ == "__main__":
    main()
