#!/usr/bin/env python3

import argparse
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run(cmd, log_path: Path) -> tuple[subprocess.CompletedProcess, float]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - started
    with log_path.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.write(proc.stdout)
        if not proc.stdout.endswith("\n"):
            f.write("\n")
        f.write(f"[exit={proc.returncode} duration_s={duration:.6f}]\n\n")
    return proc, duration


def maybe_make_mock_model(repo_root: Path, output_dir: Path) -> Path:
    model_dir = output_dir / "mock_model"
    cmd = [sys.executable, str(repo_root / "tools" / "make_mock_deepseek_model.py"), str(model_dir)]
    subprocess.run(cmd, check=True)
    return model_dir


def query_peak_vram_mb() -> float | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    values = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--machine-type", required=True)
    parser.add_argument("--gpu-type", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    build_dir = Path(args.build_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = output_dir / "raw.log"
    raw_log.write_text("", encoding="utf-8")

    ds_chat = build_dir / "ds_chat"
    if not ds_chat.exists():
        raise SystemExit(f"Missing ds_chat binary: {ds_chat}")

    temp_dir = None
    status = "success"
    failure_reason = ""
    ttft_ms = None
    decode_tokens_per_s = None
    total_duration_s = None

    try:
        if args.model_profile == "mock":
            temp_dir = Path(tempfile.mkdtemp(prefix="ds_bench_"))
            model_dir = maybe_make_mock_model(repo_root, temp_dir)
        else:
            if not args.model_dir:
                status = "failed"
                failure_reason = "missing_model_dir"
                raise RuntimeError("real_benchmark requires --model-dir")
            model_dir = Path(args.model_dir)

        info_proc, _ = run([str(ds_chat), str(model_dir), "info"], raw_log)
        if info_proc.returncode != 0:
            status = "failed"
            failure_reason = "info_failed"
            raise RuntimeError("info command failed")

        ttft_proc, ttft_duration = run(
            [
                str(ds_chat),
                str(model_dir),
                "run",
                "--prompt-ids",
                args.prompt_ids,
                "--backend",
                args.backend,
            ],
            raw_log,
        )
        if ttft_proc.returncode != 0:
            status = "failed"
            failure_reason = "run_failed"
            raise RuntimeError("run command failed")
        ttft_ms = ttft_duration * 1000.0

        gen_proc, gen_duration = run(
            [
                str(ds_chat),
                str(model_dir),
                "generate",
                "--prompt",
                args.prompt,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--backend",
                args.backend,
            ],
            raw_log,
        )
        if gen_proc.returncode != 0:
            status = "failed"
            failure_reason = "generate_failed"
            raise RuntimeError("generate command failed")

        total_duration_s = gen_duration
        if args.max_new_tokens > 0 and gen_duration > 0:
            decode_tokens_per_s = args.max_new_tokens / gen_duration
    except Exception as exc:  # noqa: BLE001
        if not failure_reason:
            failure_reason = str(exc)
        with raw_log.open("a", encoding="utf-8") as f:
            f.write(f"[failure] {exc}\n")
    finally:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
        peak_vram_mb = query_peak_vram_mb() if args.backend == "cuda" else None
        collect_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "bench" / "collect_report.py"),
            "--output-dir",
            str(output_dir),
            "--git-sha",
            args.git_sha,
            "--git-branch",
            args.git_branch,
            "--bench-case",
            "real_benchmark",
            "--machine-type",
            args.machine_type,
            "--gpu-type",
            args.gpu_type,
            "--backend",
            args.backend,
            "--model-profile",
            args.model_profile,
            "--status",
            status,
            "--failure-reason",
            failure_reason,
            "--peak-rss-mb",
            f"{peak_rss_mb:.3f}",
        ]
        if ttft_ms is not None:
            collect_cmd.extend(["--ttft-ms", f"{ttft_ms:.3f}"])
        if decode_tokens_per_s is not None:
            collect_cmd.extend(["--decode-tokens-per-s", f"{decode_tokens_per_s:.3f}"])
        if peak_vram_mb is not None:
            collect_cmd.extend(["--peak-vram-mb", f"{peak_vram_mb:.3f}"])
        if total_duration_s is not None:
            collect_cmd.extend(["--duration-s", f"{total_duration_s:.3f}"])
        subprocess.run(collect_cmd, check=True)

        status_payload = {
            "status": status,
            "failure_reason": failure_reason,
            "output_dir": str(output_dir),
        }
        (output_dir / "status.local.json").write_text(
            json.dumps(status_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
