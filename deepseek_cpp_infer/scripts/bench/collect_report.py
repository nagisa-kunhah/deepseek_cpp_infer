#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_log_tail(path: Path, limit: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def format_metric(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def build_report(data: dict, log_tail: str) -> str:
    lines = [
        "# GPU Benchmark Report",
        "",
        f"- Commit: `{data['git_sha']}`",
        f"- Branch: `{data['branch']}`",
        f"- Case: `{data['bench_case']}`",
        f"- Machine: `{data['machine_type']}`",
        f"- GPU: `{data['gpu_type']}`",
        f"- Backend: `{data['backend']}`",
        f"- Model profile: `{data['model_profile']}`",
        f"- Status: `{data['status']}`",
        f"- Failure reason: `{data['failure_reason'] or 'none'}`",
        "",
        "## Metrics",
        "",
        f"- TTFT: `{format_metric(data['ttft_ms'], ' ms')}`",
        f"- Decode throughput: `{format_metric(data['decode_tokens_per_s'], ' tok/s')}`",
        f"- Peak VRAM: `{format_metric(data['peak_vram_mb'], ' MiB')}`",
        f"- Peak RSS: `{format_metric(data['peak_rss_mb'], ' MiB')}`",
        f"- Duration: `{format_metric(data['duration_s'], ' s')}`",
        "",
        "## Baseline Delta",
        "",
        "- Reserved for nightly baseline comparison.",
    ]
    if log_tail:
        lines.extend(
            [
                "",
                "## Log Tail",
                "",
                "```text",
                log_tail,
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--bench-case", required=True)
    parser.add_argument("--machine-type", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--failure-reason", default="")
    parser.add_argument("--ttft-ms", type=float)
    parser.add_argument("--decode-tokens-per-s", type=float)
    parser.add_argument("--peak-vram-mb", type=float)
    parser.add_argument("--peak-rss-mb", type=float)
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = output_dir / "raw.log"

    data = {
        "git_sha": args.git_sha,
        "branch": args.git_branch,
        "bench_case": args.bench_case,
        "machine_type": args.machine_type,
        "gpu_type": args.gpu_type,
        "backend": args.backend,
        "model_profile": args.model_profile,
        "ttft_ms": args.ttft_ms,
        "decode_tokens_per_s": args.decode_tokens_per_s,
        "peak_vram_mb": args.peak_vram_mb,
        "peak_rss_mb": args.peak_rss_mb,
        "duration_s": args.duration_s,
        "status": args.status,
        "failure_reason": args.failure_reason,
    }

    (output_dir / "perf.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = build_report(data, load_log_tail(raw_log))
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
