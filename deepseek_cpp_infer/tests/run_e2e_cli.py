#!/usr/bin/env python3

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return proc.stdout


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print("usage: run_e2e_cli.py <repo_root> <ds_chat> [cpu|cuda]", file=sys.stderr)
        return 1

    repo_root = Path(sys.argv[1]).resolve()
    ds_chat = Path(sys.argv[2]).resolve()
    backend = sys.argv[3] if len(sys.argv) == 4 else "cpu"
    model_dir = Path(tempfile.mkdtemp(prefix="ds_mock_model_"))
    try:
        make_model = repo_root / "tools" / "make_mock_deepseek_model.py"
        run([sys.executable, str(make_model), str(model_dir)])

        info_out = run([str(ds_chat), str(model_dir), "info"])
        verify_out = run([str(ds_chat), str(model_dir), "verify"])
        strict_out = run([str(ds_chat), str(model_dir), "strict"])
        load_out = run([str(ds_chat), str(model_dir), "load"])
        run_ids_out = run([str(ds_chat), str(model_dir), "run", "--prompt-ids", "0,1,2", "--backend", backend])
        run_prompt_out = run([str(ds_chat), str(model_dir), "run", "--prompt", "hello world!", "--backend", backend])
        gen_out = run([
            str(ds_chat),
            str(model_dir),
            "generate",
            "--prompt",
            "hello world!",
            "--max-new-tokens",
            "3",
            "--backend",
            backend,
        ])

        expected = [
            ("info", "Found 1 shard(s) on disk.", info_out),
            ("verify", "OK (parsed headers).", verify_out),
            ("strict", "MoE layers:", strict_out),
            ("load", "Layer registry:", load_out),
            ("run_ids", "Next token id:", run_ids_out),
            ("run_prompt", "Next token text:", run_prompt_out),
            ("generate", "Generated ids:", gen_out),
        ]
        for name, needle, haystack in expected:
            if needle not in haystack:
                raise RuntimeError(f"{name} output missing expected text: {needle}\n{haystack}")
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
