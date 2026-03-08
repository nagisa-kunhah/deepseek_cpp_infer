# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## Project Commands (DeepSeek C++ Infer)

Build:

```bash
cd /home/nagisa/.openclaw/workspace/deepseek_cpp_infer
cmake -S . -B build
cmake --build build -j
```

Run:

```bash
./build/ds_chat <model_dir> info
./build/ds_chat <model_dir> verify
```

## Pitfalls / Notes

- `rg` (ripgrep) is not installed in this environment; use `grep -RIn` and `find` for fast searches.
- Avoid committing build outputs; `.gitignore` covers `**/build/**`.
- For large `*.safetensors` shards, prefer header-only parsing or mmap/lazy-loading (do not slurp multi-GB files into RAM by default).

## OpenClaw Service

```bash
openclaw status
openclaw gateway status
```

---

Add whatever helps you do your job. This is your cheat sheet.
