# MEMORY.md

- 2026-03-08: Goal is a runnable DeepSeek-V2-Lite inference framework in `deepseek_cpp_infer/` (C++17 + CMake). Early focus is on building the framework/wiring and getting an end-to-end run (at least model file discovery + config + safetensors header verification), before full numerical inference.
- 2026-03-08: Long-term workflow decision: do not rely on chat/session memory for project state; persist requirements/progress in repo files (e.g. `deepseek_cpp_infer/README.md`, `deepseek_cpp_infer/TODO.md`, `memory/YYYY-MM-DD.md`) and keep the codebase in small, runnable, committed milestones.
