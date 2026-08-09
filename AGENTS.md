# Instructions

- The code in this repo is meant to be run on a server with a GPU. Assume that the local machine doesn't have a GPU and run GPU work through the suite scripts' `--remote` option. Do not install GPU dependencies such as torch locally.
- Use `uv` to manage dependencies and virtual envs
- Never run `python -m compileall`
- Use `ruff` and `ty` via `uvx` to format files, linting and type checking.
- When running long tasks on the remote server, prefer to run them inside tmux.
- Try to run all required analysis and computation with generated outputs on the remote itself and just copy the final results (plots, data, etc.) when possible
