# Instructions

- The code in this repo is meant to be run of a server with a GPU. Assume that the local machine you're running on doesn't have a GPU and needs to run everything that requires a GPU on a remote server so no use of installing GPU related deps (like torch) locally.
- Use `uv` to manage dependencies and virtual envs
- Never run `python -m compileall`
- Use `ruff` and `ty` via `uvx` to format files, linting and type checking.
