# Ratatosk

Traffic control for one GPU. Ratatosk sits in front of [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (text and vision) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (images) and lets several programs share a single graphics card without stepping on each other.

- **One queue, your order.** Requests wait in turn. A priority list says who goes first, by client address, program or door, and a "drop everything" line can cut in front of a running job.
- **Models swap themselves.** The right model is loaded when a request needs it, and the GPU is freed for ComfyUI when an image is next, and back again afterwards.
- **Speaks the common APIs.** OpenAI chat, completions and Responses, Anthropic Messages, and llama.cpp's own `/completion`, `/infill`, `/tokenize` and friends. Which API a request speaks is read from its path; no per-port setting.
- **Reads what a model can do.** Vision, tool use and reasoning are detected from the model itself and shown in the UI. Each door can switch any of them off.
- **Manages models.** Names a model answers to, delete, and add one by pasting a Hugging Face link.
- **A web page to see it work.** Live queue, GPU use and the programs holding it, a request log you can open up, and settings.

Status: early. Built and used on one machine (Ubuntu, GTX 1070 8 GB). Expect rough edges.

## What you need

- Linux with an NVIDIA GPU (the memory handling reads `nvidia-smi`)
- Python 3.11 or newer
- A CUDA build of `llama-server` from llama.cpp, in the layout described in `docs/idun-notes.md`
- ComfyUI, if you want images. Run it on a different port than the one Ratatosk listens on for ComfyUI (default: ComfyUI on 8189, Ratatosk on 8188)
- GGUF models in a folder Ratatosk can read

## Run it

```bash
python -m venv venv && venv/bin/pip install -e .
venv/bin/python -m ratatosk --config data/settings.json
```

Open `http://<this machine>:5960`. Settings are stored in `data/settings.json` and edited from the Settings tab. `deploy/ratatosk.service` runs it as a systemd user service.

Default doors: `11434` general text, `11435` roleplay (goes first, short replies), `8188` ComfyUI. Point your apps at those. The UI shows each door's address with a copy button.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The tests use a fake `llama-server` and a fake ComfyUI, so no GPU is needed.

## More

- Design and decisions: `docs/superpowers/specs/`
- Build plans: `docs/superpowers/plans/`
- One real deployment, including measurements and how to roll back: `docs/idun-notes.md`, `docs/rollback.md`
- `scripts/capability_check.py` checks what a model can do through Ratatosk (text, chat, streaming, thinking, pictures, JSON schema, tools)

## License

MIT, see `LICENSE`.
