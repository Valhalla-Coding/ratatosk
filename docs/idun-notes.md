# Idun notes

- 2026-11-03: Idun Python 3.14.4 installs aiohttp 3.14.5, pytest 9.2.0, pytest-aiohttp 1.1.1 from wheels with no compiler. Venv: /home/master/ratatosk/venv.
- Free space on / is about 9.9 GB; the venv is small.
- ComfyUI launch options before the cutover (read 2026-11-03): `--preview-method auto --use-pytorch-cross-attention --enable-manager --listen`, port 8188 (default), started from Stability Matrix.
- LM Studio server before the cutover: port 11434.
- Models: `/home/master/.lmstudio/models` (28 GB, on the 120 GB SATA SSD that is 91% full). `/mnt/hdd` is a spinning 500 GB drive holding ComfyUI's models.
- Engine: LM Studio's llama-server runs standalone with `LD_LIBRARY_PATH` set to its backend folder plus `backends/vendor/linux-llama-cuda-vendor-v1`; `scripts/install_engine.sh` copies both (about 1.2 GB).

## Cutover (2026-11-05)

Ratatosk now owns the public ports on Idun. Done with the user present.
- ComfyUI (Stability Matrix launch options): `--port 8189 --preview-method auto --use-pytorch-cross-attention --enable-manager`, listening on this machine only. The old instance (still on `--listen`) had to be stopped by hand; Stability Matrix's restart left it running.
- Ratatosk routes: 11434 text (plain), 11435 text (roleplay), 8188 ComfyUI, UI on 5960. ComfyUI is reached at `http://127.0.0.1:8189`.
- The service is a user unit for `master`, enabled at boot, with linger on (`loginctl enable-linger master`).
- Settings before the cutover are kept in `/home/master/ratatosk/data/settings.before-cutover.json`.
- LM Studio is stopped, not uninstalled. If it is started again its server cannot take 11434.
- Verified after the switch: capability check 8/8 on 11434; ComfyUI page and `/system_stats` through 8188 (also from the LAN address); websocket handshake through 8188 gives 101; a queue request with a LAN Host and Origin gets ComfyUI's own 400, not 403; the engine stopped to make room for the image job.
- Not yet verified: a real image generated through 8188 followed by a text request (the ComfyUI memory retest), and a real call from the camera system.
- Rollback: `docs/rollback.md`.

## After the cutover (2026-11-05)

- Real traffic seen through Ratatosk: SillyTavern from Huginn (`node-fetch`, `/v1/completions`) and the camera system from Huginn (`OpenAI/Python`, `/v1/chat/completions`, 12 pictures, about 8.6k tokens in, 62 s per request).
- An image generated through the ComfyUI API on 8188 (Realistic Vision SD 1.5, 512x512, 6 steps; 113 s the first time because the checkpoint loads from the spinning disk, 89 s the second). The text engine stepped aside for it, and after the next text request ComfyUI held 160 MB and the engine 6.2 GB.
- `threads` must be 2: with the vision projector on the GPU, 4 threads made text slower (8.0 to 8.9 tok/s against 12.0 to 12.1 at 2). Raise it only if the projector goes back to the CPU.
- First checkpoint load after a restart is slow because the checkpoints are on the HDD; models for ComfyUI that are used often belong on the SSD or the planned M.2.

## LM Studio removed (2026-11-05)

- Deleted: the app, `~/.lmstudio` (old engine copies, logs, settings, `llmster`), its launcher and `~/.config/LM-Studio`. Freed about 6.5 GB (7.2 GB free became 14 GB).
- Models moved, not copied, to `/home/master/models` (same disk). Ratatosk's `model_folders` and `mmproj_files` point there. LM Studio's bundled embedding model (nomic-embed-text-v1.5, 84 MB) was kept in `/home/master/models/nomic-ai/` for a later embeddings endpoint.
- Archive of the LM Studio chats, saved presets and benchmark results: `/home/master/lmstudio-archive-2026-11-05.tgz` (215 KB). Credentials were not archived.
- `lm-studio-tuner` (in `~/Desktop`) points at the deleted folder and is replaced by Ratatosk's own tuning plan.
- Settings before the move: `data/settings.before-lmstudio-removal.json`.

## What the engine speaks (probed 2026-11-05) versus what Ratatosk passes on

The engine (llama-server) answers: OpenAI chat and completions, OpenAI Responses (`/v1/responses`), Anthropic Messages (`/v1/messages`), and its own `/completion`, `/infill`, `/tokenize`, `/detokenize`, `/apply-template`, `/props`, `/slots`, `/health`, `/lora-adapters`. `/metrics`, embeddings and reranking need flags (501 without them). Ratatosk passes only `/v1/models`, `/v1/chat/completions` and `/v1/completions`; everything else is a 404.

## API coverage (2026-11-05, after SillyTavern's llama.cpp type got only 404s)

Which API a request speaks is decided by its path; no per-port API type is needed. Served through Ratatosk, queued like any generation: `/v1/chat/completions`, `/v1/completions`, `/v1/messages` (Anthropic), `/v1/responses` (OpenAI Responses), `/completion` and `/infill` (llama.cpp). Passed to the engine: `/tokenize`, `/detokenize`, `/apply-template`, `/slots`, `/lora-adapters`. Answered without loading a model: `/props`, `/health`, `/v1/models`. Not served: `/metrics` (needs an engine flag), embeddings and reranking (501: need a separate small engine; the nomic model is in `/home/master/models/nomic-ai/`). Any other path is refused with a clear 404 and appears on the Now tab with status `unsupported`.
