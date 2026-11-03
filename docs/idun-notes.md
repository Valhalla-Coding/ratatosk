# Idun notes

- 2026-11-03: Idun Python 3.14.4 installs aiohttp 3.14.5, pytest 9.2.0, pytest-aiohttp 1.1.1 from wheels with no compiler. Venv: /home/master/ratatosk/venv.
- Free space on / is about 9.9 GB; the venv is small.
- ComfyUI launch options before the cutover (read 2026-11-03): `--preview-method auto --use-pytorch-cross-attention --enable-manager --listen`, port 8188 (default), started from Stability Matrix.
- LM Studio server before the cutover: port 11434.
- Models: `/home/master/.lmstudio/models` (28 GB, on the 120 GB SATA SSD that is 91% full). `/mnt/hdd` is a spinning 500 GB drive holding ComfyUI's models.
- Engine: LM Studio's llama-server runs standalone with `LD_LIBRARY_PATH` set to its backend folder plus `backends/vendor/linux-llama-cuda-vendor-v1`; `scripts/install_engine.sh` copies both (about 1.2 GB).
