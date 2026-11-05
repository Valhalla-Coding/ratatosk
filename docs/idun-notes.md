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
