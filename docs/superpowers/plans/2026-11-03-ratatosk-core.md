# Ratatosk Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A gateway service on Idun that takes over ports 11434 and 8188, queues text and image jobs on the one GPU, swaps the model and ComfyUI's memory between them, and shows the state on a Now and a Settings tab.

**Architecture:** One asyncio service (aiohttp). Each route is a listening port that either forwards OpenAI-style requests to a `llama-server` child process that Ratatosk starts and stops, or proxies ComfyUI. All GPU work goes through one priority queue; a scheduler decides when to stop the engine or free ComfyUI's memory.

**Tech Stack:** Python 3.11+ (Idun has 3.14.4), aiohttp, SQLite (stdlib), pytest + pytest-aiohttp. No Node, no build step.

**Spec:** `docs/superpowers/specs/2026-11-03-ratatosk-design.md` (sections 4 to 7 and 13.1 are this plan).

## Global Constraints

- Default routes: 11434 llm/plain, 11435 llm/rp, 8188 comfy/plain; UI port 5960. The engine listens on 127.0.0.1:8081, ComfyUI moves to 127.0.0.1:8189.
- Priorities: P0 roleplay (`rp` profile), P1 other LLM, P2 images, P3 tuning. Only P3 is preempted.
- One job at a time; one model; `--parallel 1`.
- Context buckets 8192, 16384, 32768, 65536, 131072. Needed context is `ceil(1.25 * (prompt_tokens + capped_max_tokens))`. Move up when it does not fit; move down only when every need in the last 600 s fits two buckets lower.
- `max_tokens` cap default 1024. Idle engine stop after 30 minutes. Free-space reserve default 5 GB.
- Prompt tokens: exact via the engine's `/tokenize` when loaded, otherwise `len(text) // 3`.
- Swap only when the job type changes: image next means stop the engine; text next means `POST /free` with `{"unload_models": true, "free_memory": true}` on ComfyUI first.
- LM Studio is not uninstalled by this plan; the user decides after verification.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Failure modes the spec implies that most likely bite, each pinned by a test in the owning task:

1. A prompt bigger than the largest bucket: a clear 413, engine untouched (Task 8).
2. A client disconnects mid-stream: the job is cancelled and the next job runs (Tasks 4 and 8).
3. An image request waits behind a long text job past the limit: a 503, not a hang (Tasks 7 and 9).
4. Requested model unknown, or its file is on an unmounted drive: a clear error, no crash loop (Tasks 3 and 8).
5. The engine fails to start (memory full, bad flags): that job fails with 502, state is reset, the queue keeps going (Tasks 6 and 7).

## File Structure

```
pyproject.toml            package, deps, pytest config
ratatosk/__init__.py      __version__
ratatosk/settings.py      Settings, Route, load_settings, save_settings
ratatosk/db.py            Db (SQLite request log)
ratatosk/models.py        Model, scan, resolve
ratatosk/gpuqueue.py      Job, GpuQueue, priority constants
ratatosk/buckets.py       BUCKETS, needed_ctx, pick_bucket
ratatosk/comfy.py         Comfy client
ratatosk/engine.py        Engine (llama-server lifecycle), EngineError
ratatosk/scheduler.py     Scheduler, GpuBusy
ratatosk/gateway_llm.py   LLM route handlers
ratatosk/gateway_comfy.py ComfyUI route handlers
ratatosk/web.py           /api/status, /api/settings, HTML page
ratatosk/app.py           build and run all listeners
ratatosk/__main__.py      entry point
ratatosk/static/index.html  Now and Settings tabs
tests/fakes/fake_llama.py   fake llama-server (a script)
tests/fakes/fake_comfy.py   fake ComfyUI (aiohttp app factory)
deploy/ratatosk.service, scripts/install_engine.sh, scripts/smoke_idun.sh, docs/rollback.md
```

---

### Task 1: Scaffold and Idun toolchain check

**Files:**
- Create: `pyproject.toml`, `ratatosk/__init__.py`, `.gitignore`, `tests/test_smoke.py`, `README.md` (five lines: what it is, how to run tests)

**Interfaces:**
- Produces: `ratatosk.__version__ == "0.1.0"`; pytest configured with `asyncio_mode = "auto"`.

- [ ] **Step 1: Write the failing test** `tests/test_smoke.py::test_package_imports` asserting `ratatosk.__version__ == "0.1.0"`.
- [ ] **Step 2: Run** `python -m pytest tests/test_smoke.py -v`. Expected: FAIL (no module).
- [ ] **Step 3: Create the files.** Dependencies: `aiohttp>=3.10`; dev extras: `pytest`, `pytest-aiohttp`. `.gitignore` covers `.venv/`, `data/`, `__pycache__/`.
- [ ] **Step 4: Install and run** `python -m pip install -e ".[dev]"` then the test. Expected: PASS.
- [ ] **Step 5: Idun toolchain check** (needs a fresh Idun Grant from the user). Over SSH as `claude`: create `/home/master/ratatosk/venv` with `python3 -m venv`, then `pip install aiohttp pytest pytest-aiohttp`. Expected: installs from wheels with no compiler. If it fails, record the error in `docs/idun-notes.md` and install Python 3.12 with `uv` before going on.
- [ ] **Step 6: Commit** `feat: project scaffold`.

### Task 2: Settings and request log

**Files:**
- Create: `ratatosk/settings.py`, `ratatosk/db.py`; Test: `tests/test_settings.py`, `tests/test_db.py`

**Interfaces:**
- Produces: `Route(port: int, kind: str, profile: str = "plain")` with kind `"llm"` or `"comfy"`; `Settings` dataclass with fields and defaults: `routes` (the three default routes), `ui_port=5960`, `engine_port=8081`, `comfy_url="http://127.0.0.1:8189"`, `engine_dir="/home/master/ratatosk/engine"`, `threads=max(1, os.cpu_count()//2)`, `model_folders=[{"path": "/home/master/.lmstudio/models", "label": "SSD models"}]`, `default_model=""`, `aliases={}`, `max_tokens_cap=1024`, `idle_minutes=30`, `reserve_gb=5`, `image_wait_seconds=600`, `data_dir="data"`; `load_settings(path: Path) -> Settings` (missing file gives defaults, unknown keys ignored); `save_settings(s: Settings, path: Path) -> None`; `Db(path: Path)` with `log_request(route: str, profile: str, model: str, ctx: int, prompt_tokens: int, gen_tokens: int, seconds: float, status: str) -> int` and `recent(limit: int = 20) -> list[dict]` newest first.

- [ ] **Step 1: Write failing tests:** `test_defaults` (the three routes exactly as in Global Constraints, `ui_port == 5960`, `idle_minutes == 30`); `test_roundtrip_ignores_unknown_keys`; `test_log_and_recent` (ids increase, `recent(1)` returns the newest row with all fields).
- [ ] **Step 2: Run** `python -m pytest tests/test_settings.py tests/test_db.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules. JSON for settings, one `requests` table with an auto id and a timestamp for the log.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: settings and request log`.

### Task 3: Model registry

**Files:**
- Create: `ratatosk/models.py`; Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Model(id: str, path: Path, size: int, folder: Path)` (frozen dataclass); `scan(folders: list[Path]) -> list[Model]` (recursive `*.gguf`, skipping files with `mmproj` in the name, a missing folder gives nothing; id is the lower-case file stem); `resolve(name: str, models: list[Model], aliases: dict[str, str], default: str) -> Model | None` (alias, then case-insensitive id or stem match, then the default, else `None`).

The drive speed class and the free-space guard from spec section 7 are not needed until downloads exist, so they are built in the Models sub-project.

- [ ] **Step 1: Write failing tests:** `test_scan_skips_mmproj_and_non_gguf` on a temp tree with a nested folder; `test_resolve_order` (alias beats stem match, default used when unknown, `None` when unknown with no default); `test_scan_missing_folder_returns_empty` (an unmounted drive must not raise; Review Focus 4).
- [ ] **Step 2: Run** `python -m pytest tests/test_models.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both functions.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: model registry`.

### Task 4: GPU queue

**Files:**
- Create: `ratatosk/gpuqueue.py`; Test: `tests/test_gpuqueue.py`

**Interfaces:**
- Produces: constants `P_RP=0, P_LLM=1, P_IMAGE=2, P_TUNE=3`; `Job(name: str, priority: int, run: Callable[[], Awaitable[Any]], after: Callable[[], Awaitable[None]] | None = None)`; `GpuQueue()` with `start()`, `async stop()`, `async submit(job: Job) -> Any` (returns `run`'s result as soon as `run` finishes; the runner then awaits `after` before taking the next job), `snapshot() -> dict` with keys `running` (name or None) and `pending` (list of names).

- [ ] **Step 1: Write failing tests:** `test_priority_order` (while one job runs, jobs with priorities 2, 0, 1 run as 0, 1, 2); `test_one_at_a_time` (a counter never exceeds 1); `test_cancel_pending_never_runs`; `test_cancel_running_frees_queue` (cancelling the submitter cancels `run`, and the next job still runs; Review Focus 2); `test_after_runs_after_submit_returns_before_next_job`; `test_tune_job_preempted` (a P3 job running is cancelled when a P0 to P2 job is submitted); `test_snapshot_shape`.
- [ ] **Step 2: Run** `python -m pytest tests/test_gpuqueue.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with a heap ordered by `(priority, sequence)` and one runner task. Name the module `gpuqueue` so it does not shadow the standard library.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: priority gpu queue`.

### Task 5: Context buckets

**Files:**
- Create: `ratatosk/buckets.py`; Test: `tests/test_buckets.py`

**Interfaces:**
- Produces: `BUCKETS = (8192, 16384, 32768, 65536, 131072)`; `needed_ctx(prompt_tokens: int, max_tokens: int) -> int`; `pick_bucket(needed: int, current: int | None, recent: list[tuple[float, int]], now: float, limit: int = 131072) -> int | None` (`recent` holds `(timestamp, needed)`; returns `None` when `needed` exceeds the largest bucket not above `limit`).

- [ ] **Step 1: Write failing tests:** `needed_ctx(8000, 1024) == 11280`; nothing loaded and needed 11280 gives 16384; needed above current moves up to the smallest fitting bucket; current 32768 with all needs in the last 600 s at or below 8192 moves down to 8192; one need of 20000 in that window keeps 32768; needs older than 600 s are ignored; needed 200000 returns `None` (Review Focus 1).
- [ ] **Step 2: Run** `python -m pytest tests/test_buckets.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three rules from Global Constraints.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: context buckets`.

### Task 6: ComfyUI client and engine manager

**Files:**
- Create: `ratatosk/comfy.py`, `ratatosk/engine.py`, `tests/fakes/fake_llama.py`, `tests/fakes/fake_comfy.py`; Test: `tests/test_comfy.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `Settings` (Task 2).
- Produces: `Comfy(base_url: str, session: aiohttp.ClientSession)` with `async pending() -> int` (running plus pending from `GET /queue`) and `async free() -> None` (`POST /free` with the body from Global Constraints). `EngineError(Exception)`. `Engine(settings: Settings)` with `loaded: tuple[Path, int] | None`, `url: str` (`http://127.0.0.1:<engine_port>`), `argv(model: Path, ctx: int) -> list[str]`, `async ensure(model: Path, ctx: int) -> None` (starts, or restarts when model or context differ, then waits for `GET /health` to return 200, polling every 0.5 s for up to 300 s; raises `EngineError` and leaves `loaded = None` on early exit or timeout), `async stop() -> None`, `async count_tokens(text: str) -> int | None` (via `POST /tokenize`, `None` when not running).
- `argv` is exactly: `[<engine_dir>/backend/llama-server, "--model", <path>, "--ctx-size", <ctx>, "--host", "127.0.0.1", "--port", <engine_port>, "--fit", "on", "--flash-attn", "on", "--parallel", "1", "--threads", <threads>]`, run with `LD_LIBRARY_PATH=<engine_dir>/backend:<engine_dir>/vendor`.
- Fakes: `fake_llama.py` is a script taking the same flags that serves `/health`, `/tokenize` (one token per word), and `/v1/completions` (streams three SSE chunks then `[DONE]`; appends each request body to the file named by env `FAKE_LOG`; env `FAKE_START_DELAY` and `FAKE_CHUNK_DELAY` add delays; env `FAKE_EXIT=1` exits at once). `fake_comfy.py` provides `make_app(state)` with `/queue`, `/free` (records the call), `/prompt` (adds a pending item that finishes after `state.job_seconds`), `/system_stats` and a `/ws` echo.

- [ ] **Step 1: Write failing tests:** `test_argv_exact`; `test_ensure_starts_and_waits_for_health` (fake start delay 1 s); `test_ensure_same_args_does_not_restart` (same pid); `test_ensure_new_ctx_restarts` (pid differs); `test_ensure_failure_resets_state` (fake with `FAKE_EXIT=1` raises `EngineError`, `loaded is None`; Review Focus 5); `test_count_tokens`; `test_stop_kills_process`; `test_comfy_pending_sums_running_and_pending`; `test_comfy_free_body`.
- [ ] **Step 2: Run** `python -m pytest tests/test_comfy.py tests/test_engine.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules. Use `asyncio.create_subprocess_exec` and make `Engine` accept an optional `argv_override` for tests so the fake script can stand in for `llama-server`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: comfy client and engine manager`.

### Task 7: Scheduler

**Files:**
- Create: `ratatosk/scheduler.py`; Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `GpuQueue`, `Job` (Task 4), `pick_bucket` (Task 5), `Comfy` and `Engine` (Task 6), `Model` (Task 3), `Settings` (Task 2).
- Produces: `GpuBusy(Exception)`; `Scheduler(settings, queue, engine, comfy, clock=time.monotonic)` with `async run_llm(priority: int, model: Model, needed: int, handler: Callable[[str], Awaitable[Any]]) -> Any` (inside the queue job: if the last job was an image, `comfy.free()` first; then `engine.ensure(model.path, bucket)` with the bucket from `pick_bucket` over the recent needs; then `handler(engine.url)`), `async run_image(handler: Callable[[], Awaitable[Any]]) -> Any` (inside the job: stop the engine if running, then `handler()`; the `after` hook polls `comfy.pending()` every 1 s until it is 0, holding the slot), `async idle_loop() -> None` (stops the engine after `idle_minutes` without jobs), `limit_bucket() -> int`. Raises `GpuBusy` when an image job has not started running `image_wait_seconds` after it was submitted (the wait is measured only until the job starts; a running job is never cut off, and a timed-out job is removed from the queue).

- [ ] **Step 1: Write failing tests** with simple recording fakes for engine, comfy and clock: `test_llm_after_image_frees_comfy_first`; `test_image_after_llm_stops_engine_first`; `test_llm_after_llm_does_neither`; `test_bucket_used` (needed 11280 calls `ensure` with 16384); `test_idle_stops_engine` (fake clock past 30 minutes); `test_engine_failure_fails_job_and_queue_continues` (Review Focus 5); `test_image_holds_slot_until_comfy_idle`; `test_image_wait_exceeded_raises_gpubusy` (Review Focus 3).
- [ ] **Step 2: Run** `python -m pytest tests/test_scheduler.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Track `last_kind` (`"llm"`, `"image"` or `None`) and the recent needs list; reset `last_kind` to `None` if `ensure` raises.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: scheduler with swapping and idle stop`.

### Task 8: LLM gateway

**Files:**
- Create: `ratatosk/gateway_llm.py`; Test: `tests/test_gateway_llm.py`

**Interfaces:**
- Consumes: `Scheduler`, `Engine`, `Db`, `Settings`, `scan`, `resolve`, `needed_ctx`.
- Produces: `make_llm_app(route: Route, ctx: AppContext) -> web.Application`, where `AppContext` (defined in this module) bundles `settings`, `scheduler`, `engine`, `db`, `models_provider: Callable[[], list[Model]]`. Handles `POST /v1/completions` and `POST /v1/chat/completions` and `GET /v1/models`. Behavior: cap `max_tokens` (also `n_predict`) to `max_tokens_cap`; count prompt tokens (Global Constraints); `needed > largest allowed bucket` gives 413 JSON `{"error": ...}`; unresolvable model gives 404; engine failure gives 502; priority is `P_RP` for the `rp` profile else `P_LLM`; the handler streams the engine's response through unchanged (SSE or plain); one `Db.log_request` row per request.

- [ ] **Step 1: Write failing tests** using the real `Engine` pointed at `fake_llama.py`: `test_completion_streams_through`; `test_max_tokens_capped` (client asks 65536, fake's log shows 1024); `test_unknown_model_404_no_default` and `test_missing_file_clear_error` (model file in an unmounted folder; Review Focus 4); `test_oversized_prompt_413_engine_untouched` (Review Focus 1); `test_models_endpoint_lists_aliases_and_ids`; `test_client_disconnect_cancels_and_next_request_runs` (slow fake, client closes mid-stream, a second request then completes; Review Focus 2); `test_rp_profile_runs_before_plain` (two pending requests, rp first); `test_request_logged`.
- [ ] **Step 2: Run** `python -m pytest tests/test_gateway_llm.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the handlers. Use `aiohttp.ClientSession` streaming with `iter_any()`; the server runners are started with `handler_cancellation=True` so a client disconnect cancels the handler task.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: llm gateway`.

### Task 9: ComfyUI gateway

**Files:**
- Create: `ratatosk/gateway_comfy.py`; Test: `tests/test_gateway_comfy.py`

**Interfaces:**
- Consumes: `Scheduler.run_image`, `GpuBusy`, `Settings.comfy_url`, `fake_comfy`.
- Produces: `make_comfy_app(route: Route, ctx: AppContext) -> web.Application`. `POST /prompt` goes through `run_image` (forwarding to ComfyUI and returning its response); a `GpuBusy` gives 503 `{"error": "GPU busy"}`. Every other path, and the `/ws` websocket, is proxied straight through with no queueing.

- [ ] **Step 1: Write failing tests:** `test_passthrough_system_stats`; `test_prompt_waits_behind_llm_job` (a slow LLM job runs; the engine is stopped before the fake sees `/prompt`); `test_prompt_503_when_wait_exceeded` (Review Focus 3); `test_second_prompt_waits_until_comfy_idle`; `test_ws_echo_both_directions`.
- [ ] **Step 2: Run** `python -m pytest tests/test_gateway_comfy.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the proxy; copy request headers except `Host`, stream bodies, and relay websocket messages in both directions.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: comfyui gateway`.

### Task 10: Web UI and wiring

**Files:**
- Create: `ratatosk/web.py`, `ratatosk/app.py`, `ratatosk/__main__.py`, `ratatosk/static/index.html`; Test: `tests/test_web.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `GET /api/status` returning `{"queue": snapshot, "engine": {"model": str, "ctx": int} | null, "comfy_pending": int, "gpu": {"used_mb": int, "total_mb": int} | null, "recent": [...]}` (`gpu` comes from `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`, `null` if missing); `GET /api/settings`; `PUT /api/settings` (validates route kind and profile, 400 otherwise, saves to `data/settings.json`); `GET /` serving the page with two tabs, Now and Settings, polling status every 2 s; `async run(settings_path: Path) -> None` in `app.py` starting one aiohttp site per route plus the UI port; `python -m ratatosk --config data/settings.json`.

- [ ] **Step 1: Write failing tests:** `test_status_shape`; `test_settings_put_persists_and_rejects_bad_kind`; `test_index_has_both_tabs`; `test_run_binds_all_ports_and_shuts_down_engine` (ports chosen free by the test).
- [ ] **Step 2: Run** `python -m pytest tests/test_web.py tests/test_app.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The page is plain HTML and a small script, no framework. The Settings tab edits routes and the main numeric settings as a form.
- [ ] **Step 4: Run the whole suite** `python -m pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: web ui and app wiring`.

### Task 11: Deployment assets and Idun cutover

**Files:**
- Create: `scripts/install_engine.sh`, `deploy/ratatosk.service`, `docs/rollback.md`, `scripts/smoke_idun.sh`

**Interfaces:**
- Produces: engine copied to `/home/master/ratatosk/engine/{backend,vendor}`; a systemd user unit (`Restart=always`) for user `master`; the rollback steps from the spec (stop Ratatosk, relaunch ComfyUI on 0.0.0.0:8188, start `llama-server` or LM Studio on 11434); a smoke script.

This task changes Idun and needs the user present. Do nothing here without the user's go at each numbered step.

- [ ] **Step 1: Write the assets.** `install_engine.sh` copies the newest `llama.cpp-linux-x86_64-nvidia-cuda-avx2-*` backend to `engine/backend` and `vendor/linux-llama-cuda-vendor-v1` to `engine/vendor`, then runs `llama-server --version` with the library path. The unit sets `WorkingDirectory=/home/master/ratatosk` and runs the venv's `python -m ratatosk`. The smoke script sends a short completion to :11434, reads `/api/status`, and prints `nvidia-smi` memory before and after.
- [ ] **Step 2: Install on Idun** (with a Grant): copy the repo and venv, run `install_engine.sh`. Expected: `llama-server --version` prints a version. Nothing is listening on the public ports yet; test on temporary ports 21434 and 18188 first by editing `data/settings.json`.
- [ ] **Step 3: Test side by side.** Start Ratatosk on the temporary ports with LM Studio and ComfyUI untouched. Run the smoke script. Expected: a completion streams, `/api/status` shows the loaded model, and `nvidia-smi` shows the model loaded at the chosen context.
- [ ] **Step 4: Cutover (user present).** Together: close LM Studio's server, change ComfyUI's launch options in Stability Matrix to `--listen 127.0.0.1 --port 8189`, set the real ports in Settings, start the service with `systemctl --user enable --now ratatosk`. Point nothing else; existing clients already use 11434 and 8188.
- [ ] **Step 5: Acceptance** (spec 13.1). A SillyTavern chat works through 11434; a ComfyCharacters render works through 8188; an image then a chat then an image shows the memory freed each time in `nvidia-smi`; a larger prompt causes exactly one reload. Record the results in `docs/idun-notes.md`.
- [ ] **Step 6: Commit** `feat: deployment assets and acceptance notes`.
