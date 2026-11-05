# Settings, Priority List and Model Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved Settings page and the backend it needs: per-door limits and feature switches, an ordered priority list with "drop everything", model capabilities, names, delete and add-by-Hugging-Face-link.

**Architecture:** Priority becomes a number computed per request from an ordered rule list (`ratatosk/priority.py`), replacing the three fixed constants; the queue gains a `preempt` flag that cancels the running job. Per-door request filters live in the LLM gateway. Model facts come from a small GGUF metadata reader. The UI is the mockup wired to new `/api/*` endpoints.

**Tech Stack:** Python 3.11+, aiohttp, SQLite, pytest + pytest-aiohttp, plain HTML/JS (no build step).

**Spec:** `docs/superpowers/specs/2026-11-05-settings-and-priority-design.md` (mockup: `docs/mockups/settings.html`; base design: `2026-11-03-ratatosk-design.md`)

## Global Constraints

- Nothing program-, camera- or machine-specific in code, defaults or UI text. Defaults use the home folder (`Path.home()`), never `/home/master`.
- No new runtime dependency (aiohttp only). Tests need no GPU and no network: use `tests/fakes/`.
- Tuning jobs stay last and preemptible; with the default priority list behaviour is identical to today (roleplay, general text, ComfyUI).
- Old `settings.json` files must still load (unknown keys ignored, `max_tokens_cap*` migrate to door `max_tokens`).
- UI works at phone width, keyboard-reachable, no motion when `prefers-reduced-motion`.
- Existing 121 tests keep passing; run the whole suite at the end of every task.
- Ratatosk's deletes (model files) always need an explicit user action in the UI; the API refuses while the engine serves that model.

## Review Focus

- A drop-everything request arrives while an image is being generated: ComfyUI is interrupted, the image client gets 503 "dropped by <name>", the GPU is not left held.
- Two drop-everything lines: the later arrival drops the earlier running one only if its line is higher in the list; otherwise it waits.
- A model file that is not valid GGUF or lacks a chat template: shows "text only", never crashes the scan or the page.
- Deleting the model the engine is serving, or the only model: refused with a clear message; a model's shared vision file is not deleted while another model uses it.
- A Hugging Face download that stops half-way, or a disk without room: no partial file listed as a model; resuming works; error shown in the UI.

---

### Task 1: Settings shape and migration

**Files:** Modify `ratatosk/settings.py`, `ratatosk/web.py` (`validate_settings`); Test `tests/test_settings.py`, `tests/test_web.py`.

**Interfaces:**
- Produces: `Route` gains `max_tokens: int | None = None` and `allow: dict` (keys `vision`, `reasoning`, `tools`, default all true). `Settings.priority: list[dict]` (shapes in the spec) with a default list built from the routes by `default_priority(routes)`. `Settings.max_tokens_for(route) -> int`.

- [ ] Write failing tests: `test_old_caps_migrate_to_doors` (file with `max_tokens_cap=512, max_tokens_cap_plain=4096` gives rp route 512, plain 4096), `test_default_priority_matches_old_behaviour` (rp door, plain door, comfy door in that order), `test_priority_validation_rejects_unknown_door_and_bad_ip` (via `validate_settings`).
- [ ] Run them, see them fail.
- [ ] Implement in `settings.py` and `validate_settings`; keep `max_tokens_cap*` readable for old files but stop writing them.
- [ ] Run the whole suite, commit.

### Task 2: Rule resolution

**Files:** Create `ratatosk/priority.py`; Test `tests/test_priority.py`.

**Interfaces:**
- Produces: `@dataclass Rank: priority: int, preempt: bool, label: str`; `rank_for(rules: list[dict], route: Route, ip: str, agent: str) -> Rank`. A matching line at index `i` gives `priority=i`, or `-1000 + i` with `preempt=True` when `drop` is set; no match gives `priority=len(rules)`, `label="Everything else"`. `P_TUNE` becomes `10**6`.

- [ ] Write failing tests: first match wins; client rule with agent substring matches case-insensitively and an address-only rule matches any agent; door rule matches by port; unmatched gets `len(rules)`; drop lines rank above all normal lines; `P_TUNE` is greater than any `len(rules)`.
- [ ] Run, see fail; implement `rank_for`; run whole suite; commit.

### Task 3: Queue preemption and dropped requests

**Files:** Modify `ratatosk/gpuqueue.py`, `ratatosk/scheduler.py`, `ratatosk/gateway_llm.py`, `ratatosk/gateway_comfy.py`, `ratatosk/comfy.py` (add `interrupt()`), `ratatosk/db.py` (status `dropped`, column `dropped_by`); Test `tests/test_gpuqueue.py`, `tests/test_gateway_llm.py`, `tests/test_gateway_comfy.py`, `tests/fakes/fake_comfy.py` (record `/interrupt`).

**Interfaces:**
- Consumes: `Rank` from Task 2.
- Produces: `Job.preempt: bool = False` and `Job.owner: str = ""`; `class Preempted(Exception)` with `.by: str`; `GpuQueue.submit` kills the running job when `job.preempt` and the running job's `priority` is greater, raising `Preempted(by=job.owner)` in the killed job's caller. `Scheduler.run_llm(rank, model, needed, handler)` and `run_image(rank, handler)` take a `Rank`. A killed image job calls `Comfy.interrupt()` before the queue moves on.

- [ ] Write failing tests: `test_drop_line_cancels_running_text_job` (second request flagged drop; first gets 503 body containing "dropped by"; second completes), `test_drop_line_interrupts_comfy` (fake ComfyUI saw `/interrupt`, the held GPU is released), `test_equal_or_lower_drop_does_not_preempt`, `test_dropped_request_logged_with_name`.
- [ ] Run, see fail; implement; map `Preempted` to 503 in both gateways; run whole suite; commit.

### Task 4: Gateways use the priority list

**Files:** Modify `ratatosk/gateway_llm.py`, `ratatosk/gateway_comfy.py`, `ratatosk/app.py`; Test `tests/test_gateway_llm.py`, `tests/test_gateway_comfy.py`.

**Interfaces:** Consumes `rank_for`, `Settings.priority`, `Settings.max_tokens_for`. Client address comes from `request.remote`, program from the `User-Agent` header. Settings changes take effect without restart (read from the live settings object per request).

- [ ] Write failing tests: a client line moves a plain-door request ahead of a queued roleplay-door request; with the default list the old order holds; the longest reply for a door comes from the door's `max_tokens`; a ComfyUI request obeys a client line.
- [ ] Run, see fail; implement; run whole suite; commit.

### Task 5: Per-door feature switches

**Files:** Modify `ratatosk/gateway_llm.py`; Test `tests/test_gateway_llm.py`.

**Interfaces:** Consumes `Route.allow` and the model's `capabilities` (Task 6; until then treat reasoning as supported). Pure function `apply_allow(body: dict, allow: dict, can_reason: bool) -> str | None` returns an error message or `None`, mutating `body`.

- [ ] Write failing tests: `test_vision_off_rejects_pictures_400`, `test_tools_off_strips_tool_fields` (`tools`, `tool_choice`, `functions`, `function_call`), `test_reasoning_off_forces_enable_thinking_false_over_client`, `test_reasoning_off_ignored_when_model_cannot_toggle`.
- [ ] Run, see fail; implement and call it in `complete` before queueing; run whole suite; commit.

### Task 6: Model capabilities from GGUF

**Files:** Create `ratatosk/gguf.py`; Modify `ratatosk/models.py` (`Model.capabilities`); Test `tests/test_gguf.py`, `tests/test_models.py`.

**Interfaces:** Produces `read_chat_template(path: Path) -> str | None` (reads only the header key/value block, never the tensors; returns `None` on any malformed file) and `Model.capabilities: dict` with `vision`, `tools`, `reasoning` booleans per the spec.

- [ ] Write failing tests with synthetic GGUF bytes built in the test: template with `tools` and `enable_thinking` gives tools and reasoning; no template gives all false; truncated file returns `None`; an `mmproj` beside the file gives vision.
- [ ] Run, see fail; implement; run whole suite; commit.

### Task 7: Model management API

**Files:** Create `ratatosk/modelstore.py`; Modify `ratatosk/web.py`, `ratatosk/settings.py` (`hf_base_url: str = "https://huggingface.co"`); Test `tests/test_modelstore.py`, `tests/test_web.py`, `tests/fakes/fake_hf.py`.

**Interfaces:** Endpoints: `GET /api/models` (id, file, size, capabilities, names, default flag); `DELETE /api/models/{id}`; `POST /api/models/find` body `{"url"}` returns the repo's GGUF files with sizes and any vision projector; `POST /api/models/download` body `{"repo","file"}` starts a background download, `GET /api/models/downloads` returns progress. Downloads write to `<name>.part` and rename when complete; they resume with a `Range` request; they refuse when free disk is under file size plus 1 GB.

- [ ] Write failing tests: list shape; delete removes file, names, and an unshared vision file, and is refused for the model the engine serves (409); find parses a repo URL variants (`/owner/repo`, `/owner/repo/blob/main/x.gguf`); download completes against `fake_hf`, resumes after a cut, and leaves no `.part` listed as a model after failure; low disk refused.
- [ ] Run, see fail; implement; run whole suite; commit.

### Task 8: Clients and door activity

**Files:** Modify `ratatosk/db.py`, `ratatosk/web.py`; Test `tests/test_db.py`, `tests/test_web.py`.

**Interfaces:** `Db.recent_clients(limit) -> list[dict]` (`client`, `agent`, `count`, `last_seen`, newest first); `GET /api/clients`; `/api/status` gains per door `today`, `last_seen`, `busy`.

- [ ] Write failing tests: grouping by address plus agent, ordering, counts; status fields.
- [ ] Run, see fail; implement; run whole suite; commit.

### Task 9: Settings page

**Files:** Modify `ratatosk/static/index.html` (Settings tab replaced by the mockup, wired to the API); Test: `tests/test_web.py` serves the page and `GET/PUT /api/settings` round-trips every new field; browser check by hand.

**Interfaces:** Consumes everything above. Unsaved-change counter and restart notes: a field map says per setting whether it applies live, restarts the engine, or restarts the service. Raw editor under Advanced. Drag order, Alt+Up/Down, suggestions, name modal, delete confirmation, copy buttons, ComfyUI note and chip as in the mockup.

- [ ] Write the round-trip test; see it fail; implement the page; run whole suite.
- [ ] Verify in a browser (desktop and phone width): add and remove a door, reorder the list, tick Drop everything, add a name, copy an address, save and reload, confirm the page shows the saved state. Report what was checked.
- [ ] Commit.

### Task 10: Public repository polish

**Files:** Create `.github/workflows/test.yml` (install `.[dev]`, run pytest on Python 3.11 and 3.13), `CONTRIBUTING.md` (short), `LICENSE` (the user's choice); Modify `README.md` (screenshots of the page, the final feature list), `docs/idun-notes.md` (retitle as "One deployment: notes").

- [ ] Add the files; run the whole suite locally; commit. Do not push; the user adds the remote.

### Task 11: Install on Idun

**Files:** Modify `docs/idun-notes.md`, `docs/rollback.md`.

- [ ] Back up `data/settings.json`; copy the app to `/home/master/ratatosk/app`; restart the service; confirm migration (doors keep their old reply lengths; `request_defaults` unchanged).
- [ ] Run `scripts/capability_check.py` and `scripts/smoke_idun.sh`; send one request from SillyTavern; check the Now tab and the new Settings page.
- [ ] Record the result and rollback in the docs; commit.
