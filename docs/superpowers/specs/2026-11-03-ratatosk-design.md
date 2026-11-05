# Ratatosk: GPU traffic control for Idun

Status: draft for review. Date: 2026-11-03.

Ratatosk is the squirrel that runs messages up and down the world tree. Here it carries every request between SillyTavern (and other tools) and the one GPU on Idun, decides who goes next, and checks what comes back.

## 1. Purpose

Idun has one small GPU that can do one thing at a time: write text (llama.cpp) or draw images (ComfyUI). Today the two fight over video memory, nothing queues text, a stalled request blocks everything, and the roleplay model writes replies that are slow, bloated and sometimes malformed.

Ratatosk fixes that by sitting in front of both:

1. One queue and one owner of the GPU, so text and images take turns and free memory for each other.
2. A model manager that fits models to Idun's real drives and GPU, and tunes each one properly.
3. A roleplay layer that validates, repairs and shortens replies, and shows what each reply is made of.

Success means: roleplay replies are fast, have a valid game-state block every time, are short, and you can see and tune what they contain; images and text never fight over memory.

## 2. Environment (measured, 2026-11-03)

- Idun: Ubuntu 26.04, Python 3.14.4, no Docker, no Node. LM Studio and ComfyUI run as desktop apps under user `master`.
- GPU: GTX 1070, 8 GB. CPU RAM 15 GB.
- Drives: `/` is a 120 GB SATA SSD, 91% full (about 10 GB free), holds LM Studio's models (28 GB). `/mnt/hdd` is a 500 GB spinning HDD (235 GB free) holding ComfyUI's models. The NAS (192.168.1.9) has about 3 TB free over the network. A fast 1 TB M.2 is planned.
- Ports today: LM Studio 11434, ComfyUI 8188, both on all interfaces.
- LM Studio's bundled `llama-server` runs standalone when `LD_LIBRARY_PATH` includes its backend folder and `backends/vendor/linux-llama-cuda-vendor-v1` (about 1.2 GB together). It supports `--fit`, `--api-key`, `--cache-reuse`, `/tokenize` and grammar-constrained output. There is no `llama-bench`.
- Why it is slow today: `lm-studio-tuner` benchmarked the roleplay model at a 262,144 token context. At that size only 61% of the layers fit on the GPU, so it settled on 2 threads and 5.7 tokens/s. Real prompts are 3 to 9k tokens.

## 3. Scope

Four sub-projects, each with its own plan and tests. This spec covers the whole design; each sub-project gets its own implementation plan before it is built.

1. **Core:** gateway, queue, engine manager, ComfyUI coordination, model adoption, Now and Settings tabs.
2. **Models:** Hugging Face search and download, storage tab, tuning per context size.
3. **Roleplay checks and skills:** a dedicated port that validates, repairs and retries replies, and adds selected skills to the prompt.
4. **Composition and shaving:** sentence labelling, percentage log, shortening, later sliders.

Not in scope: running more than one model at once, multi-GPU, user accounts, any cloud service, training models, replacing ComfyUI's own interface.

## 4. Architecture

One Python service (asyncio, aiohttp, SQLite, numpy), running as a systemd user service for `master`. The web page is plain HTML and JavaScript with no build step.

```
SillyTavern, other tools            ComfyCharacters, browsers
          |                                   |
   :11434 plain LLM   :11435 roleplay         :8188 ComfyUI
          \________________  Ratatosk  ______/        :5960 UI
                  |                       |
          llama-server (127.0.0.1)   ComfyUI (127.0.0.1)
```

- **Takeover.** Ratatosk listens on the public ports 11434 and 8188 so existing clients need no change. LM Studio is retired, and `llama-server` and ComfyUI bind to localhost only. ComfyUI's launch options change in Stability Matrix; this is done together with the user.
- **Routes.** A route is a listening port, a backend kind (`llm` or `comfy`) and a profile (`plain` or `rp`). The Settings tab edits the list. Defaults: 11434 llm/plain, 11435 llm/rp, 8188 comfy/plain, UI on 5960.
- **Engine.** Ratatosk starts and stops `llama-server` itself with exact flags. One model at a time, one parallel slot.
- **Rollback.** If Ratatosk misbehaves: stop the service, relaunch ComfyUI on 0.0.0.0:8188, start `llama-server` or LM Studio on 11434. The steps live in `docs/rollback.md`. The service restarts itself after a crash (`Restart=always`).
- **LM Studio is not uninstalled until Ratatosk has been verified.** First the engine and its CUDA libraries are copied to Ratatosk's own folder, tested, and only then, with the user's go, is LM Studio removed (this frees roughly 5 GB of old engine versions).

## 5. Components

Each is one file or small package with one job.

| Module | Job | Depends on |
|---|---|---|
| `gateway` | Accepts requests on each route, forwards them, streams responses back, passes ComfyUI's websocket through | `queue`, `routes` |
| `queue` | One priority queue for the whole GPU; runs one job at a time; cancels abandoned jobs | none |
| `engine` | Start, stop, health-check `llama-server`; pick flags from the tune for the needed context size | `models`, `tunes` |
| `comfy` | Read ComfyUI's queue, send "free memory", detect when image jobs are pending | none |
| `models` | List installed model files across the storage folders; read GGUF metadata; adopt LM Studio's files | none |
| `storage` | The list of model folders with speed class, free-space guard, move between folders | none |
| `hf` | Search and download from Hugging Face, with a free-space check | `storage` |
| `tuner` | Find the fastest settings per model and context size using the server's own timings | `engine` |
| `rp` | Roleplay checks, repair, retry, shaving | `engine`, `classify` |
| `classify` | Label sentences; compute composition | embeddings (section 9) |
| `db` | SQLite access | none |
| `web` | JSON API and the HTML pages | all |

## 6. Queue and GPU rules

- **Priority:** P0 roleplay requests, P1 other LLM requests, P2 image jobs, P3 tuning and benchmarks. Only P3 can be interrupted: any other request aborts a running benchmark and waits.
- **One job at a time.** A text job runs to completion before an image job starts, and vice versa.
- **Swapping.** Only when the type of the next job differs from the loaded one. Next job is an image: stop `llama-server` (frees its memory). Next job is text: call ComfyUI's `POST /free` with `unload_models` and `free_memory` true, then start the engine if it is not running. After 30 idle minutes (setting) the engine is stopped.
- **Cancellation.** If a client disconnects, its job is cancelled, including a running generation.
- **Length cap.** On `rp` and `plain` routes, `max_tokens` is capped to a setting (default 1024). SillyTavern currently asks for 65,536.
- **Context sizes.** Fixed buckets: 8k, 16k, 32k, 64k, 128k. The needed size is prompt tokens (exact, from the engine's `/tokenize` when loaded, otherwise characters / 3) plus the capped `max_tokens`, times 1.25, rounded up to a bucket. Move up when a request does not fit. Move down only when every request in the last 10 minutes would have fitted two buckets lower. A reload takes a minute or more, so the system errs towards staying loaded.
- **Status.** The Now tab shows the running job, the queue, GPU memory (`nvidia-smi`), the loaded model and context, and tokens per second.

## 7. Storage

- Model folders are a list in Settings. Each has a path, a label and a speed class, detected from `/sys/block/*/queue/rotational` and mount type: `ssd`, `hdd` or `network`. Initially: the LM Studio models folder (SSD) and `/mnt/hdd` (HDD). Adding the future M.2 is adding one entry and moving models.
- Existing LM Studio files are adopted in place (read, never copied or changed).
- **Free-space guard.** A download or move is refused if it would leave less than the reserve (default 5 GB) free on the target. The Models tab shows free space per folder at all times.
- Loading from an `hdd` or `network` folder shows an estimated wait (size / measured read speed) and offers to copy the model to a faster folder first. The tab suggests moving rarely used models to a slower folder.
- Models that cannot fit the 8 GB GPU at any usable context are flagged, not hidden.

## 8. Tuning

- Replaces `lm-studio-tuner`. Its search is ported: coordinate descent over GPU layers, KV cache type (f16, q8_0, q4_0), flash attention and CPU threads, each candidate measured with a real load and generation.
- Settings map to `llama-server` flags: layers `--n-gpu-layers`, cache `--cache-type-k/-v`, flash `--flash-attn`, threads `--threads`, context `--ctx-size`. Speed comes from the server's response `timings`, which cover prompt and generation separately.
- Results are stored per (model, context bucket). Tuning at 262k is never the default; the default bucket is 32k.
- Tuning runs only when the queue is empty and can be stopped; it is resumable. Time is not critical.
- `--fit` is the first candidate, because it sizes layers and context to free memory when nothing is pinned.

## 9. Roleplay pipeline (the `rp` profile)

Requests on the `rp` route are buffered: the reply is generated in full, checked, then returned (as a stream of quickly delivered chunks if the client asked for streaming).

**Checks** (plain code, no second model):
1. Exactly one `<UpdateVariable><JSONPatch>` block at the end, valid JSON, only `replace`, only known `/Intent/...` paths, mood values formatted `id:mood` with a known mood.
2. No thinking or draft text, and no second copy of the reply.
3. No echo of the player's last message.
4. Word count at or under the cap (setting, default 150).
5. No banned phrases (editable list).
6. Clothing words in the prose but no appearance entry in the block.

**Repair.** If only the block is wrong or missing, one short extra call asks the model for just the block, constrained by a JSON grammar so it cannot be malformed. Small fixes are applied without a call (for example a mood word missing its character id).

**Retry.** If the prose breaks a rule, regenerate up to 2 times with a one-line hint appended. After that the best attempt is delivered and flagged in the log.

**Shaving.** After checks the reply is split into sentences and each is labelled (below). Sentences are dropped whole, lowest value first, until the reply is within the word cap and each category is within its budget. Default order of what goes first: philosophy and reflection, atmosphere, feelings, description, exposition; dialogue, action and plot beats are kept. Rules: never edit inside quotation marks, never touch the update block, keep at least the first and last sentence of the reply, keep a sentence that a quote depends on. Modes: off, light (default), strong. The original is stored; the shortened text is what is returned, so it also becomes the model's example for the next turn.

**Labels.** Dialogue, action, physical description, setting and atmosphere, feelings and inner thoughts, reflection and philosophy, exposition and lore, plot beat. Rules handle dialogue (text in quotes) and most action. For the rest, each sentence is embedded with a small CPU embedding model served by a second `llama-server` (`--embeddings`, no GPU; Idun already has `nomic-embed-text`, so no PyTorch is needed) and classified by nearest labelled examples (numpy). Low confidence gives `uncertain`. The user can relabel any sentence in the log; relabelled sentences become new examples. A starter set of about 100 sentences is drafted by a model and corrected by the user.

**Steering through sliders** is built on skills (next section): a slider fills in a skill template, for example "about {n} words, mostly dialogue and action".

### Skills (prompt injection)

A skill is a block of instruction text that Ratatosk adds to the prompt before relaying it. The roleplay model obeys what sits at the very end of the prompt and half-ignores the rest, so position is part of a skill.

- **Format.** A folder `skills/<name>/SKILL.md` (the layout Claude skills use): a header with `name`, `description`, optional `position` (`start` or `end`, default `end`) and optional `trigger` (keywords; empty means always on when selected), then the text. Only the text file is ever read; scripts and other files are ignored, and nothing is executed.
- **Selection.** Each route has a list of enabled skill names, edited as checkboxes in Settings. Default: none.
- **Injection.** For a text prompt in ChatML form, `end` skills become one extra `<|im_start|>system` block placed immediately before the final `<|im_start|>assistant`, and `start` skills a block at the very beginning. For a chat-style request (`messages`), the same blocks are added as system messages first and last. If the prompt has no recognisable markers, nothing is injected and the log page says so. `start` is for stable text only, because a change at the beginning discards the saved prompt work after it.
- **Never stored.** Injected text is not part of the client's chat history, so it cannot pile up or be copied by the model into later replies. The log page shows the injected text and the final prompt for every reply.
- **Budget.** Each skill shows its size in tokens; the total is checked against a setting (default 600 tokens) and over-budget selections show a warning. Each rule should have one owner: the game's card keeps its game logic, skills carry style and format.
- **Add from GitHub.** The user pastes a repository address; Ratatosk lists every `SKILL.md` found (through GitHub's public API, only when the user asks), shows previews and sizes, and copies the chosen ones in. Imported skills start disabled, record their source and commit, can be updated or removed, and are flagged when large. This is the only network access the feature has, and it is user-initiated.

## 10. Interface

Four tabs, served on port 5960.

- **Now:** live job, queue (text and images together), GPU memory, tokens per second, last few requests.
- **Models:** installed models by folder with size and speed class, free space, Hugging Face search with fit-to-GPU hints, download, tune per context size, move.
- **Roleplay:** one card per reply. A coloured stacked bar shows the composition in percent. Hovering a segment highlights those sentences; a setting switches highlighting and colouring. Shows checks passed or failed, retries, tokens before and after shaving, and the original with dropped sentences marked. Click a sentence to relabel it.
- **Settings:** routes (listening port to backend), paths, reserve, caps, thresholds, banned phrases, shave mode. Under each roleplay route, a skill list: a checkbox, name, description, position and token size per row, a preview and editor, and an "Add from GitHub" button.

## 11. Data

SQLite file `data/ratatosk.db`: `requests` (time, route, profile, model, context, prompt and generated tokens, seconds, speed, status, attempts), `replies` (request id, original, delivered, checks, composition, sentence labels), `models` (path, folder, size, quant, architecture, last used), `tunes` (model, bucket, settings, speed, time to first token), `labels` (sentence, label, source). Settings in `data/settings.json`.

## 12. Testing

- Fakes: a small fake `llama-server` and a fake ComfyUI (aiohttp apps) let the queue, swapping, context buckets, caps and cancellation be tested on the development PC without a GPU.
- Unit tests: free-space guard, bucket choice, each roleplay check, repair, shaving rules (never edits quotes, never touches the block), label rules.
- Smoke tests on Idun (scripts, run by hand): a real chat through port 11435, an image job, a swap in each direction with `nvidia-smi` before and after.

## 13. Build order and acceptance

1. **Core.** SillyTavern chats work through 11434. ComfyCharacters images work through 8188. Swapping in both directions shows the memory freed in `nvidia-smi`. A request needing a larger context reloads once. Tests pass.
2. **Models.** A model can be searched, downloaded (refused when it does not fit), tuned at 32k, and the tune is used on the next load. The tuned roleplay model reaches clearly more than the old 5.7 tokens/s.
3. **Roleplay checks.** Twenty scripted turns through port 11435 give a valid block every time, no leaked thinking, no echoes.
4. **Composition and shaving.** Labelling agrees with the user's corrections on at least 80% of a held-out set of 100 sentences. Shaved replies are at most half the original tokens, and a manual review of 20 shows the plot beats kept.

## 14. Risks and open questions

- Python 3.14 is new. The wheels for aiohttp and numpy must install on Idun; check this first in Phase 1. Fallback: install a supported Python alongside.
- Qwen 3.5 uses a newer design where reusing saved prompt state is limited, and `--fit` has had a reported bug with a similar architecture. Tuning must verify both rather than assume.
- Buffering delays the first word of roleplay replies. Acceptable, as replies already take minutes, but worth watching.
- The takeover makes Ratatosk a single point of failure for text and images. Mitigated by auto-restart and a documented rollback.
- ComfyUI's browser interface uses a websocket that the gateway must pass through unchanged. Verify early.
- Skills written for coding agents are long and assume tools, so most will not suit a 9B roleplay model; the size warning and the disabled-on-import default exist for that reason. Skills from GitHub are third-party text in the prompt: the model has no tools, so the risk is output quality, but they are previewed before use.
- Open: a separate Images tab (not now, easy to add); an API key on the routes (off by default, LAN only).

## 15. Changes made while building (2026-11-05)

Real use on Idun showed things the design did not cover. Each change below was measured, not guessed.

- **Vision.** A model has an optional vision file (`mmproj`), found beside it or set under `mmproj_files`; the engine gets `--mmproj` and `--image-max-tokens`. A picture sent to a model without one gets a clear 400. The roleplay model reads pictures with the camera model's vision file, so one loaded model serves both (an alias maps the camera's model name to it), with no swapping.
- **Where the vision projector runs.** On the 8 GB GPU the projector halves text speed (25.1 to 11.8 tok/s) but reads pictures about 4 times faster than on the CPU (49 s against 183 s for 12 real-sized frames). `mmproj_on_gpu` chooses; Idun uses the GPU because its camera system sends a request every few minutes.
- **Context sizing.** The rule in section 6 is now `1.25 * (prompt + pictures * image_max_tokens + min(capped reply, reply_reserve))`, with `image_max_tokens` 512 (a real camera frame measured about 430 to 500) and `reply_reserve` 2048. Reserving the whole cap and 1024 per picture had forced a 32k context, which cut text speed from 25 to about 14 tok/s; 16k is enough for a 12-frame camera request (about 6 to 7k tokens).
- **Reply caps per route.** `max_tokens_cap` (roleplay route, 1024) and `max_tokens_cap_plain` (other routes, 8192): a thinking model spends over a thousand tokens before it answers, and the first replay of the camera request used all 1024 thinking.
- **Request defaults.** `request_defaults` maps a model name to request fields added when the client left them out (client values win, nested objects merge). It plays the role of LM Studio's per-model presets, for example turning thinking off for one client name.
- **ComfyUI memory.** `free()` waits for ComfyUI's own held memory (`torch_vram_total`) to drop, not for a quiet reading; the first live test had started the engine before ComfyUI let go (3.7 against 25 tok/s).
- **Request log.** Requests keep client, program, endpoint, token cap asked and allowed, waiting time, speed, errors, and for the newest 50 the prompt and reply text. Pictures are stored as `[image]`. The GPU card names the programs using GPU memory.
- **Capability check.** `scripts/capability_check.py` tests text, chat, streaming, thinking, pictures, structured JSON output and tool calls against any model name; it passes for both names on Idun.

| Measured on Idun | Result |
|---|---|
| Old LM Studio setup (262k context, tuned by lm-studio-tuner) | 5.7 tok/s |
| Ratatosk, model fully on the GPU, 8k or 16k context | 25.1 tok/s |
| Same with the vision projector on the GPU | 11.8 to 12.9 tok/s |
| Same at 32k context | about 14 tok/s (projector on CPU) |
| 12 real-sized frames, projector on the GPU / on the CPU | 49 s / 183 s |

## Appendix: evidence for shaving

A real reply from the game was 374 tokens (1,707 characters). Almost all of it was restated feelings, run-on atmosphere and filler; the actual events (Sera covers herself, answers hesitantly, is embarrassed) fit in two or three sentences. The same reply also exposed an engine log line, `mood ignored (unknown cast id): nervous`, where the model wrote a mood word without a character id and an unknown mood. Check 1 and the repair step exist to catch exactly this.
