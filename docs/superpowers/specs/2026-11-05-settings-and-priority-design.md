# Settings page, priority list and model management (addendum to the Ratatosk design)

Extends `2026-11-03-ratatosk-design.md`. The visual design is `docs/mockups/settings.html` (approved by the user on 2026-11-05); where this text and the mockup differ, the mockup is the intent and this text is what the code must do.

## Goals

Ratatosk is a general app for anyone with one GPU, not a private tool. Nothing in the UI or settings may be specific to one program, camera or machine. What a model can do is read from the model and shown, not configured.

## Doors

A door is a port plus a purpose: general text, roleplay, or ComfyUI images. Each door row shows status, address (with copy) and activity (requests today, time of the last one, a pulse while working). Opening a row shows:

- **Port** and **purpose** (editable; a port change restarts the service).
- **Longest reply** (text doors only): replaces `max_tokens_cap` and `max_tokens_cap_plain`; stored on the door as `max_tokens`. Old settings files migrate: a door without `max_tokens` takes the old cap for its purpose.
- **Allowed on this door** (text doors only): Vision, Reasoning, Tool use, each on by default.
  - Vision off: a request carrying pictures gets 400 "pictures are turned off on this door".
  - Tool use off: `tools`, `tool_choice`, `functions`, `function_call` are removed from the request.
  - Reasoning off: `chat_template_kwargs.enable_thinking = false` is forced, overriding the client. Only offered when the model's chat template has that switch; otherwise the tick is greyed with a hint.
- ComfyUI doors show only port and purpose, plus the red note: "Apps expect ComfyUI on 8188. The real ComfyUI must use another port (now 8189, under Advanced)." The collapsed row shows an amber "not the usual 8188" chip when the port differs.

`request_defaults` stays (Advanced, raw editor) for fields no toggle covers.

## Priority list

An ordered list of lines; the top goes first. Each line matches either a **door** (by port) or a **client** (source address, optionally with the program name from `User-Agent`). Below all lines sits the fixed "Everything else", served in arrival order. The first matching line wins; a client line beats a door line only by being earlier in the list.

A line may be flagged **Drop everything**. Those lines sit above all others (red) and, when a request matching one arrives, the running job is cancelled at once and that request runs next. The cancelled request fails with 503 "dropped by <client>" and is logged with status `dropped` and the name of the line that dropped it. A text reply already streaming is cut off; there is no retry. Lines apply to text and to ComfyUI image jobs alike. A running image job is cancelled by `POST /interrupt` to ComfyUI.

The default list reproduces today's behaviour: roleplay door, general text door, ComfyUI door. Tuning jobs stay last and preemptible.

Adding a line: the box suggests recent clients (address plus program, count and age, newest first, from the request log) and doors not yet listed; typing filters; an address typed by hand and Enter adds a client line. Clicking the box reopens the suggestions at any time.

Settings shape: `priority: [{"door": 11435} | {"client": "192.168.1.12", "agent": "node-fetch"}, ...]` each with optional `"drop": true` and `"label"`.

## Models

The list shows every model found in the model folders: name, size, capability pills, names it answers to, a default radio, and a delete button.

- **Capabilities** are read from the GGUF file's metadata. Vision: a vision projector (`mmproj*.gguf`) sits beside the model or is named in `mmproj_files`. Tool use: the chat template mentions tools. Reasoning: the chat template mentions `enable_thinking` or a `<think>` block. A model whose file cannot be read shows "text only" and a hint.
- **Names** are the existing `aliases`. Each has a remove ×. "+ name" offers suggestions built from that model (file name, family name, publisher-style name; names in use are left out) and a box for a custom one; a name already used by another model is refused.
- **Delete** removes the model file (and its vision file when no other model uses it) after a confirmation, drops its names, and is refused while the engine is serving that model.
- **Add a model**: paste a Hugging Face link, Ratatosk lists the repo's GGUF files, the user picks one, and it downloads into the first model folder with progress, resuming a partial file and refusing when the disk lacks room. A vision projector in the same repo is offered alongside.
- **Model folders** as before.

## Not in this round

- Searching Hugging Face by name (a link is enough for now).
- Starting a clean engine install from llama.cpp releases; `docs/idun-notes.md` describes the layout `engine_dir` expects.
- Embeddings and reranking, `/metrics`.
- Skills, reply checking and shaving (still the later sub-projects).

## Settings page behaviour

Sections: Doors, Priority, Models, GPU and memory, History, Advanced. A bar at the bottom counts unsaved changes and says which restart (engine or service); Save writes the file, Discard reverts. Changes that need no restart take effect at once. The raw settings editor stays under Advanced. The "reads pictures on GPU or CPU" setting is removed; Ratatosk keeps its current behaviour (`mmproj_on_gpu`, still in the file).
