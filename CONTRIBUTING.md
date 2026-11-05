# Contributing

Ratatosk is small on purpose. Before a larger change, open an issue and say what you want to do.

- Tests first: every behaviour has a test that failed before the code existed. `python -m pytest` runs them all; they need no GPU and no network (see `tests/fakes/` for the stand-in llama-server, ComfyUI and Hugging Face).
- Keep the UI free of anything specific to one program, model or machine.
- The design decisions are in `docs/superpowers/specs/`, the build plans in `docs/superpowers/plans/`. Read the relevant spec before changing behaviour it describes.
- `docs/mockups/settings.html` is the approved look of the Settings page.
