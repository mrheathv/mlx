# MLX Studio

A native macOS SwiftUI app for local, on-device image generation with
switchable FLUX.1 / Stable Diffusion 3.5 models, running entirely through
MLX on Apple Silicon — no cloud calls beyond the one-time model download.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full design rationale
(why a Python/MLX subprocess instead of `mlx-swift` bindings, which
checkpoints were chosen and why, memory/thermal handling, and explicitly
flagged risks). Read that before making changes here — several choices
(process-over-socket IPC, one-model-resident-at-a-time, hand-synced model
registries) are load-bearing decisions, not oversights.

## ⚠️ Build status

This repository was developed in a Linux sandbox with **no Swift
toolchain, no Xcode, and no Apple Silicon/Metal**. Nothing here has been
compiled or run. Swift code is written to compile by inspection against
documented SwiftUI/Foundation APIs; the Python backend's protocol,
registry, and dispatch logic *has* been exercised in this sandbox with
`mflux`/`diffusionkit` mocked out (see `backend/tests/`), since those
engines themselves require Apple Silicon. **Before relying on this as
working software, build and run it on the target Mac** (a MacBook Air, M5,
24GB, per the original brief) and treat the first pass as a debugging
session, not a rubber-stamp.

## Repository layout

```
ARCHITECTURE.md              design doc — read this first
Package.swift                 SwiftUI app, as a Swift Package
Sources/MLXStudioApp/         app source (Models/Services/ViewModels/Views)
Tests/MLXStudioAppTests/      XCTest suite (unverified, see above)
backend/                      bundled Python/MLX inference backend
  mlxstudio_backend/          stdio JSON-lines server, model registry, engines
  tests/                      pytest suite (runs today, no Apple Silicon needed)
  pyproject.toml
```

## Running the Swift app

Requires macOS 14+ and Xcode 15+ (or the `swift` CLI from a matching
toolchain).

```bash
# Option 1: Xcode
open Package.swift   # Xcode opens it as a project directly

# Option 2: command line
swift build
swift run MLXStudioApp
```

The app expects the Python backend (below) to be set up first — it shells
out to `python3 -m mlxstudio_backend` in the `backend/` directory
(`BackendProcessManager.Configuration.developmentDefault()` in
`Sources/MLXStudioApp/Services/BackendProcessManager.swift`) — this
development-mode path assumes the repo checkout layout and a `python3` on
`PATH` with the backend's dependencies installed; packaging a bundled
interpreter for distribution is future work (ARCHITECTURE.md §1).

Run the Swift test suite:

```bash
swift test
```

## Setting up the Python backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This pulls in `mlx`, `mflux`, and `diffusionkit` (pinned versions — see
`pyproject.toml`'s comment on why), plus `huggingface_hub` and `psutil`.
`mlx`/`mflux`/`diffusionkit` require Apple Silicon; the rest of the
backend (protocol parsing, the model registry, dispatch, cancellation) can
be tested without them:

```bash
python3 -m pytest -q backend/tests
```

You can also run the backend standalone to sanity-check the protocol
before wiring up the Swift side — it's just line-based JSON on stdio:

```bash
cd backend
python3 -m mlxstudio_backend
# then paste a line like:
{"type":"list_models","id":"11111111-1111-1111-1111-111111111111"}
```

## Models

Three switchable models, all 4-bit quantized, one resident in memory at a
time. Exact checkpoints, memory footprints, and why they were chosen over
alternatives are in **ARCHITECTURE.md §2** — summary:

| Model | Role | Engine | ~Download | ~Runtime RAM |
|---|---|---|---|---|
| FLUX.1 [schnell] | fast iteration | `mflux` | 7 GB | 11 GB |
| FLUX.1 [dev] | final quality | `mflux` | 7 GB | 11.5 GB |
| Stable Diffusion 3.5 Large | alternate architecture | `diffusionkit` | 5 GB | 9 GB |

The registry is data (`backend/mlxstudio_backend/models.py`, mirrored in
`Sources/MLXStudioApp/Resources/models.json`), not code — swapping in
different checkpoints later is a config edit.

Models download on first use, through the app's UI, to
`~/Library/Application Support/MLXStudio/models/`. Generated images and
their metadata live in the same app-support directory, under
`Generations/` and `history.json`.

## Out of scope for v1

Image-to-image, inpainting, outpainting, ControlNet/LoRA, upscaling,
iCloud sync, and any cloud/network feature beyond the model download —
see ARCHITECTURE.md §7 for the full future-work list.
