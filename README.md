# MLX Studio

A native macOS SwiftUI app for local, on-device image generation on Apple
Silicon, with three switchable models — FLUX.2 [klein] 4B, FLUX.2 [klein] 9B
and Z-Image Turbo — all running through MLX on the GPU (using the M5's GPU
Neural Accelerators on macOS 26.2+). No cloud calls beyond downloading
Python packages and model weights.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the design: why a Python/MLX
child process instead of `mlx-swift`, how the checkpoints were chosen and
verified, why not Core ML / the Neural Engine, and what's still
unverified.

## Build a DMG and install it

On an Apple Silicon Mac with Xcode or the Command Line Tools
(`xcode-select --install`):

```bash
git clone --branch claude/macos-diffusion-flux-app-3d9dac https://github.com/mrheathv/mlx.git
cd mlx
./scripts/build-dmg.sh
open dist/MLXStudio-0.1.0.dmg
```

Drag **MLX Studio** to Applications and launch it. You don't need Python,
Homebrew or a virtualenv:

1. **First launch** shows a one-time setup screen. Click **Set Up**; the app
   downloads its own Python 3.11 and the MLX libraries (roughly 1 GB) into
   `~/Library/Application Support/MLXStudio/`.
2. **Pick a model** and click **Download** the first time you use it
   (FLUX.2 [klein] 4B ~4.8 GB, Z-Image Turbo ~5.9 GB, FLUX.2 [klein] 9B ~10 GB).
   klein 4B is the smallest and fastest — a good first test.
3. **Generate.**

If something fails, **Help → Show Logs in Finder** opens
`~/Library/Logs/MLXStudio/` — `setup.log` and `backend.log` contain the
details (Python tracebacks included). The first lines of `backend.log` after
each launch say whether the M5 GPU Neural Accelerators are in use, and every
generation logs its seconds per step.

The DMG is ad-hoc signed, which is fine on the Mac that built it. If you
copy it to another Mac, open the app there with right-click → **Open** the
first time.

## Develop

```bash
swift run MLXStudioApp        # or: open Package.swift in Xcode
swift test
```

When not running from a `.app`, the app runs the backend from the repo's
`backend/` directory, using `backend/.venv` if it exists. To create it you
need Python 3.11+ (macOS's built-in `python3` is 3.9 — too old); the
easiest way is `uv`:

```bash
cd backend
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Backend tests don't need Apple Silicon (mflux is mocked):

```bash
cd backend
uv run --no-project --python 3.11 --with pytest --with psutil --with huggingface_hub python -m pytest -q
```

## Repository layout

```
ARCHITECTURE.md              design doc — read this first
Package.swift                 SwiftUI app, as a Swift Package
Sources/MLXStudioApp/         app source (Models/Services/ViewModels/Views)
Tests/MLXStudioAppTests/      XCTest suite
backend/                      Python/MLX inference backend
  mlxstudio_backend/          stdio JSON-lines server, model registry, mflux engine
  tests/                      pytest suite
  pyproject.toml              pinned dependencies (the app installs from this)
scripts/build-dmg.sh          builds dist/MLX Studio.app and the DMG
```

## Models

| Model | Role | Download | Est. runtime RAM |
|---|---|---|---|
| FLUX.2 [klein] 4B, 4-bit | fast iteration (4 steps) | ~4.8 GB | ~8 GB |
| FLUX.2 [klein] 9B, 4-bit | final quality (4 steps; non-commercial license) | ~10 GB | ~13 GB |
| Z-Image Turbo, 4-bit | alternate architecture (9 steps) | ~5.9 GB | ~9 GB |

One model is resident at a time. Models live in
`~/Library/Application Support/MLXStudio/models/`; generated images and
their metadata in `Generations/` and `history.json` next to them.

## Out of scope for v1

Image-to-image, inpainting, outpainting, ControlNet/LoRA, upscaling,
iCloud sync — see ARCHITECTURE.md §7.
