# MLX Studio

A native macOS SwiftUI app for local, on-device image generation on Apple
Silicon, with three switchable models — FLUX.1 [schnell], FLUX.1 [dev] and
Z-Image Turbo — all running through MLX. No cloud calls beyond downloading
Python packages and model weights.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the design: why a Python/MLX
child process instead of `mlx-swift`, how the checkpoints were chosen and
verified, why Z-Image Turbo replaced Stable Diffusion 3.5, and what's still
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
   (FLUX.1 models ~9.5 GB each, Z-Image Turbo ~5.9 GB). Z-Image Turbo is the
   smallest and a good first test.
3. **Generate.**

If something fails, **Help → Show Logs in Finder** opens
`~/Library/Logs/MLXStudio/` — `setup.log` and `backend.log` contain the
details (Python tracebacks included).

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
| FLUX.1 [schnell], 4-bit | fast iteration | ~9.5 GB | ~13 GB |
| FLUX.1 [dev], 4-bit | final quality | ~9.5 GB | ~13.5 GB |
| Z-Image Turbo, 4-bit | alternate architecture | ~5.9 GB | ~9 GB |

One model is resident at a time. Models live in
`~/Library/Application Support/MLXStudio/models/`; generated images and
their metadata in `Generations/` and `history.json` next to them.

## Out of scope for v1

Image-to-image, inpainting, outpainting, ControlNet/LoRA, upscaling,
iCloud sync — see ARCHITECTURE.md §7.
