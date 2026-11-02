# MLX Studio — Architecture

Native macOS SwiftUI app for local text-to-image generation on Apple
Silicon using MLX, with three switchable models (FLUX.2 [klein] 4B,
FLUX.2 [klein] 9B, Z-Image Turbo), generation history, and on-demand model
downloads. No cloud inference, ever.

Target hardware for dev/test: MacBook Air, M5, 24GB unified memory, fanless.
Every design choice below is made against that ceiling, not against a Mac
Studio with 128GB and active cooling.

## 1. Inference backend: Python/MLX child process, not mlx-swift bindings

**Decision: the Swift app runs a Python child process that does all
inference through MLX, and talks to it with newline-delimited JSON on
stdin/stdout.** The Swift app never does inference math itself.

### Why not `mlx-swift` bindings directly

`mlx-swift` (the array library) is solid, but the *model implementations*
we need — FLUX.2's transformer, Z-Image's single-stream DiT, their Qwen3
text encoders and VAEs — don't exist in maintained Swift form.

| Project | Language | Status for our models |
|---|---|---|
| `ml-explore/mlx-swift-examples` | Swift | Has a Stable Diffusion example (SD1.x/SDXL-class UNet). No FLUX, no Z-Image. |
| `argmaxinc/DiffusionKit` | Python + Swift | Archived March 2026. Python side pins `mlx==0.17.3`. |
| `mflux-community/mflux` | Python | Actively maintained; 15+ model families incl. FLUX.2, FLUX 3, Z-Image, Krea 2, Qwen-Image 2.1. **Used.** |
| `flux-2-swift-mlx`, `mlx-flux2`, `flux.swift` | Swift | Individual projects, partial coverage, thin testing. |

Porting diffusion architectures plus their text encoders and VAEs to Swift,
and keeping up with upstream fixes, is months of work with a nasty failure
mode: subtle mistakes (attention masking, VAE scaling) degrade images
instead of crashing. Depending on `mflux` reuses an implementation that's
already correct and maintained.

### Why not Core ML / the Apple Neural Engine

The brief ruled out Core ML conversion pipelines, and the hardware agrees:

- On the M5, the hardware that accelerates these models is the **Neural
  Accelerators built into each GPU core**, not the separate Neural Engine
  (ANE). MLX uses them from v0.30 on: its Metal backend has dedicated
  "NAX" kernels for matrix multiplies, attention, and quantized matmuls
  with scales and biases — exactly what our 4-bit models run. Verified in
  MLX's source (`is_nax_available()` in `mlx/backend/metal/device.cpp`):
  they're used when macOS is 26.2+ and the GPU architecture generation is
  17+ (M5). Only MLX's macOS 26 build (`mlx-metal` wheel tagged
  `macosx_26_0`) contains them; `uv` installs that build on macOS 26.
  Apple's machine-learning research blog reports FLUX running over 3.8×
  faster on M5 than M4 with this path.
- The ANE is reached only through Core ML, which needs each model converted
  ahead of time, works best on fixed shapes and convolution-heavy networks
  (Apple's own Core ML Stable Diffusion port targets the SD1.x/SDXL UNet),
  and falls back to GPU/CPU for unsupported ops. Multi-billion-parameter
  diffusion transformers with large attention and LLM text encoders are a
  poor fit, and every new model would need its own conversion pipeline.

So the app doesn't target the ANE, but it does use the M5's neural
hardware — provided MLX ≥ 0.30 and macOS ≥ 26.2. Because that can't be
verified from a Linux build environment, the backend logs it on every
start (`diagnostics.py`, mirroring MLX's rule), e.g.
`M5 Neural Accelerators: ACTIVE`, or which condition failed. It also logs
seconds-per-step for every generation.

### Why a pipe, not a local socket

- No port allocation, no socket file lifecycle, nothing to clean up on crash.
- Only the parent process can reach the backend — there is no listening
  address at all.
- Message rate is low (a few progress ticks per second) and payloads are
  small: images are written to disk by Python and referenced by path.

Trade-offs accepted:

- The app needs a Python 3.11 environment with `mlx` and `mflux`. The
  packaged app provisions it itself on first launch (§5).
- One more process that can crash independently of the UI.
  `BackendProcessManager` routes events per request; if the process
  exits, every in-flight request fails with `backend_exited` (including
  the last line the process wrote to stderr) instead of hanging, and the
  next request starts a fresh process.
- The process is kept warm across generations so a loaded model stays
  resident instead of being reloaded from disk per image.
- Python's stdout is the protocol channel, so the backend points
  `sys.stdout` at stderr at startup. Stray `print()`s from mflux land in
  the log, not in the protocol stream.

## 2. Models and checkpoints

Only one model is resident at a time; switching unloads the current one
first (the UI shows "Loading model…", distinct from "Generating…").

| Role | Model | Released | Checkpoint (4-bit, mflux format) | Download | Est. runtime RAM | Steps |
|---|---|---|---|---|---|---|
| Fast | FLUX.2 [klein] 4B | Jan 2026 | `Runpod/FLUX.2-klein-4B-mflux-4bit` | ~4.8 GB | ~8 GB | 4 |
| Quality | FLUX.2 [klein] 9B | Jan 2026 | `mlx-community/flux2-klein-9b-4bit` | ~10 GB | ~13 GB | 4 |
| Alternate architecture | Z-Image Turbo | Nov 2025 | `filipstrand/Z-Image-Turbo-mflux-4bit` (by mflux's author) | ~5.9 GB | ~9 GB | 9 |

Download and RAM figures are estimates from parameter counts and mflux's
published measurements, used by the memory warning (§4); they need
confirming on the M5. All three are guidance-distilled, so the UI hides
the guidance slider and negative prompt (FLUX.2 klein's `generate_image`
doesn't take a negative prompt at all).

Licensing: Black Forest Labs releases klein 4B under Apache 2.0 and klein
9B under the FLUX Non-Commercial License. Check the Z-Image Turbo model
card for its terms before any commercial use.

### Why this lineup

mflux's own model table calls FLUX.1 (Aug 2024) a "legacy option". For a
fanless 24GB machine the deciding factors are memory and step count (heat
per image): klein runs in 4 steps at under ~8 GB (4B) / ~13 GB (9B), versus
FLUX.1 [dev]'s 12B parameters and 25 steps. Considered and not chosen as
defaults:

- **FLUX 3 Image Klein** (Oct 2026, newest from BFL; strongest text
  rendering): 16 steps, ~18 GB peak measured at 4-bit before mflux's
  lower-memory VAE decode — worth adding once measured on the M5.
- **Krea 2 Turbo** (Jun 2026, 12B, 8 steps): good style range; bigger.
- **Qwen-Image 2.1**: 27 GB peak even at 8-bit — too big for 24 GB.

Stable Diffusion 3.5 (in the original brief) was dropped earlier: its only
MLX implementation, DiffusionKit, pins `mlx==0.17.3`, which can't share an
environment with mflux (and would forgo the M5 Neural Accelerators).

### How the checkpoints were verified

mflux loads stored weights with `module.update(weights, strict=False)`.
If a checkpoint was saved with a module layout that differs from the
installed mflux, nothing errors — the weights simply don't apply and the
model generates noise. Two defenses:

1. **Layout diff across versions.** Each model's module tree was built with
   MLX's lazy initialization (no weights allocated) in every mflux version
   that could have saved these checkpoints, quantized to 4 bits with that
   version's predicate, and every parameter name and shape diffed against
   the pinned 0.24.0: FLUX.2 klein 4B and 9B as built by 0.16.0 (the first
   version with klein) and 0.18.0, and Z-Image Turbo as built by 0.13.0 —
   all identical. So any mflux-saved checkpoint of these models loads
   correctly.
2. **Load-time check** (`checkpoint_check.py`). Hugging Face is
   unreachable from the build environment, so the specific repos' files
   couldn't be inspected. Instead, after building a model the engine reads
   the tensor names from each checkpoint file's safetensors header and
   compares them with the parameters the model expects. A mismatch raises
   a clear "would generate noise" error instead of producing garbage. This
   check was exercised against mflux 0.24's real FLUX.2 klein module tree,
   with a matching checkpoint and with one tensor renamed.

`mflux` is pinned to exactly 0.24.0 in `backend/pyproject.toml`. The engine
(`engines/mflux_engine.py`) was written against 0.24.0's source; its import
paths, per-model `generate_image` arguments, image saving and step-callback
hookup were exercised against that real code with heavy dependencies
stubbed.

The registry lives in `backend/mlxstudio_backend/models.py` (authoritative)
and is mirrored for the UI in `Sources/MLXStudioApp/Resources/models.json`;
`backend/tests/test_models_registry.py` fails if the two disagree.

## 3. Swift app structure

A Swift Package (`Package.swift`) with a macOS executable target using the
SwiftUI `App` lifecycle — diffable, no generated `.pbxproj`. Opens in Xcode
via `Package.swift`, runs with `swift run`, and `scripts/build-dmg.sh`
assembles it into a signed `.app` and DMG.

```
Sources/MLXStudioApp/
  App.swift                        @main App; gates the UI on backend setup
  Models/
    GenerationParameters.swift     steps, guidance, seed, size, batch count
    GenerationRequest.swift        prompt + params + model id
    HistoryEntry.swift             persisted record of one generated image
    ModelDescriptor.swift          registry entry; loads models.json
    BackendMessages.swift          Codable wire protocol
  Services/
    BackendEnvironment.swift       first-run Python setup via bundled uv
    BackendProcessManager.swift    runs/supervises the backend; per-request routing; LogFile
    InferenceBackendClient.swift   typed generate/download update streams
    ModelManager.swift             downloads, download-complete detection
    MemoryGuard.swift              host_statistics64-based free RAM check
    HistoryStore.swift             JSON index + image files on disk
  ViewModels/
    GenerationViewModel.swift
    GalleryViewModel.swift
  Views/
    BackendSetupView.swift         first-run setup screen
    ContentView.swift              sidebar: Generate / Gallery
    ModelPickerView.swift          segmented control + download state
    PromptFormView.swift           prompt and the parameters the model supports
    GenerationProgressView.swift   per-step progress or spinner + elapsed time
    GalleryView.swift
    GenerationDetailView.swift     metadata + "Reuse Settings"
    MemoryWarningView.swift
  Resources/
    models.json
```

### Data flow — generation

1. `PromptFormView` edits state in `GenerationViewModel`.
2. On submit, `MemoryGuard` compares the model's estimated footprint with
   free + inactive memory; if short, `MemoryWarningView` offers Cancel or
   Try Anyway.
3. `InferenceBackendClient.generate` asks `BackendProcessManager` to start
   the process if needed, register for events with the request's `id`,
   and write a `generate` command.
4. The backend loads the model if it isn't already active (emitting
   `model_loading`), then streams `progress` events from mflux's per-step
   callback and finishes with `result` (image paths + seeds) or `error`.
5. Progress drives `GenerationProgressView` — step N of M plus elapsed
   time, never an ETA (see §4).
6. On `result`, `HistoryStore` records each image with its full settings;
   the gallery reads from the store.

Cancel sends a `cancel` command; the backend's step callback raises on the
next step, which mflux propagates out of `generate_image`.

### Data flow — model download

`ModelManager` sends `download`; the backend runs `huggingface_hub`'s
`snapshot_download` into `~/Library/Application Support/MLXStudio/models/<model id>/`
and reports progress by polling bytes on disk against the repo's total
size. Only after a complete download does it write a
`.mlxstudio-download-complete` marker; both sides treat a model as
downloaded only if the marker exists, so an interrupted download shows as
not downloaded and resumes (already-fetched files are reused) when retried.

## 4. Handling the hardware constraints

- **Memory**: `MemoryGuard` reads free + inactive pages via
  `host_statistics64` and compares them with the model's estimate plus a
  4 GB reserve for macOS and the app. A shortfall shows a warning with an
  informed "Try Anyway", never a silent hang.
- **No active cooling**: the UI never promises a duration. It shows steps
  and elapsed time. Batches run sequentially, so a throttled image late in
  a batch doesn't hold back earlier ones. The lineup favors low step
  counts, and the backend log records seconds-per-step per generation so
  throttling across back-to-back images is visible.

## 5. Packaging and first run

`scripts/build-dmg.sh` (run on an Apple Silicon Mac) builds a release
binary and assembles `MLX Studio.app`:

```
MLX Studio.app/Contents/
  MacOS/MLXStudio                  the SwiftUI app
  MacOS/uv                         pinned uv release, checksum-verified
  Resources/models.json
  Resources/backend/               the Python backend's source + pyproject.toml
  Info.plist
```

It ad-hoc signs the bundle and wraps it in a DMG with an Applications
shortcut. Ad-hoc signing is enough on the Mac that built it; a copy moved
to another Mac needs right-click → Open (or Developer ID signing and
notarization, not done here).

Python is not bundled. On first launch `BackendEnvironment` shows a setup
screen and runs the bundled `uv`:

1. `uv venv --python 3.11 --managed-python` → a uv-managed CPython 3.11
   under `~/Library/Application Support/MLXStudio/python-installs`, venv at
   `…/python-env`.
2. `uv pip install -r Resources/backend/pyproject.toml` → MLX, mflux and
   their pinned dependencies (on macOS 26, the Neural-Accelerator build of
   MLX).

The environment counts as installed only while a saved copy of the
`pyproject.toml` it was built from matches the bundled one, so an app
update that changes dependencies (like this mflux upgrade) re-runs setup.
The backend runs with `PYTHONPYCACHEPREFIX` pointed at
`~/Library/Caches/MLXStudio`, so nothing writes into the signed bundle.

Logs go to `~/Library/Logs/MLXStudio/` (`setup.log`, `backend.log`;
Help → Show Logs in Finder).

Under `swift run`/Xcode the app skips all of this and runs the backend
from the repo's `backend/` directory, preferring `backend/.venv`.

## 6. Risks and unverified areas

- **Not yet generating on Apple Silicon.** Developed in a Linux container.
  Verified there: backend protocol/registry/dispatch/cancellation/download
  handling and the checkpoint and diagnostics logic (64 pytest tests,
  mflux mocked); the dependency set resolving for macOS arm64; the exact
  `uv` setup commands; checkpoint layouts and the engine's calls against
  real mflux 0.24 code. Not verified: real generation, actual Neural
  Accelerator use, memory and thermal behavior — the startup and
  per-generation log lines exist to measure those on the M5.
- **Checkpoint repos weren't inspected directly** (Hugging Face is blocked
  from the build environment); the load-time check covers this.
- **Memory estimates** in the registry are reasoned, not measured.

## 7. Out of scope (future work)

From the brief: image-to-image, inpainting, outpainting, ControlNet/LoRA,
upscaling, iCloud sync, and any network feature beyond model downloads.
FLUX.2 klein supports image editing, which makes it a natural base for
image-to-image later. Also deferred: FLUX 3 Image Klein as an option once
measured, an app icon, Developer ID signing/notarization, keeping more
than one model resident, and generating `models.json` from the Python
registry at build time (a test keeps them in sync for now).
