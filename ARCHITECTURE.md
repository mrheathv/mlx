# MLX Studio — Architecture

Native macOS SwiftUI app for local text-to-image generation on Apple
Silicon using MLX, with three switchable models (FLUX.1 schnell, FLUX.1
dev, Z-Image Turbo), generation history, and on-demand model downloads.
No cloud inference, ever.

Target hardware for dev/test: MacBook Air, M5, 24GB unified memory, fanless.
Every design choice below is made against that ceiling, not against a Mac
Studio with 128GB and active cooling.

## 1. Inference backend: Python/MLX child process, not mlx-swift bindings

**Decision: the Swift app runs a Python child process that does all
inference through MLX, and talks to it with newline-delimited JSON on
stdin/stdout.** The Swift app never does inference math itself.

### Why not `mlx-swift` bindings directly

`mlx-swift` (the array library) is solid, but the *model implementations*
we need — FLUX.1's rectified-flow transformer, Z-Image's single-stream DiT,
their T5/CLIP/Qwen3 text encoders and VAEs — don't exist in maintained
Swift form.

| Project | Language | Status for our models |
|---|---|---|
| `ml-explore/mlx-swift-examples` | Swift | Has a Stable Diffusion example (SD1.x/SDXL-class UNet). No FLUX, no Z-Image. |
| `argmaxinc/DiffusionKit` | Python + Swift | Archived March 2026. Swift MLX side never finished. Python side pins `mlx==0.17.3` (see §2). |
| `filipstrand/mflux` | Python | Actively maintained. FLUX.1, FLUX.2, Z-Image, Qwen-Image and more, with quantization. **Used.** |
| `flux-2-swift-mlx`, `mlx-flux2`, `flux.swift` | Swift | Individual projects, partial coverage, thin testing. |

Porting two diffusion architectures plus their text encoders and VAEs to
Swift, and keeping up with upstream fixes, is months of work with a nasty
failure mode: subtle mistakes (attention masking, VAE scaling) degrade
images instead of crashing. Depending on `mflux` reuses an implementation
that's already correct and maintained, and keeps the Swift layer what it
should be: a thin native UI and persistence layer.

### Why a pipe, not a local socket

- No port allocation, no socket file lifecycle, nothing to clean up on crash.
- Only the parent process can reach the backend — there is no listening
  address at all.
- Message rate is low (a few progress ticks per second) and payloads are
  small: images are written to disk by Python and referenced by path,
  never sent through the pipe.

Trade-offs accepted:

- The app needs a Python 3.11 environment with `mlx` and `mflux`. The
  packaged app provisions it itself on first launch (§5); about 1 GB.
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

| Role | Model | Checkpoint | On disk | Est. runtime RAM | Defaults |
|---|---|---|---|---|---|
| Fast | FLUX.1 [schnell], 4-bit | `dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit` | ~9.5 GB | ~13 GB | 4 steps, no CFG |
| Quality | FLUX.1 [dev], 4-bit | `dhairyashil/FLUX.1-dev-mflux-4bit` | ~9.5 GB | ~13.5 GB | 25 steps, guidance 3.5 |
| Alternate architecture | Z-Image Turbo, 4-bit | `filipstrand/Z-Image-Turbo-mflux-4bit` (by mflux's author) | ~5.9 GB | ~9 GB | 9 steps, no CFG |

Runtime RAM figures are estimates used by the memory warning (§4), not
measurements; they need confirming on the M5.

### Why Z-Image Turbo instead of Stable Diffusion 3.5

The original brief asked for SD3.5 as the third model. Its only MLX
implementation is DiffusionKit, which hard-pins `mlx==0.17.3`; `mflux`
0.18 needs `mlx` 0.29.3. They can't share one Python environment (found
by resolving both for macOS arm64 with `uv pip compile`). Running SD3.5
in a second environment would double setup size and add a second IPC
hop, and whether mlx 0.17 works on an M5 at all is unknown. Z-Image Turbo
is a different architecture from FLUX (a 6B single-stream DiT with a
Qwen3 text encoder), runs on the same `mflux`, and is smaller.

### How the checkpoints were verified

mflux loads stored weights with `module.update(weights, strict=False)`.
If a checkpoint was saved by an mflux version whose module names differ
from the installed one, nothing errors — the weights simply don't apply
and the model generates noise. So each checkpoint was checked against
the pinned mflux 0.18.0 by building both versions' model trees with MLX's
lazy initialization (no weights allocated), applying 4-bit quantization
with each version's own predicate, and diffing every parameter name and
shape:

- FLUX.1 schnell/dev saved by mflux **0.6.2** vs 0.18.0: every parameter
  0.18 needs is present with an identical shape (including quantized
  `scales`/`biases`); one extra key (`norm_out.linear.bias`) is ignored.
  Both `dhairyashil` repos state they were made with 0.6.2. The
  `madroid` repos considered earlier don't say which version saved them,
  so they were dropped.
- Z-Image Turbo saved by mflux **≥0.13** vs 0.18.0: identical trees.

`mflux` is pinned to exactly 0.18.0 in `backend/pyproject.toml`; changing
it means redoing this check. The engine (`engines/mflux_engine.py`) was
written against 0.18.0's source rather than docs, and its import paths,
`generate_image` keyword arguments and callback hookup were exercised
against that real code with heavy dependencies stubbed.

Also pinned: `pillow<13`. Pillow 13.0.0 does `from numbers import
_IntegralLike`, which fails on Python 3.11–3.13 and breaks `import mflux`.

The registry lives in `backend/mlxstudio_backend/models.py` (authoritative)
and is mirrored for the UI in `Sources/MLXStudioApp/Resources/models.json`.
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
    PromptFormView.swift           prompt, negative prompt, parameters
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
  informed "Try Anyway", never a silent hang. How MLX/Metal behaves at the
  edge on a 24GB M5 still needs observing on the hardware.
- **No active cooling**: the UI never promises a duration. It shows steps
  and elapsed time. Batches run sequentially, so a throttled image late in
  a batch doesn't hold back earlier ones.

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
   their pinned dependencies.

The environment counts as installed only while a saved copy of the
`pyproject.toml` it was built from matches the bundled one, so an app
update that changes dependencies re-runs setup. The backend runs with
`PYTHONPYCACHEPREFIX` pointed at `~/Library/Caches/MLXStudio`, so nothing
writes into the signed bundle.

Logs go to `~/Library/Logs/MLXStudio/` (`setup.log`, `backend.log`;
Help → Show Logs in Finder). A packaged app has no terminal, so this is
where Python tracebacks end up.

Under `swift run`/Xcode the app skips all of this and runs the backend
from the repo's `backend/` directory, preferring `backend/.venv`.

## 6. Risks and unverified areas

- **Not yet run on Apple Silicon.** This was developed in a Linux
  container. Verified there: the backend's protocol, registry, dispatch,
  cancellation and download handling (46 pytest tests, mflux mocked);
  the dependency set resolving for macOS arm64; the exact `uv` commands
  the setup screen runs; checkpoint parameter layouts and the engine's
  calls against real mflux 0.18 code. Not verified: compiling the Swift
  code, real generation, Metal memory behavior. The Swift side did
  compile on the M5 at an earlier commit; the files changed since then
  are compiled for the first time by `build-dmg.sh`.
- **Checkpoints are community uploads** (the FLUX ones) or the mflux
  author's (Z-Image). Structure was verified; image quality wasn't.
- **Memory estimates** in the registry are reasoned, not measured.

## 7. Out of scope (future work)

From the brief: image-to-image, inpainting, outpainting, ControlNet/LoRA,
upscaling, iCloud sync, and any network feature beyond model downloads.
Also deferred: an app icon, Developer ID signing/notarization, keeping
more than one model resident, and generating `models.json` from the Python
registry at build time (a test keeps them in sync for now).
