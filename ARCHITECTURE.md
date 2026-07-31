# MLX Studio — Architecture

Native macOS SwiftUI app for local text-to-image generation on Apple Silicon
using MLX, with switchable FLUX / Stable Diffusion 3.5 models, generation
history, and on-demand model downloads. No cloud inference, ever.

Target hardware for dev/test: MacBook Air, M5, 24GB unified memory, fanless.
Every design choice below is made against that ceiling, not against a Mac
Studio with 128GB and active cooling.

## 1. Inference backend: bundled Python/MLX process, not mlx-swift bindings

**Decision: shell out to a bundled Python subprocess that talks MLX
directly, communicated with over newline-delimited JSON on stdin/stdout.**
The Swift app never does inference math itself.

### Why not `mlx-swift` bindings directly

`mlx-swift` (the array/autograd library) is solid, but the *model
implementations* we need — FLUX.1/FLUX.2's rectified-flow transformer, SD3.5's
MMDiT, T5/CLIP/Qwen3 text encoders, VAEs — are not.

Researched as of 2026-07:

| Project | Language | FLUX | SD3.5 | Status |
|---|---|---|---|---|
| `ml-explore/mlx-swift-examples` | Swift (mlx-swift) | not present | has a `StableDiffusionExample` (SD1.x/SDXL-class UNet only, no MMDiT) | actively maintained, but doesn't cover our models |
| `argmaxinc/DiffusionKit` | Python (MLX) + Swift (Core ML, mlx-swift "under construction") | ✅ FLUX.1 schnell/dev | ✅ SD3/SD3.5 | **archived March 2026**, Python side works and is what we use; Swift side was never finished |
| `filipstrand/mflux` | Python (MLX) | ✅ FLUX.1 + full FLUX.2 family, actively supported | ❌ not supported | actively maintained, broad model + quantization support |
| `VincentGourbin/flux-2-swift-mlx`, `scf4/mlx-flux2`, `flux.swift` | Swift (mlx-swift) | partial FLUX.2 | none | individual/community projects, not production-grade, thin test coverage |

There is no single actively-maintained Swift/mlx-swift implementation that
covers FLUX **and** SD3.5. Building and maintaining our own Swift ports of
two different diffusion transformer architectures (rectified-flow FLUX,
MMDiT SD3.5), their text encoders (T5-XXL, CLIP-L/G, Qwen3 for FLUX.2 klein),
and their VAEs — plus keeping up with upstream fixes — is a multi-month
undertaking with real correctness risk (subtly wrong attention masking or
VAE scaling silently degrades image quality rather than crashing). That's
effort spent re-deriving what `mflux` and `diffusionkit` already got right
and keep patching, for zero product benefit.

Shelling out to Python lets the app depend on the two projects that already
do this correctly and are updated as BFL/Stability ship new checkpoints,
while the Swift layer stays what it should be: a thin, fast, native UI and
persistence layer.

### Why a subprocess pipe, not a local socket

Newline-delimited JSON over the child process's own `stdin`/`stdout` pipes,
not a TCP or Unix-domain socket:

- No port allocation, no socket file lifecycle, nothing to clean up on crash.
- The process is *only* reachable by its parent (`Process`/`NSTask`) by
  construction — there is no listening address at all, which is a stronger
  version of "no network calls" than "a localhost socket nobody else happens
  to connect to."
- Our message rate is low (a handful of progress ticks per second at most,
  plus start/end-of-generation events) and payloads are small — images are
  written to disk by the Python side and referenced by path, never
  base64-encoded through the pipe. A socket's extra complexity (framing,
  reconnect logic, port conflicts) buys nothing here.

Trade-offs accepted with this design, called out explicitly:

- The app must bundle or provision a Python 3.11+ environment with `mlx`,
  `mflux`, and `diffusionkit` installed. This is real packaging work (see
  §5) and adds ~1–2GB to install size. Mitigation: use `uv` to materialize a
  project-local virtualenv on first launch rather than shipping a frozen
  interpreter in the `.app` bundle for v1; revisit embedding a
  `python-build-standalone` interpreter directly in the bundle once the app
  is otherwise stable.
- One extra process boundary means one more thing that can crash
  independently of the UI. Mitigation: `BackendProcessManager` treats the
  backend as supervised — a dead process is detected (EOF on stdout) and
  surfaced as a recoverable error, not an app crash, with a restart action.
- The backend process is kept warm (not spawned per-request) specifically
  so a loaded model's weights stay resident in unified memory across
  consecutive generations — reloading a 4-bit FLUX checkpoint from disk
  takes real time (many seconds) that we don't want to pay per image.

## 2. Models and checkpoints (verified, not assumed)

Only one model is resident in the backend process at a time — with a 24GB
budget shared with macOS, keeping two diffusion models loaded
simultaneously is not viable. Switching models unloads the current one
before loading the next; the UI shows this as a "Loading model…" state,
distinct from "Generating…".

Memory figures below are the on-disk quantized weight size; resident RAM
while generating is higher (activations, VAE, text encoder, MLX's own
working buffers) — the app's memory guard (§4) budgets ~1.4–1.6x the
on-disk weight size plus a fixed 4GB reserved for macOS + the app itself,
and treats these as estimates to warn from, not hard cutoffs.

| Role | Model | Checkpoint researched | Engine | On-disk size | Notes |
|---|---|---|---|---|---|
| Fast/distilled | FLUX.1-schnell, 4-bit | `madroid/flux.1-schnell-mflux-4bit` (mflux-native format; `dhairyashil/FLUX.1-schnell-mflux-4bit` as fallback mirror) | `mflux` | ~7GB | 1–4 step distilled model, no CFG (guidance fixed at 1.0) — this is the "quick iteration" model |
| Quality | FLUX.1-dev, 4-bit | `madroid/flux.1-dev-mflux-4bit` (mflux-native format; `AITRADER/FLUX1-dev-mlx-4bit` as fallback) | `mflux` | ~7GB (est., same transformer size as schnell) | full CFG, 20–30 steps typical, this is the "final render" model |
| Alternate architecture | Stable Diffusion 3.5 Large, 4-bit | `argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized` | `diffusionkit` (Python) | ~4.7–5GB | Large fits comfortably once quantized, so it beats Medium on quality for no memory penalty — chosen over Medium per the "whichever fits memory better" guidance in the brief |

The registry is **not hardcoded as a single source of truth in two
languages long-term** — see `backend/mlxstudio_backend/models.py` (Python,
authoritative) and `Sources/MLXStudioApp/Resources/models.json` (Swift,
generated copy for UI display without needing the backend running yet).
Flagged as v1 tech debt in §6: a build step that generates the Swift JSON
from the Python registry instead of hand-keeping both in sync.

FLUX.2 (klein/dev) was evaluated as the "fast" and "quality" tier
respectively — `mflux` officially supports the FLUX.2 Klein family and it's
the more actively-developed line — but FLUX.1 schnell/dev was chosen for
v1 because pre-quantized, mflux-format checkpoints for FLUX.1 are more
numerous and better battle-tested at time of writing. The model registry
is a data file, not code, specifically so swapping in FLUX.2 checkpoints
later (or when better-vetted quantized FLUX.2 mirrors show up) is a config
change, not a rewrite.

**Everything in the table above was found via live search of Hugging Face
in July 2026, not assumed from training knowledge** — repo names, formats,
and file sizes in this space change often; the app treats the registry as
swappable for exactly that reason.

## 3. Swift app structure

Packaged as a Swift Package (`Package.swift`) with a macOS executable
target using the SwiftUI `App` protocol, rather than a hand-authored
`.xcodeproj`. Opens directly in Xcode (`File > Open...` on `Package.swift`)
or builds with `swift build`/`swift run`. This keeps the project
diffable/reviewable in plain text and avoids maintaining a generated
`.pbxproj`, at the cost of a couple of Xcode conveniences (asset catalogs
need `Bundle.module`, entitlements are wired via `Package.swift` plugin
settings) — an acceptable trade for a project this size.

```
Sources/MLXStudioApp/
  App.swift                        @main App, wires environment objects
  Models/
    GenerationParameters.swift     steps, guidance, seed, size, batch count
    GenerationRequest.swift        prompt + params + model id
    HistoryEntry.swift             persisted record of one generated image
    ModelDescriptor.swift          registry entry (id, engine, repo, footprint)
    BackendMessages.swift          Codable wire protocol (request/response/progress/error)
  Services/
    BackendProcessManager.swift    spawns/supervises the Python subprocess
    InferenceBackendClient.swift   line-buffered JSON-lines request/response + progress streams
    ModelManager.swift             download orchestration, active-model state
    MemoryGuard.swift              host_statistics64-based available RAM check
    HistoryStore.swift             JSON index + file-backed image storage
  ViewModels/
    GenerationViewModel.swift      drives the generate form + progress state
    GalleryViewModel.swift         drives history browsing/reuse
  Views/
    ContentView.swift              split view: generate / gallery
    ModelPickerView.swift          segmented control across the 3 models
    PromptFormView.swift           prompt, negative prompt, params
    GenerationProgressView.swift   per-step progress bar or elapsed-time spinner
    GalleryView.swift              grid of past generations
    GenerationDetailView.swift     metadata detail + "reuse settings"
    MemoryWarningView.swift        non-blocking warning sheet
  Resources/
    models.json
```

### Data flow — generation

1. `PromptFormView` edits a `GenerationRequest` bound in `GenerationViewModel`.
2. On submit, `GenerationViewModel` asks `MemoryGuard` to check the selected
   model's footprint against currently-free RAM; if short, shows
   `MemoryWarningView` (proceed anyway / cancel) instead of continuing blind.
3. `GenerationViewModel` calls `InferenceBackendClient.generate(request)`,
   which lazily asks `BackendProcessManager` to ensure the subprocess is
   running, then writes one JSON line (`{"type":"generate", "id", "model",
   "prompt", "negative_prompt", "steps", "guidance", "seed", "width",
   "height", "batch_size"}`) to its stdin.
4. The backend loads the model (skipped if already the active model),
   denoises, and streams `{"type":"progress","id","step","total_steps",
   "elapsed_s"}` lines as they become available, then
   `{"type":"result","id","images":[{"path","seed"}]}` or
   `{"type":"error","id","code","message"}`.
5. `InferenceBackendClient` demultiplexes lines by `id` (a `UUID` per
   request) onto `AsyncStream`s the ViewModel consumes; progress updates a
   `@Published` state driving `GenerationProgressView`, which shows
   per-step progress when `total_steps` is known and falls back to an
   indeterminate spinner + elapsed-time counter when it isn't (e.g. during
   model load) — no hardcoded "should take N seconds" anywhere in the UI,
   per the thermal-throttling constraint.
6. On `result`, `GenerationViewModel` hands the image paths + full request
   metadata to `HistoryStore.record(...)`, which copies the file(s) into
   `Application Support/MLXStudio/Generations/` (if not already written
   there by the backend) and appends a `HistoryEntry` to the JSON index.
   `GalleryViewModel` observes the store and updates the grid.

### Data flow — model download

`ModelManager.ensureDownloaded(modelId)` sends `{"type":"download","id",
"model"}`; the backend uses `huggingface_hub`'s `snapshot_download` with a
progress callback, streaming `{"type":"download_progress","id",
"bytes_done","bytes_total"}` lines. The UI shows a determinate progress bar
since Hugging Face reports content-length. Downloads are cached under
`~/Library/Application Support/MLXStudio/models/<repo>/` and re-checked
(not re-downloaded) on subsequent launches — `snapshot_download`'s own
local-cache behavior handles this for free.

## 4. Handling the hardware constraints

- **Memory**: `MemoryGuard` reads `ProcessInfo.processInfo.physicalMemory`
  and free/inactive pages via `host_statistics64`, compares against the
  selected model's declared footprint (registry) plus a fixed reserve for
  macOS + the app (~4GB). Short-on-memory is a warning sheet with an
  informed "try anyway" override, never a silent hang or a crash — actual
  OOM behavior in MLX/Metal on a real 24GB M5 unit needs to be verified on
  hardware (flagged in §6, this container has no Apple Silicon to test on).
- **No active cooling / thermal throttling**: the UI never promises a
  duration. Progress is step-based when the backend reports steps, elapsed
  time always shown, batches process sequentially with per-image results
  streamed back individually (so a throttled 5th image in a batch of 5
  doesn't block the first four from already being visible in the gallery).

## 5. Model management & first-run

First selection of a not-yet-downloaded model shows a download sheet
before enabling "Generate". `ModelManager` persists which models have
completed download (checking the on-disk cache is authoritative; a local
"download complete" flag is a cache of that, not the source of truth, so a
partially-downloaded or manually-deleted model is re-detected rather than
assumed present).

## 6. Explicitly flagged risks / immature areas

- **mlx-swift diffusion support is immature** for our target models (§1) —
  this is the central reason for the subprocess architecture. Revisit if
  `mlx-swift-examples` gains a maintained FLUX/SD3.5 implementation.
- **DiffusionKit (Python) is archived** (March 2026). It still works and is
  pinned to a specific released version, but receives no further fixes.
  If it breaks against a future `mlx` release, the fallback is to drop the
  SD3.5 tier or replace it with an `mflux`-supported alternative (e.g.
  Z-Image or Qwen-Image) via the same swappable registry — not a rewrite.
- **No per-step progress guarantee from every engine.** `mflux` exposes a
  callback system (`callback.py`/`callback_manager.py`); `diffusionkit`'s
  step-level reporting wasn't confirmed in this research pass. The wire
  protocol and UI both already treat `progress` events as optional/best
  effort (indeterminate spinner fallback), so this degrades gracefully
  either way.
- **Nothing in this repo has been built or run** — this development
  container is Linux with no Swift toolchain and no Apple Silicon/Metal,
  so neither the SwiftUI app nor the MLX Python backend can be compiled or
  executed here. The Python backend's protocol/registry/memory-check logic
  is unit-tested in this sandbox with the inference engines mocked out
  (see `backend/tests/`); the Swift code compiles by inspection against
  documented SwiftUI/Foundation APIs but is unverified by an actual build.
  **This must be built and exercised on the target M5 MacBook Air before
  being considered working software.**

## 7. Out of scope for v1 (future work)

Per the brief: image-to-image, inpainting, outpainting, ControlNet/LoRA,
upscaling, iCloud sync, and any cloud/network feature beyond the one-time
model download. Also deferred: multi-model-resident-at-once, cancel
mid-step (protocol has room for it via an `{"type":"cancel","id"}` message
but it isn't wired up end-to-end in v1), and auto-generating
`models.json` from the Python registry at build time (currently
hand-synced, called out above as tech debt).
