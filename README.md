# ming-tts-serve

[![CI](https://github.com/sbusso/ming-tts-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/sbusso/ming-tts-serve/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/sbusso/ming-tts-serve)](https://github.com/sbusso/ming-tts-serve/releases/latest)

ming-tts-serve runs [Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b) (inclusionAI) natively on Apple silicon. It is written in Zig on top of MLX through mlx-c, with no Python at run time.

It provides:

- `ming-tts`: an HTTP server that implements [API.md](API.md), and a CLI for one-off synthesis and benchmarks.
- The `ming` Zig module, which embeds the same engine and API routes in another Zig program, without HTTP.
- `ming-parity` and `ming-fused-bench`: development tools for numerical parity and kernel timing.

All Ming tasks are supported: TTS, voice cloning, MULTILINGUAL, BASIC, EMOTION, DIALECT, IP (built-in voices), FREEFORM and VOICE_DESIGN.

## Install from a release

A release is a prebuilt `ming-tts` binary with the MLX runtime. It needs a Mac with Apple silicon and macOS 26.2 or later, but no Xcode, Zig or CMake. To build it yourself instead, see [Build from source](#build-from-source).

1. Download and check the tarball from the [releases page](https://github.com/sbusso/ming-tts-serve/releases/latest). Set `VERSION` to the release you want, without the leading `v`:

   ```sh
   VERSION=0.1.0
   NAME=ming-tts-serve-$VERSION-macos-arm64
   curl -fLO https://github.com/sbusso/ming-tts-serve/releases/download/v$VERSION/$NAME.tar.gz
   curl -fLO https://github.com/sbusso/ming-tts-serve/releases/download/v$VERSION/$NAME.tar.gz.sha256
   shasum -a 256 -c $NAME.tar.gz.sha256
   ```

2. Extract it anywhere. The folder holds `bin/ming-tts`, the MLX libraries in `lib/`, and the docs. Keep `bin/` and `lib/` together; the binary finds the libraries through a path relative to itself.

   ```sh
   tar -xzf $NAME.tar.gz
   cd $NAME
   ```

   If you downloaded the tarball with a browser, macOS marks the files as quarantined and refuses to run them. Clear the mark:

   ```sh
   xattr -dr com.apple.quarantine .
   ```

   The binaries are ad-hoc signed, not notarized by Apple. Notarization is planned for a later release.

3. Download the model (about 3 GB) with [uv](https://docs.astral.sh/uv/):

   ```sh
   uvx --from huggingface_hub hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir models/Ming-3.0-tts-0.5b
   ```

4. Make your first audio:

   ```sh
   bin/ming-tts say --out hello.wav \
     --json '{"task":"IP","voice":"UK_George","text":"Hello from Ming."}'
   afplay hello.wav
   ```

   `ming-tts` looks for the model in `models/Ming-3.0-tts-0.5b` under the current directory. From another directory, pass `--model DIR` or set `MING_MODEL_DIR`.

The rest of this README uses `zig-out/bin/ming-tts`, the path of a source build. With a release, use `bin/ming-tts` instead.

## Build from source

These steps take you from a new Mac to spoken audio in about 10 minutes. Most of that time is the model download and the first build of MLX.

### 1. Check the requirements

- A Mac with Apple silicon (M1 or later; the fused kernels are tuned for M5).
- macOS 26.2 or later.
- About 5 GB of free disk space: 3 GB for the model, the rest for MLX and the build.

### 2. Install the tools

1. Install Xcode 26.2 or later from the App Store, then open it once to accept the license.
2. Install the Metal toolchain:

   ```sh
   xcodebuild -downloadComponent MetalToolchain
   ```

3. Install Zig 0.17, CMake and uv with [Homebrew](https://brew.sh):

   ```sh
   brew install zig cmake uv
   ```

4. Check the versions:

   ```sh
   zig version          # 0.17.0
   xcrun -f metal       # prints a path
   ```

### 3. Get the code and build it

```sh
git clone --recurse-submodules https://github.com/sbusso/ming-tts-serve.git
cd ming-tts-serve
zig build -Doptimize=ReleaseFast
```

The first build compiles the pinned MLX and mlx-c submodules into `lib/mlx` (about 2 minutes on an M5 Max). Later builds reuse it. The binaries are written to `zig-out/bin`.

If you cloned without `--recurse-submodules`, run `git submodule update --init` first.

### 4. Download the model

Run this in the `ming-tts-serve` folder. The tools look for the model in `models/Ming-3.0-tts-0.5b` by default:

```sh
uvx --from huggingface_hub hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir models/Ming-3.0-tts-0.5b
```

To keep the model somewhere else, pass `--model DIR` or set `MING_MODEL_DIR`.

### 5. Make your first audio

```sh
zig-out/bin/ming-tts say --out hello.wav \
  --json '{"task":"IP","voice":"UK_George","text":"Hello from Ming."}'
afplay hello.wav
```

### 6. Clone a voice

Record 5 to 15 seconds of clean speech, and write down exactly what you said:

```sh
zig-out/bin/ming-tts say --out clone.wav --json '{
  "task": "TTS",
  "text": "This is my cloned voice.",
  "prompt_wav_path": "/abs/path/my-voice.wav",
  "prompt_text": "Exact transcript of my-voice.wav."
}'
afplay clone.wav
```

For French, German, Spanish, Japanese or Korean, use `"task": "MULTILINGUAL"` with `"language": "fr"` (or `de`, `es`, `ja`, `ko`).

### 7. Run the server

```sh
zig-out/bin/ming-tts serve
```

In another terminal:

```sh
curl -s localhost:8765/health      # wait for "loaded": true
curl -s localhost:8765/tts -H 'Content-Type: application/json' \
  -d '{"text":"Hello from the server."}'
```

The response holds the path of the generated WAV file. See [Run the server](#run-the-server) for options and [API.md](API.md) for every route.

### 8. Run the tests (optional)

```sh
zig build test
```

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `xcrun: error: unable to find utility "metal"` | Run `xcodebuild -downloadComponent MetalToolchain`. |
| `cmake: command not found` during the first build | Run `brew install cmake`. |
| Build fails in `lib/mlx-src` or `lib/mlxc-src` | Run `git submodule update --init`, then `zig build mlx`. |
| `missing weight` or `FileNotFound` at start | The model is not in `models/Ming-3.0-tts-0.5b`. Pass `--model DIR` or set `MING_MODEL_DIR`. |
| `dyld: Library not loaded: libmlxc.dylib` | Run the binaries from `zig-out/bin` inside the checkout; they find MLX through a relative path. For a release, keep `bin/` and `lib/` in the same folder. |
| A release binary is "damaged" or "cannot be opened" | Clear the browser quarantine: `xattr -dr com.apple.quarantine <release folder>`. |

## Use the CLI

Synthesize one request to a WAV file. The JSON body is the same as `POST /generate` in [API.md](API.md):

```sh
zig-out/bin/ming-tts say --out hello.wav \
  --json '{"task":"IP","voice":"UK_George","text":"Hello from Ming."}'
```

Measure the real-time factor (generation time divided by audio duration) over fixed requests. The clone case runs only when you pass a reference clip:

```sh
zig-out/bin/ming-tts bench --runs 3 --ref-wav ref.wav
```

Print the text frontend's fragments and token ids without loading the model weights:

```sh
zig-out/bin/ming-tts frontend --json '[{"task":"TTS","text":"Bonjour à tous."}]'
```

## Run the server

```sh
zig-out/bin/ming-tts serve
```

The server listens on `http://127.0.0.1:8765`. Options:

| Option | Environment variable | Default |
| --- | --- | --- |
| `--model DIR` | `MING_MODEL_DIR` | `models/Ming-3.0-tts-0.5b` |
| `--host ADDR` | | `127.0.0.1` |
| `--port N` | `MING_PORT` | `8765` |
| `--output-dir DIR` | `MING_OUTPUT_DIR` | `~/Library/Caches/ming-tts` |
| | `MING_MAX_TEXT_CHARS` | `1000` |

The model loads in the background. `GET /health` reports `"loaded": true` when it is ready.

```sh
curl -s localhost:8765/generate -H 'Content-Type: application/json' \
  -d '{"task":"EMOTION","text":"What a day!","prompt_wav_path":"/abs/path/ref.wav","controls":{"emotion":"高兴"}}'
```

The response holds the absolute path of the generated WAV file. See [API.md](API.md) for every route, field and error.

## Environment switches

These switches change engine behavior. They exist for A/B comparisons and diagnostics; the defaults are the recommended settings. Set a switch to `0` to turn its feature off.

| Variable | Default | Effect |
| --- | --- | --- |
| `MING_FUSED` | on | Fused Metal kernels for the DiT stacks and the LM decode step. `0` runs the plain MLX graph, which mirrors the Python engine op for op. The fused kernels round at the same points but are not bit-exact (about 1e-2 relative per CFM sample). |
| `MING_FUSED_DIT` | on | Fused kernels for the DiT stacks only (flow head and aggregator). |
| `MING_FUSED_LM` | on | Fused kernels for the Qwen2 LM decode step only. |
| `MING_PROMPT_NORM` | on | Bring the reference clip to the built-in voices' speech level before cloning. |
| `MING_SOFT_LIMIT` | on | Soft-limit output peaks (ceiling 0.98) instead of hard clipping at 1.0. |
| `MING_GUARD` | on | Regenerate text fragments whose speaking rate is implausible. |
| `MING_FR_TN` | on | French text normalization for MULTILINGUAL with language `fr`: spells out numbers, times and amounts, and flattens French typography. |
| `MING_PROFILE` | off | Any value prints per-step timing and step budgets to stderr. |

With `MING_FUSED=0`, `MING_PROMPT_NORM=0`, `MING_SOFT_LIMIT=0`, `MING_GUARD=0` and `MING_FR_TN=0`, the engine follows the upstream Ming pipeline op for op.

## Embed the engine in a Zig program

Add the package as a dependency in your `build.zig.zon`:

```zig
.dependencies = .{
    .ming_tts_serve = .{ .path = "../ming-tts-serve" },
},
```

Link it in your `build.zig`. Build the engine with `ReleaseFast` even when your program is a Debug build, because the host-side text and audio code is slow without optimization:

```zig
const ming_tts = @import("ming_tts_serve");

const dep = b.dependency("ming_tts_serve", .{ .target = target, .optimize = .ReleaseFast });
ming_tts.link(dep, exe); // makes @import("ming") available to exe
```

If `lib/mlx` is not built yet, `link` makes your build run `scripts/build-mlx.sh` in the dependency first. The module records an absolute rpath to the dependency's `lib/mlx/lib`.

Call the API routes in process:

```zig
const ming = @import("ming");

const svc = try ming.Service.start(io, gpa, .{ .model_dir = "/abs/path/Ming-3.0-tts-0.5b" });
var status: u16 = 0;
const json = try svc.handle(arena, "POST", "/generate", body, &status);
```

`Service.handle` takes the same JSON as the HTTP server and returns the same JSON. `Service.handleStream` also passes each text fragment's audio to a callback as soon as it is generated. For a typed API without JSON, use `ming.engine` (`src/ming_tts.zig`). All MLX work runs on one engine thread, so `handle` can be called from any thread.

## Audio samples

These clips were generated with `ming-tts say` and encoded to AAC. GitHub serves `.m4a` files as downloads, so they open in your audio player rather than in the browser.

- [English, built-in voice `UK_George`](docs/samples/en-uk-george.m4a) (8.8 s)
- [Chinese, built-in voice `灵小柔`](docs/samples/zh-ling-xiaorou.m4a) (6.0 s)
- [French, `MULTILINGUAL` with language `fr`, cloned from upstream's `lingguang.wav` Mandarin voice preset](docs/samples/fr-lingguang.m4a) (5.0 s)

`docs/samples/make.sh` regenerates them.

## Engine comparison

The same three requests on each engine: an English sentence and a Chinese sentence with a built-in voice, and a voice clone from a 4-second reference clip. Each request runs 3 times with seeds 0–2 after a warm-up; the table shows the median real-time factor (RTF), which is generation time divided by audio duration. Lower is faster: 0.1 means 10 seconds of audio in 1 second.

Measured on an Apple M5 Max (128 GB, macOS 27.0.1), bf16 weights, one request at a time, on October 30, 2026.

| Engine | TTS, English | TTS, Chinese | Voice clone | Speed vs. official |
| --- | --- | --- | --- | --- |
| Official [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) code, PyTorch 2.6.0 on MPS | 1.841 | 1.907 | 1.951 | 1× |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.6.3, Python, MLX 0.32.4 | 0.231 | 0.236 | 0.208 | about 8× |
| ming-tts-serve, plain MLX graph (`MING_FUSED=0`) | 0.133 | 0.137 | 0.144 | about 14× |
| **ming-tts-serve, fused kernels (default)** | **0.104** | **0.106** | **0.112** | **about 18×** |

The mlx-audio row uses its own Ming-3.0 pipeline with two changes that make it do the same work as the other engines: the upstream VAE posterior formula for the clone prompt (mlx-audio 0.6.3 uses a different one), and a CAM++ speaker embedding for the clone. [bench/README.md](bench/README.md) describes both.

What makes the difference:

- **mlx-audio vs. PyTorch:** the same Python pipeline on MLX instead of PyTorch's MPS backend.
- **Plain MLX graph vs. mlx-audio:** no Python, the whole flow-matching sampler compiled as one graph, the next decode step queued while the GPU runs the current one, and fixed tables (rotary, timestep embeddings) computed once at load. The output matches the Python MLX engine (see [Check parity](#check-parity)).
- **Fused kernels vs. plain MLX graph:** custom Metal kernels run each DiT block as 4 GPU dispatches instead of about 15 and read each weight once per Euler step. This cuts the flow head from about 2,350 to about 340 dispatches per decode step. The fused kernels are not bit-exact, so the same seed gives a slightly different clip. On 4 cases (English, Chinese, French and a voice clone) × 5 seeds, the Whisper word or character error rate is 0.018 for the plain graph and 0.013 for the fused kernels, and the CAM++ speaker similarity to the reference clip is 0.733 and 0.741. The differences are within seed-to-seed variation. See [bench/README.md](bench/README.md#quality-check).

Other differences that matter in practice:

| | Official code | mlx-audio | ming-tts-serve |
| --- | --- | --- | --- |
| Time to first audio | full request | first text fragment, with `stream=True` | about 0.7 s (streams each text fragment) |
| Runtime | Python, PyTorch | Python, MLX | one native binary, no Python |
| Tasks | all | all | all |
| French numbers and typography | digits left to the model | digits left to the model | spelled out in French (`MING_FR_TN`) |

To reproduce the table, see [bench/README.md](bench/README.md). It runs the PyTorch and mlx-audio harnesses in `bench/` and `ming-tts bench --runs 3 --ref-wav lingguang.wav` with `MING_FUSED=0` and `1`.

## Check parity

`ming-parity` is a development tool. It compares each stage of the pipeline (text frontend, LM, DiT, CFM, aggregator, VAE, CAM++) with tensors dumped from a Python MLX reference run, and always runs the plain MLX graph. The dumps are not part of this repository, so most users do not need it.

```sh
zig-out/bin/ming-parity models/Ming-3.0-tts-0.5b <dump-dir> bf16 lingguang.wav
```

The last argument is the reference clip the dumps were made with (upstream Ming's `data/voice_presets/voice_audio/lingguang.wav`); `MING_REF_WAV` can replace it. Use `f32` instead of `bf16` for a CPU run that must agree to rounding.

## Make a release

1. Set `.version` in `build.zig.zon` to the new version, and commit.
2. Tag the commit with the same version and a leading `v`, and push the tag:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

The release workflow checks that the tag matches `build.zig.zon`, runs the tests, builds the tarball with `scripts/package.sh` and publishes it as a GitHub release. To build the same tarball locally, run `scripts/package.sh`; it writes to `dist/`.

## Layout

| Path | Contents |
| --- | --- |
| `src/ming_model.zig` | Model: Qwen2 LM, DiT flow head, aggregator, VAE, sampling loop. |
| `src/ming_fused.zig` | Fused Metal kernels. |
| `src/ming_tts.zig` | Engine: requests, fragments, streaming, cloning. |
| `src/ming_text.zig`, `src/ming_fr.zig` | Text frontend, prompt layout, French normalization. |
| `src/ming_audio.zig`, `src/ming_campplus.zig` | Audio I/O and resampling, CAM++ speaker embedding. |
| `src/ming_service.zig` | In-process API service (the `ming` module root). |
| `src/ming_main.zig` | `ming-tts` server and CLI. |
| `src/mlx.zig`, `src/tokenizer.zig`, `src/wav.zig`, `src/log.zig`, `src/io_util.zig` | MLX bindings, tokenizer, WAV and logging helpers, adapted from mlx-serve. |
| `scripts/build-mlx.sh`, `patches/` | MLX and mlx-c build. |
| `scripts/package.sh` | Release tarball (also run by `.github/workflows/release.yml`). |

## Credits

- [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) by inclusionAI: the model, its reference implementation and the voice presets.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-c](https://github.com/ml-explore/mlx-c) by Apple.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu: this engine started as a branch of mlx-serve, and the MLX bindings, tokenizer, WAV code and MLX build script come from it. The git history of those files is kept.

## License

MIT. See [LICENSE](LICENSE). The model weights have their own license; see the model card.
