# ming-tts-serve

[![CI](https://github.com/sbusso/ming-tts-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/sbusso/ming-tts-serve/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/sbusso/ming-tts-serve)](https://github.com/sbusso/ming-tts-serve/releases/latest)

ming-tts-serve runs [Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b) (inclusionAI) natively on Apple silicon. It is written in Zig on top of MLX through mlx-c, with no Python at run time.

It provides:

- `ming-tts`: an HTTP server that implements [API.md](API.md), and a CLI for one-off synthesis and benchmarks.
- The `ming` Zig module, which embeds the same engine and API routes in another Zig program, without HTTP.
- `ming-parity` and `ming-fused-bench`: development tools for numerical parity and kernel timing.

All Ming tasks are supported: TTS, voice cloning, MULTILINGUAL, BASIC, EMOTION, DIALECT, IP (built-in voices), FREEFORM and VOICE_DESIGN.

## Install

On a Mac with Apple silicon and macOS 26.2 or later:

```sh
curl -fsSL https://raw.githubusercontent.com/sbusso/ming-tts-serve/main/install.sh | sh
```

The script installs the latest release in `~/.local/share/ming-tts-serve`, adds the `ming-tts` command to `~/.local/bin`, and downloads the model (about 3 GB) when [uv](https://docs.astral.sh/uv/) is installed. Then:

```sh
ming-tts say --out hello.wav --json '{"task":"IP","voice":"UK_George","text":"Hello from Ming."}'
afplay hello.wav
```

Options, set before `sh`: `MING_VERSION=0.1.0` installs a given release, `MING_NO_MODEL=1` skips the model download, and `MING_PREFIX` and `MING_BIN_DIR` change the install folders. To install by hand, download a tarball from the [releases page](https://github.com/sbusso/ming-tts-serve/releases/latest) and read [install.sh](install.sh).

The rest of this README uses `zig-out/bin/ming-tts`, the path of a source build. After the script install, use `ming-tts` instead.

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

Other tasks detect these languages from the text (see `MING_LANG_DETECT`). Detected text keeps its accents and skips the Chinese/English normalizer, and French numbers are spelled out in French. TTS, BASIC and EMOTION also get the `Target language:` prompt line. IP, FREEFORM and VOICE_DESIGN do not, because the line changes the built-in or described voice. Upstream sends such text through the Chinese/English normalizer, which spells numbers in English and replaces accented letters with `，`.

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

## Long text

Requests accept up to 50,000 characters by default (about an hour of speech; `MING_MAX_TEXT_CHARS` changes the limit). The engine splits the text and generates it one fragment at a time and frees MLX's buffer cache after each one, so GPU memory stays the same for any length (peak memory footprint of `ming-tts say` about 5.5 GB for 5 minutes, against 31 GB without freeing the cache); the finished audio takes about 10.6 MB of memory per minute. Generation runs at about 0.11 s per second of audio: a 5-minute text takes about 35 seconds on an M5 Max.

How the text is split:

1. Paragraphs: every line break starts a paragraph, except a hard wrap (a line that does not end a sentence, followed by a line that starts with a lowercase letter), which is joined with a space.
2. Sentences and clauses: each paragraph goes through the upstream frontend, which cuts it into fragments of about 24 words or Chinese characters at sentence ends, then at commas.

Each pair of consecutive fragments is joined with a pause chosen by what separates them. The engine cuts each fragment's leading and trailing silence (keeping 40 ms before the speech and 80 ms after it), fades the cut edges over 10 ms, and inserts silence so that the gap from the end of one fragment's speech to the start of the next is:

| Boundary | Pause |
| --- | --- |
| Inside a sentence (the fragment ends at a comma or mid-clause) | 0.20 s |
| End of a sentence | 0.40 s |
| End of a paragraph | 0.85 s |

The values follow the model's own pauses inside a fragment: on a 5-minute English clone, the median pause after a comma is 120 ms (90th percentile 280 ms) and after a sentence end 390 ms (90th percentile 460 ms). The `pause_scale` request field multiplies all three (0 to 4, default 1); the gap never drops below the 120 ms of kept edges. The start of the first fragment and the end of the last keep the model's own silence, so text that fits in one fragment produces the same audio as before. Streaming clients receive the pauses as runs of zero samples.

`MING_JOIN=0` turns this off and concatenates the fragments as upstream does (each fragment keeps up to 300 ms of trailing silence and all of its leading silence, whatever the boundary).

Measured on 5-minute English (built-in voice and clone) and 2.5-minute French (clone) texts: Whisper word error rate 0.4 to 2.7 % (mostly digits versus spelled numbers), CAM++ similarity between the first and last 30 seconds 0.95 to 0.98, no clipped samples, and fragment speech levels within ±1 dB (largest jump between neighbors 2.6 dB). Every fragment is generated from the same prompt with no state carried over, so voice and speaking rate do not drift: the per-fragment speaking rate changes by -3 % to +3 % from the first to the last quarter of the English texts, which is the spread of the text itself.

Breaths: the model's `[Breathing]` tag at the start of a paragraph was tested and is not used. Over 40 tagged clips (10 sentences in English, French and Chinese, 4 voices, 4 seeds each), an audible breath (0.2 s or more before the speech) appeared in 22; it lowered CAM++ similarity to the untagged clip (0.82 to 0.91, against 0.89 to 0.94 between seeds) and added a character error to the first word in 3 of 8 Chinese clips. No words were inserted.

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
| | `MING_MAX_TEXT_CHARS` | `50000` |

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
| `MING_LANG_DETECT` | `auto` | Language detection for tasks other than MULTILINGUAL. `auto`: French, German, Spanish, Japanese or Korean text takes the MULTILINGUAL text path, and TTS, BASIC and EMOTION also get the `Target language:` prompt line. `text`: the text path only. `prompt`: the line for every task except DIALECT. `0`: off, as upstream. |
| `MING_JOIN` | on | Join text fragments with pauses by boundary (see [Long text](#long-text)). `0` concatenates them as upstream does. |
| `MING_CLEAR_CACHE` | on | Free MLX's buffer cache after each fragment of a multi-fragment text. Output is unchanged; `0` keeps the cache, which grows with the text. |
| `MING_PROFILE` | off | Any value prints per-step timing and step budgets to stderr. |

With `MING_FUSED=0`, `MING_PROMPT_NORM=0`, `MING_SOFT_LIMIT=0`, `MING_GUARD=0`, `MING_FR_TN=0`, `MING_LANG_DETECT=0` and `MING_JOIN=0`, the engine follows the upstream Ming pipeline op for op.

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

`Service.handle` takes the same JSON as the HTTP server and returns the same JSON. `Service.handleStream` also passes each text fragment's audio, and the pause after it, to a callback as soon as it is generated; the callback's optional `progress` function receives the fragment count and the share of the text done, for a progress bar or a time estimate. For a typed API without JSON, use `ming.engine` (`src/ming_tts.zig`). All MLX work runs on one engine thread, so `handle` can be called from any thread.

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
