# ming-tts-serve

ming-tts-serve runs [Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b) (inclusionAI) natively on Apple silicon. It is written in Zig on top of MLX through mlx-c, with no Python at run time.

It provides:

- `ming-tts`: an HTTP server that implements [API.md](API.md), and a CLI for one-off synthesis and benchmarks.
- The `ming` Zig module, which embeds the same engine and API routes in another Zig program, without HTTP.
- `ming-parity` and `ming-fused-bench`: development tools for numerical parity and kernel timing.

All Ming tasks are supported: TTS, voice cloning, MULTILINGUAL, BASIC, EMOTION, DIALECT, IP (built-in voices), FREEFORM and VOICE_DESIGN.

## Requirements

- A Mac with Apple silicon.
- macOS 26.2 or later. MLX is built with deployment target 26.2 so that its NAX kernels (M5 neural accelerators) are compiled in.
- Xcode 26.2 or later with the Metal toolchain. If `xcrun metal` is missing, run `xcodebuild -downloadComponent MetalToolchain`.
- CMake, to build MLX.
- Zig 0.17.

## Get the model

```sh
uvx --from huggingface_hub hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir models/Ming-3.0-tts-0.5b
```

The tools look for the model in `models/Ming-3.0-tts-0.5b` under the current directory. Use `--model DIR` or set `MING_MODEL_DIR` to point elsewhere.

## Build

```sh
git clone --recurse-submodules <this repo> ming-tts-serve
cd ming-tts-serve
zig build -Doptimize=ReleaseFast
```

The first build compiles the pinned MLX and mlx-c submodules (`lib/mlx-src`, `lib/mlxc-src`) into `lib/mlx` with `scripts/build-mlx.sh` (about two minutes on an M5 Max). Later builds reuse `lib/mlx`. After you move a submodule to a new commit, run `zig build mlx` to rebuild it.

The binaries are written to `zig-out/bin`. They find `libmlxc.dylib` and `mlx.metallib` in `lib/mlx/lib` through a path relative to the binary, so keep `zig-out` inside the checkout.

Run the unit tests with:

```sh
zig build test
```

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

## Performance

Median real-time factor from `ming-tts bench --runs 3` on an Apple M5 Max (macOS 27), bf16 weights. Lower is faster; 0.1 means 10 seconds of audio in 1 second.

| Request | Plain MLX graph (`MING_FUSED=0`) | Fused kernels (default) |
| --- | --- | --- |
| TTS, English | 0.142 | 0.103–0.108 |
| TTS, Chinese | 0.141 | 0.110 |
| Voice clone | 0.149 | 0.116 |

The plain MLX graph is itself pipelined and compiled; before those changes it ran at about 0.159 for TTS.

## Check parity

`ming-parity` compares each stage of the pipeline (text frontend, LM, DiT, CFM, aggregator, VAE, CAM++) with tensors dumped from the Python MLX engine. It always runs the plain MLX graph. The voice-clone case needs the reference clip the dumps were made with, upstream Ming's `data/voice_presets/voice_audio/lingguang.wav`:

```sh
zig-out/bin/ming-parity models/Ming-3.0-tts-0.5b <dump-dir> bf16 lingguang.wav
```

The last argument can be replaced with `MING_REF_WAV`. Use `f32` instead of `bf16` for a CPU run that must agree to rounding.

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
| `src/mlx.zig`, `src/tokenizer.zig`, `src/wav.zig`, `src/log.zig`, `src/io_util.zig` | Helpers shared with mlx-serve. |
| `scripts/build-mlx.sh`, `patches/` | MLX and mlx-c build. |

## Credits

- [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) by inclusionAI: the model, its reference implementation and the voice presets.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-c](https://github.com/ml-explore/mlx-c) by Apple.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu: this engine started as a branch of mlx-serve, and the MLX bindings, tokenizer, WAV code and MLX build script come from it. The git history of those files is kept.

## License

MIT. See [LICENSE](LICENSE). The model weights have their own license; see the model card.
