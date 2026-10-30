# Benchmarks

This folder reproduces the [Engine comparison](../README.md#engine-comparison) in the main README:

- `engines.py` times the official Ming-3.0-tts PyTorch code and mlx-audio on the same requests as `ming-tts bench`.
- `quality.py` checks that the fused kernels (`MING_FUSED=1`) do not change speech quality compared with the plain MLX graph (`MING_FUSED=0`).

## Requirements

- A built `zig-out/bin/ming-tts` and the model in `models/Ming-3.0-tts-0.5b` (see the main README). Set `MING_TTS` or `MING_MODEL_DIR` to use other paths.
- [uv](https://docs.astral.sh/uv/). Each script runs in one dependency group of `pyproject.toml`; `uv run --group NAME` syncs `.venv` to that group. The groups conflict with each other (the official code pins torch 2.6.0 and transformers 4.46.3; mlx-audio needs transformers 5), so switching groups reinstalls packages.
- Network access on the first run: the scripts clone upstream [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) at a pinned commit into `~/.cache/ming-tts-bench/Ming-3.0-tts`, and `quality.py` downloads the Whisper model. Set `MING_UPSTREAM_DIR` to use an existing checkout.

The voice-clone reference clip is upstream's `data/voice_presets/voice_audio/lingguang.wav` (4 seconds of Mandarin), with the transcript `诶，你今天有空吗？我们能不能聊聊这件事啊？`.

Run one benchmark at a time on an otherwise idle Mac.

## Engine comparison

Run these commands from this folder:

```sh
uv run --group torch python engines.py torch
uv run --group mlx-audio python engines.py mlx-audio

REF=~/.cache/ming-tts-bench/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav
MING_FUSED=0 ../zig-out/bin/ming-tts bench --model ../models/Ming-3.0-tts-0.5b --runs 3 --ref-wav $REF
MING_FUSED=1 ../zig-out/bin/ming-tts bench --model ../models/Ming-3.0-tts-0.5b --runs 3 --ref-wav $REF
```

Run one of the `engines.py` commands first, so that the upstream checkout with the reference clip exists.

Each command prints one line per request with the median real-time factor (RTF) and the median audio duration. RTF is the synthesis wall time divided by the audio duration. Every engine gets one warm-up request, then runs each request with seeds 0, 1 and 2:

| Request | Text | Voice |
| --- | --- | --- |
| `tts-en` | The garden will remain open until sunset, provided the weather stays clear, and visitors are asked to leave by the north gate. | Built-in `UK_George` |
| `tts-zh` | 雨后的山谷很安静，远处的云慢慢变成了金色，我们沿着小路走回了村子。 | Built-in `灵小柔` |
| `clone` | I never thought a little machine could sound this much like me. | Clone of `lingguang.wav`, with its transcript and CAM++ speaker embedding |

### How each engine runs

- **Official code:** upstream `MingTTS.synthesize` from `ming_tts.py`, bf16, on the MPS device. Two changes let it run off CUDA: `torch.autocast(device_type="cuda")` is redirected to MPS, and `PYTORCH_ENABLE_MPS_FALLBACK=1` runs the vocoder's `col2im` on the CPU. pynini (the Chinese text normalizer) has no reliable macOS wheel, so the normalizer is replaced by an identity function; the benchmark texts contain no numbers, so the token ids are unchanged.
- **mlx-audio 0.6.3:** its own Ming-3.0 pipeline, `Model.generate`, with the model loaded from the same local folder. Two changes make the output match the other engines:
  - The VAE posterior. mlx-audio 0.6.3 samples the clone prompt latent with `std = exp(0.5 * logvar)`. Upstream (diffusers' Oobleck VAE) uses `std = softplus(scale) + 1e-4`. `engines.py` replaces `encode_latent` with the upstream formula; the cost is the same.
  - The speaker embedding. mlx-audio does not compute a CAM++ embedding for clones. The other engines do, so `engines.py` computes it with `campplus.onnx` on onnxruntime and passes it as `spk_emb`. This time is included in the measurement.
- **ming-tts-serve:** `ming-tts bench`, with `MING_FUSED=0` for the plain MLX graph and `MING_FUSED=1` (the default) for the fused kernels.

### Results

Apple M5 Max, 128 GB, macOS 27.0.1, bf16 weights, October 30, 2026. PyTorch 2.6.0, mlx-audio 0.6.3 with MLX 0.32.4.

```text
torch-mps  tts-en  rtf=1.841  audio=8.20s
torch-mps  tts-zh  rtf=1.907  audio=6.80s
torch-mps  clone   rtf=1.951  audio=3.20s
mlx-audio  tts-en  rtf=0.231  audio=8.00s
mlx-audio  tts-zh  rtf=0.236  audio=6.40s
mlx-audio  clone   rtf=0.208  audio=3.70s
MING_FUSED=0
zig        tts-en  rtf=0.133  audio=8.00s
zig        tts-zh  rtf=0.137  audio=6.40s
zig        clone   rtf=0.144  audio=3.30s
MING_FUSED=1
zig        tts-en  rtf=0.104  audio=7.90s
zig        tts-zh  rtf=0.106  audio=6.30s
zig        clone   rtf=0.112  audio=3.10s
```

RTF varies by a few percent between runs. The engines draw different random noise for the same seed, so audio durations differ slightly.

## Quality check

```sh
uv run --group quality python quality.py
```

The script starts `ming-tts serve` once with `MING_FUSED=0` and once with `MING_FUSED=1`, and generates each case below with seeds 0 to 4 (change the count with `--seeds`):

| Case | Request |
| --- | --- |
| `en` | `IP`, voice `UK_George`: "Please remember to bring an umbrella, because the forecast says it will rain all afternoon." |
| `zh` | `IP`, voice `灵小柔`: "今天的会议改到下午举行，请大家提前准备好材料，并在门口签到。" |
| `fr` | `MULTILINGUAL`, language `fr`, clone of `lingguang.wav`: "Le train pour Lyon partira avec un léger retard à cause de la neige tombée cette nuit." |
| `clone` | `TTS`, clone of `lingguang.wav`: "I never thought a little machine could sound this much like me." |

It then measures:

- **Intelligibility:** [Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) (mlx-whisper) transcribes each clip with the expected language. The script scores the transcript against the input text after lowercasing and removing punctuation: word error rate (WER) for English and French, character error rate (CER) for Chinese.
- **Speaker similarity to the reference:** cosine similarity of CAM++ embeddings (`campplus.onnx` from the model folder) between each clone and `lingguang.wav`.
- **Plain vs. fused similarity:** CAM++ cosine similarity between the plain and the fused clip of the same case and seed.

The script prints each transcript, then a summary of means over the seeds.

### Results

Same machine and date as above, 5 seeds per case:

| Case | WER/CER, plain | WER/CER, fused | Similarity to reference, plain | Similarity to reference, fused | Similarity, plain vs. fused |
| --- | --- | --- | --- | --- | --- |
| en | 0.000 | 0.000 | - | - | 0.963 |
| zh | 0.000 | 0.000 | - | - | 0.993 |
| fr | 0.071 | 0.035 | 0.766 | 0.783 | 0.974 |
| clone | 0.000 | 0.017 | 0.699 | 0.700 | 0.967 |
| all | 0.018 | 0.013 | 0.733 | 0.741 | 0.974 |

The fused kernels are not bit-exact, so the same seed gives a slightly different clip. The differences between the two modes are within seed-to-seed variation. The French errors are one or two words per clip (for example "léger" heard as "légère", "tombée" as "tombait") and fall on different seeds in each mode. The one fused clone error is "this" heard as "as" in one seed.
