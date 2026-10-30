# Ming TTS local API (contract)

`ming-tts serve` implements this contract over HTTP. Embedding apps get the same routes in process through `Service.handle` (see README.md). Change it only in agreement with the clients.

- Base URL: `http://127.0.0.1:8765` (`--host`, `--port` or `MING_PORT` change it).
- All bodies are JSON (`Content-Type: application/json`), UTF-8.
- The server runs on the same machine as its clients, so audio is exchanged as absolute file paths, not uploads.
- Output WAV files (16-bit PCM, mono, 44.1 kHz) go in the output directory: `--output-dir`, else `MING_OUTPUT_DIR`, else `~/Library/Caches/ming-tts`.

## GET /health

```json
{"status": "ok", "engine": "mlx", "model": "Ming-3.0-tts-0.5b", "device": "gpu", "sample_rate": 44100, "loaded": true}
```

`engine` is `mlx` (an embedding app can set its own name through `Config.engine_name`). `loaded` is false while weights load; synthesis routes answer 503 until then. If loading fails, `status` is `error` and `error` holds the message.

## GET /voices

Built-in voices, if the model has any.

```json
{"voices": [{"id": "...", "name": "...", "language": "zh"}]}
```

## POST /tts

Plain TTS, built-in voice, or voice design.

```json
{"text": "Hello world", "voice": null, "instruction": null, "seed": 0}
```

- `voice`: optional id from `/voices`.
- `instruction`: optional voice-design text (for example "a calm, low female voice"). Ignored when `voice` is set.
- `seed`: optional integer.

## POST /clone

Zero-shot voice cloning from a reference clip.

```json
{"text": "Target text", "prompt_wav_path": "/abs/path/ref.wav", "prompt_text": "Transcript of ref.wav", "seed": 0}
```

`prompt_text` is optional if the model does not need it.

## Response for /tts and /clone

```json
{"wav_path": "/Users/me/Library/Caches/ming-tts/20261030-120000-ab12.wav", "sample_rate": 44100, "duration_s": 3.2, "elapsed_s": 1.1, "rtf": 0.34}
```

## GET /capabilities (v2)

Lists what the loaded engine supports, so the UI builds its forms from data, not hard-coded lists.

```json
{
  "tasks": ["TTS", "MULTILINGUAL", "BASIC", "EMOTION", "DIALECT", "IP", "FREEFORM", "VOICE_DESIGN"],
  "languages": {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French", "es": "Spanish"},
  "dialects": ["chuanyu dialect", "cantonese"],
  "emotions": ["高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性"],
  "basic": {"speed": ["慢速", "中速", "快速"], "pitch": ["低", "中", "高"], "volume": ["低", "中", "高"]},
  "paralinguistic_tags": ["[Breathing]", "[Cough]", "[Sigh]", "[Laughter]", "..."],
  "max_prompt_s": 20,
  "max_text_chars": 1000
}
```

Values come from upstream Ming-3.0-tts `tts_tasks.py` and `frontend/toolkit.py` (`PARALINGUISTIC_TAGS`). `max_text_chars` follows `MING_MAX_TEXT_CHARS`.

## POST /generate (v2)

One endpoint for every Ming task. `/tts` and `/clone` stay as shortcuts.

```json
{
  "task": "EMOTION",
  "text": "Target text, may contain paralinguistic tags",
  "language": "ja",
  "prompt_wav_path": "/abs/path/ref.wav",
  "prompt_text": "Transcript of ref.wav",
  "voice": null,
  "instruction": null,
  "controls": {"emotion": "高兴"},
  "seed": 0
}
```

Rules follow upstream `tts_tasks.build_task_generation_kwargs`:

| task | prompt_wav_path | prompt_text | language | controls |
| --- | --- | --- | --- | --- |
| TTS | optional (clone) | optional | zh/en optional | — |
| MULTILINGUAL | required | required | ja/ko/de/fr/es required | — |
| BASIC | required | — | — | speed, pitch, volume |
| EMOTION | required | — | — | emotion |
| DIALECT | required | — | not accepted | dialect |
| IP | — | — | — | `voice` = built-in voice id |
| FREEFORM | — | — | — | `instruction` |
| VOICE_DESIGN | — | — | — | `controls.style` or `instruction` |

Invalid combinations return 400 with the upstream error message. Response shape is the same as `/tts`.

## Errors

HTTP 4xx/5xx with `{"error": "message"}`.
