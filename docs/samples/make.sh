#!/bin/sh
# Regenerates the audio samples in this folder. Run from the repository root
# after building ming-tts and downloading the model (see README.md).
#
#   docs/samples/make.sh
#
# The French sample clones upstream Ming's lingguang.wav voice preset. The
# script clones upstream Ming-3.0-tts into ~/.cache/ming-tts-bench (the same
# pinned checkout bench/ uses) unless MING_UPSTREAM_DIR points at one.
set -eu

BIN=${MING_TTS:-zig-out/bin/ming-tts}
MODEL=${MING_MODEL_DIR:-models/Ming-3.0-tts-0.5b}
UPSTREAM=${MING_UPSTREAM_DIR:-$HOME/.cache/ming-tts-bench/Ming-3.0-tts}
OUT=docs/samples

if [ ! -f "$UPSTREAM/ming_tts.py" ]; then
  git clone -q https://github.com/inclusionAI/Ming-3.0-tts "$UPSTREAM"
  git -C "$UPSTREAM" checkout -q 9332fb9443325d4726ce02cefd4ec1c211df4079
fi
REF="$(cd "$UPSTREAM" && pwd)/data/voice_presets/voice_audio/lingguang.wav"

# sample NAME JSON: synthesize to WAV, then encode to 64 kbit/s AAC.
sample() {
  "$BIN" say --model "$MODEL" --out "$OUT/$1.wav" --json "$2"
  afconvert -f m4af -d aac -b 64000 "$OUT/$1.wav" "$OUT/$1.m4a"
  rm "$OUT/$1.wav"
}

sample en-uk-george '{"task":"IP","voice":"UK_George","seed":0,
  "text":"Hello. This sample was generated on a Mac by ming-tts-serve, with one of the built-in voices of Ming three point oh."}'

sample zh-ling-xiaorou '{"task":"IP","voice":"灵小柔","seed":0,
  "text":"你好，这段语音是在本地的苹果电脑上生成的，使用的是模型自带的音色。"}'

sample fr-lingguang '{"task":"MULTILINGUAL","language":"fr","seed":0,
  "prompt_wav_path":"'"$REF"'","prompt_text":"诶，你今天有空吗？我们能不能聊聊这件事啊？",
  "text":"Bonjour. Cette voix est clonée à partir d’un court extrait en chinois, puis lue en français."}'
