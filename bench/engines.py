"""Median RTF of the official PyTorch code or mlx-audio on the `ming-tts bench` requests.

    uv run --group torch python engines.py torch
    uv run --group mlx-audio python engines.py mlx-audio

Same requests, seeds and output format as `ming-tts bench --runs 3 --ref-wav lingguang.wav`
(src/ming_main.zig): one warm-up request, then each case with seeds 0..runs-1.
RTF = synthesis wall time / audio duration.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
import types

import numpy as np

from common import MODEL_DIR, REF_TEXT, CamPlus, load_mono, ref_wav, upstream_dir

CASES = [
    ("tts-en", ("The garden will remain open until sunset, provided the weather stays clear, "
                "and visitors are asked to leave by the north gate.")),
    ("tts-zh", "雨后的山谷很安静，远处的云慢慢变成了金色，我们沿着小路走回了村子。"),
    ("clone", "I never thought a little machine could sound this much like me."),
]


def default_voice(text: str) -> str:
    """Same rule as `Engine.tts` with no voice: the model has no neutral speaker."""
    return "灵小柔" if any("一" <= c <= "鿿" for c in text) else "UK_George"


class Torch:
    """Upstream `MingTTS` on MPS, patched as little as needed to run off CUDA."""

    def __init__(self):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # col2im in the vocoder's ISTFT
        os.environ.setdefault("TQDM_DISABLE", "1")  # upstream prints a progress bar per fragment
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from loguru import logger

        logger.remove()  # upstream logs every prompt at INFO
        logger.add(sys.stderr, level="WARNING")

        self.torch = torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.name = f"torch-{device}"

        # Upstream hardcodes torch.autocast(device_type='cuda').
        original = torch.autocast

        class Autocast(original):
            def __init__(self, device_type="cuda", *a, **k):
                super().__init__(device if device_type == "cuda" else device_type, *a, **k)

        torch.autocast = Autocast
        sys.path.insert(0, str(upstream_dir()))
        from ming_tts import MingTTS

        self.model = MingTTS(str(MODEL_DIR), device=device, cache_prompts=False)
        # pynini (Chinese FST normalization) has no reliable macOS wheel. The bench
        # texts have no numbers, so an identity normalizer gives the same tokens.
        self.model._normalizer = types.SimpleNamespace(normalize=lambda t: t)
        self.sample_rate = int(self.model.sample_rate)

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)

    def tts(self, text, voice, seed):
        self._seed(seed)
        w = self.model.synthesize(task="IP", text=text, controls={"ip": voice}, use_zero_spk_emb=True)
        return w.squeeze(0).float().numpy()

    def clone(self, text, wav_path, prompt_text, seed):
        self._seed(seed)
        w = self.model.synthesize(task="TTS", text=text, language_id="English", prompt_wav_path=wav_path,
                                  prompt_text=prompt_text, use_spk_emb=True)
        return w.squeeze(0).float().numpy()


class MlxAudio:
    """mlx-audio's own Ming-3.0 pipeline (`Model.generate`), with two changes for parity.

    1. VAE posterior: mlx-audio 0.6.3 samples the clone prompt latent with
       std = exp(0.5 * logvar); upstream (diffusers Oobleck) uses softplus(scale) + 1e-4.
       The patch below uses the upstream formula. It costs the same.
    2. Speaker embedding: mlx-audio does not compute CAM++ for clones. The other
       engines do, so this one computes it too (onnxruntime CPU, inside the timing).
    """

    def __init__(self):
        import mlx.core as mx
        from mlx import nn
        from mlx_audio.tts.utils import load_model

        self.mx = mx
        self.name = "mlx-audio"
        self.model = load_model(MODEL_DIR)
        audio = self.model.audio

        def encode_latent(self, waveform, waveform_length):
            frame_num = mx.ceil(waveform_length / self.config["enc_kwargs"]["input_dim"]).astype(mx.int32)
            if self.patch_size != -1:
                frame_num = mx.ceil(frame_num / self.patch_size).astype(mx.int32)
            mean, scale = mx.split(self.encoder(waveform), 2, axis=-1)
            latent = mean + (nn.softplus(scale) + 1e-4) * mx.random.normal(mean.shape, dtype=mean.dtype)
            return latent, frame_num

        audio.encode_latent = types.MethodType(encode_latent, audio)
        self.campplus = CamPlus()
        self.sample_rate = int(self.model.sample_rate)

    def _run(self, seed, **kw):
        self.mx.random.seed(seed)
        parts = [np.array(r.audio, dtype=np.float32) for r in self.model.generate(**kw)]
        return np.concatenate(parts)

    def tts(self, text, voice, seed):
        return self._run(seed, text=text, voice=voice)

    def clone(self, text, wav_path, prompt_text, seed):
        emb = self.campplus(*load_mono(wav_path))
        return self._run(seed, text=text, ref_audio=str(wav_path), ref_text=prompt_text, spk_emb=emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["torch", "mlx-audio"])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    eng = Torch() if args.engine == "torch" else MlxAudio()
    ref = str(ref_wav())
    eng.tts("Warm up.", "UK_George", 0)
    for name, text in CASES:
        rtfs, durs = [], []
        for seed in range(args.runs):
            t = time.perf_counter()
            if name == "clone":
                w = eng.clone(text, ref, REF_TEXT, seed)
            else:
                w = eng.tts(text, default_voice(text), seed)
            el = time.perf_counter() - t
            dur = w.size / eng.sample_rate
            rtfs.append(el / dur)
            durs.append(dur)
        print(f"{eng.name:<10} {name:<7} rtf={statistics.median(rtfs):.3f}  audio={statistics.median(durs):.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
