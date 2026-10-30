"""Paths, the upstream checkout and the CAM++ speaker embedding shared by the bench scripts."""

from __future__ import annotations

import os
import subprocess
from math import gcd
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MING_MODEL_DIR", REPO / "models" / "Ming-3.0-tts-0.5b")).resolve()
MING_TTS = Path(os.environ.get("MING_TTS", REPO / "zig-out" / "bin" / "ming-tts")).resolve()

UPSTREAM_URL = "https://github.com/inclusionAI/Ming-3.0-tts"
UPSTREAM_COMMIT = "9332fb9443325d4726ce02cefd4ec1c211df4079"
REF_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"


def upstream_dir() -> Path:
    """The official Ming-3.0-tts code: $MING_UPSTREAM_DIR, or a pinned clone in ~/.cache."""
    if d := os.environ.get("MING_UPSTREAM_DIR"):
        return Path(d).resolve()
    d = Path.home() / ".cache" / "ming-tts-bench" / "Ming-3.0-tts"
    if not (d / "ming_tts.py").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", UPSTREAM_URL, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", UPSTREAM_COMMIT], check=True)
    return d


def ref_wav() -> Path:
    """The 4-second voice-clone reference clip from upstream's voice presets."""
    return upstream_dir() / "data" / "voice_presets" / "voice_audio" / "lingguang.wav"


def load_mono(path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return wav.mean(axis=1), int(sr)


class CamPlus:
    """CAM++ speaker embedding: `campplus.onnx` from the model dir on onnxruntime CPU.

    Features follow upstream `spkemb_extractor.py` (80-bin Kaldi fbank, no dither,
    mean-normalized). Resampling to 16 kHz uses scipy's polyphase filter instead of
    torchaudio's, so embeddings differ from upstream's in the last digits.
    """

    def __init__(self, onnx_path=MODEL_DIR / "campplus.onnx"):
        self.session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Return a (192,) float32 embedding for a mono waveform."""
        if sr != 16000:
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(16000, wav.tolist())
        fbank.input_finished()
        feat = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        feat = feat - feat.mean(axis=0, keepdims=True)
        return self.session.run(None, {self.input_name: feat[None].astype(np.float32)})[0].reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
