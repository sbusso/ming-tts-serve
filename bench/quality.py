"""Speech quality of the fused kernels (MING_FUSED=1) against the plain MLX graph (MING_FUSED=0).

    uv run --group quality python quality.py [--seeds 5]

For each mode, starts `ming-tts serve`, generates every case with seeds 0..N-1, then:
- transcribes each clip with Whisper (mlx-whisper) and scores it against the input text:
  word error rate for English and French, character error rate for Chinese;
- for cases with a reference clip, computes CAM++ speaker similarity to that clip;
- computes CAM++ similarity between the plain and fused clip of the same case and seed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path

import jiwer
import mlx_whisper

from common import MING_TTS, MODEL_DIR, REF_TEXT, CamPlus, cosine, load_mono, ref_wav

ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
PORT = 8799


def cases(ref: str) -> dict[str, tuple[str, dict]]:
    """name -> (Whisper language, /generate body without seed)."""
    clone = {"prompt_wav_path": ref, "prompt_text": REF_TEXT}
    return {
        "en": ("en", {"task": "IP", "voice": "UK_George",
                      "text": "Please remember to bring an umbrella, because the forecast says it will rain all afternoon."}),
        "zh": ("zh", {"task": "IP", "voice": "灵小柔",
                      "text": "今天的会议改到下午举行，请大家提前准备好材料，并在门口签到。"}),
        "fr": ("fr", {"task": "MULTILINGUAL", "language": "fr", **clone,
                      "text": "Le train pour Lyon partira avec un léger retard à cause de la neige tombée cette nuit."}),
        "clone": ("en", {"task": "TTS", **clone,
                         "text": "I never thought a little machine could sound this much like me."}),
    }


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def generate_all(fused: str, out_dir: Path, todo: dict, seeds: int) -> dict:
    """Run one `ming-tts serve` with MING_FUSED=fused; return {(case, seed): wav_path}."""
    env = {**os.environ, "MING_FUSED": fused}
    srv = subprocess.Popen([str(MING_TTS), "serve", "--model", str(MODEL_DIR), "--port", str(PORT),
                            "--output-dir", str(out_dir)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if json.load(r).get("loaded"):
                        break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("ming-tts serve did not load the model")
        return {(name, s): post("/generate", {**body, "seed": s})["wav_path"]
                for name, (_, body) in todo.items() for s in range(seeds)}
    finally:
        srv.terminate()
        srv.wait()


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces."""
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.lower())
    return " ".join(text.split())


def error_rate(lang: str, ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    if lang == "zh":
        return jiwer.cer(ref.replace(" ", ""), hyp.replace(" ", ""))
    return jiwer.wer(ref, hyp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    ref = str(ref_wav())
    todo = cases(ref)
    camplus = CamPlus()
    ref_emb = camplus(*load_mono(ref))
    modes = {"plain": "0", "fused": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        wavs = {m: generate_all(f, Path(tmp) / m, todo, args.seeds) for m, f in modes.items()}
        rows = {}
        for name, s in wavs["plain"]:
            lang, body = todo[name]
            embs = {}
            for m in modes:
                path = wavs[m][(name, s)]
                hyp = mlx_whisper.transcribe(path, path_or_hf_repo=ASR_MODEL, language=lang)["text"]
                embs[m] = camplus(*load_mono(path))
                r = rows.setdefault((name, m), {"err": [], "sim_ref": [], "sim_pf": []})
                r["err"].append(error_rate(lang, body["text"], hyp))
                if "prompt_wav_path" in body:
                    r["sim_ref"].append(cosine(embs[m], ref_emb))
                print(f"{name:<6} {m:<6} seed={s}  err={r['err'][-1]:.3f}  {hyp.strip()}", flush=True)
            rows[(name, "fused")]["sim_pf"].append(cosine(embs["plain"], embs["fused"]))

    def mean(xs):
        return f"{statistics.mean(xs):.3f}" if xs else "-"

    print(f"\n{args.seeds} seeds per case. err: WER (en, fr) or CER (zh). sim: CAM++ cosine similarity.")
    print("| Case | err, plain | err, fused | sim to ref, plain | sim to ref, fused | sim plain vs. fused |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name in todo:
        p, f = rows[(name, "plain")], rows[(name, "fused")]
        print(f"| {name} | {mean(p['err'])} | {mean(f['err'])} | {mean(p['sim_ref'])} | {mean(f['sim_ref'])} "
              f"| {mean(f['sim_pf'])} |")
    allv = {(m, k): [x for (_, mm), r in rows.items() if mm == m for x in r[k]]
            for m in modes for k in ("err", "sim_ref", "sim_pf")}
    print(f"| all | {mean(allv['plain', 'err'])} | {mean(allv['fused', 'err'])} | {mean(allv['plain', 'sim_ref'])} "
          f"| {mean(allv['fused', 'sim_ref'])} | {mean(allv['fused', 'sim_pf'])} |")


if __name__ == "__main__":
    main()
