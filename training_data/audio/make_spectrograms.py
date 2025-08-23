#!/usr/bin/env python3
"""
Generate spectrogram PNGs for all .wav files in the current directory.

- Reads the first N samples (default: 150000). If a file has fewer samples, uses all of them.
- Handles mono or multi-channel audio (averages channels to mono).
- Saves one PNG per file into the output folder (default: ./spectrograms).

Usage:
    python make_spectrograms.py
    # or customize:
    python make_spectrograms.py --nsamples 120000 --out my_specs --nperseg 2048 --noverlap 1024
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt


def to_mono(x):
    """Average channels if multi-channel."""
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def to_float01(x):
    """Convert integer PCM to float in [-1, 1]. Leave float as-is."""
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32, copy=False)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        denom = max(abs(info.min), info.max)
        if denom == 0:
            return x.astype(np.float32)
        return (x.astype(np.float32) / float(denom))
    # Fallback
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=150000, help="Max samples per file")
    ap.add_argument("--out", type=str, default="spectrograms", help="Output folder")
    ap.add_argument("--nperseg", type=int, default=1024, help="STFT window length")
    ap.add_argument("--noverlap", type=int, default=512, help="STFT overlap (must be < nperseg)")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(Path(".").glob("*.wav"))
    if not wav_paths:
        print("No .wav files found in the current directory.", file=sys.stderr)
        sys.exit(1)

    for p in wav_paths:
        try:
            sr, data = wavfile.read(p)
        except Exception as e:
            print(f"[skip] {p.name}: failed to read ({e})", file=sys.stderr)
            continue

        data = to_mono(np.asarray(data))
        data = to_float01(data)

        n = min(len(data), max(1, args.nsamples))
        if n < 2:
            print(f"[skip] {p.name}: not enough samples ({n})", file=sys.stderr)
            continue

        nperseg = max(2, min(args.nperseg, n))
        noverlap = min(args.noverlap, nperseg - 1)

        # Compute spectrogram (power spectral density), then convert to dB
        f, t, Sxx = spectrogram(
            data[:n],
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

        # Plot (single figure, no explicit color settings)
        plt.figure(figsize=(8, 4), dpi=150)
        plt.pcolormesh(t, f, Sxx_db, shading="auto")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.title(f"Spectrogram: {p.name} (first {n} samples)")
        cbar = plt.colorbar()
        cbar.set_label("Power/Frequency [dB/Hz]")
        plt.tight_layout()

        out_path = outdir / f"{p.stem}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[ok] {p.name} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
