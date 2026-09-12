"""Recover decorrelation-stretch parameters from a before/after image pair.

    python3 Tools/recover.py before.png after.png
    python3 Tools/recover.py --split composite.png      # one side-by-side image
    python3 Tools/recover.py before.png after.png --diagnose

Prints the recovered base space, axis weights, and target sigmas, plus a ready-to-paste
Swift preset.

The pair must be a RAW stretch: same pixel dimensions, no crop, resize, or recompression
between them, and no auto-contrast or colour adjustment applied afterwards. PNG, not JPEG.
Validated to about 0.3% against known parameters by Tools/selftest.py.

How it works, in two stages, each of which independently reports whether its own
assumption held:

  1. The pipeline is affine *in its own working space*, so fit that affine map by least
     squares. Whichever candidate space gives R2 near 1 identifies the base family, and
     settles whether sRGB was linearised before LAB.

  2. M is not free - it is fixed by the before-image covariance. With the fitted T, solve
     T = D^-1 . Sigma_T . (D V D)^-1/2 . D for D and Sigma_T. The (D, Sigma_T) ->
     (kD, k.Sigma_T) degeneracy means only ratios are identifiable, so D[0] is pinned to 1.

Read the health checks. Published figures usually fail them, and then no number below is
trustworthy - use --diagnose to see why.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from dsfit import (SPACES, box_downsample, fit_affine, max_abs_correlation, recover,
                   split_composite, usable_mask)

SWIFT_FAMILY = {"rgb01": ".rgb", "yuv": ".yuv", "lab": ".lab", "lab_nolin": ".lab"}
NOMINAL = {"rgb01": np.array([1.0, 1.0, 1.0]),
           "yuv": np.array([1.0, 0.872, 1.230]),
           "lab": np.array([100.0, 128.0, 128.0]),
           "lab_nolin": np.array([100.0, 128.0, 128.0])}
CANDIDATES = ["rgb01", "yuv", "lab", "lab_nolin"]


def load_pair(args):
    if args.split:
        before, after, score = split_composite(args.before)
        print(f"split composite: {before.shape[1]}x{before.shape[0]} per half, "
              f"registration peak/sigma {score:.0f}")
        if score < 8:
            print("  WARNING: weak registration - the halves may be scaled, not just offset")
        return before, after
    before = np.asarray(Image.open(args.before).convert("RGB")).astype(np.float64)
    after = np.asarray(Image.open(args.after).convert("RGB")).astype(np.float64)
    if before.shape != after.shape:
        sys.exit(f"shape mismatch: {before.shape[:2]} vs {after.shape[:2]} - "
                 "the pair must be the same crop at the same size")
    return before, after


def diagnose(before, after):
    """When the health checks fail, work out which assumption broke."""
    print("\ndiagnosis")
    print("  affine R2 vs box-downsample factor (climbing then plateauing means part of")
    print("  the mismatch is compression or resampling and part is genuinely non-affine)")
    print("    factor  " + "".join(f"{s:>12s}" for s in CANDIDATES))
    for factor in (1, 2, 4, 8, 16):
        b = box_downsample(before, factor).reshape(-1, 3)
        a = box_downsample(after, factor).reshape(-1, 3)
        ok = usable_mask(b, a)
        if ok.sum() < 2000:
            continue
        row = ""
        for space in CANDIDATES:
            _, _, r2, _ = fit_affine(SPACES[space](b[ok] / 255.0), SPACES[space](a[ok] / 255.0))
            row += f"{r2:12.4f}"
        print(f"    {factor:>6d}  {row}")

    print("  most-decorrelated sub-region of the AFTER image (tests whether statistics")
    print("  came from a selection rather than the whole frame)")
    h, w, _ = after.shape
    for space in CANDIDATES:
        best = (1.0, None)
        for rows in (1, 2, 3):
            for cols in (1, 2, 3):
                for i in range(rows):
                    for j in range(cols):
                        block = after[i * h // rows:(i + 1) * h // rows,
                                      j * w // cols:(j + 1) * w // cols].reshape(-1, 3) / 255.0
                        if len(block) < 2000:
                            continue
                        c = max_abs_correlation(SPACES[space](block))
                        if c < best[0]:
                            best = (c, f"{rows}x{cols}[{i},{j}]")
        print(f"    {space:11s} {best[0]:.3f}  at {best[1]}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("before")
    parser.add_argument("after", nargs="?")
    parser.add_argument("--split", action="store_true",
                        help="treat the single input as one side-by-side composite")
    parser.add_argument("--diagnose", action="store_true",
                        help="if the health checks fail, work out which assumption broke")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()
    if args.help or (not args.split and not args.after):
        sys.exit(__doc__)

    before, after = load_pair(args)

    fb, fa = before.reshape(-1, 3), after.reshape(-1, 3)
    ok = usable_mask(fb, fa)
    print(f"usable pixels: {ok.sum()} / {len(ok)} ({100 * ok.mean():.1f}%) after dropping clipped")
    if ok.sum() < 5000:
        sys.exit("too few unclipped pixels to fit")
    b, a = fb[ok] / 255.0, fa[ok] / 255.0

    print("\nhealth checks")
    print(f"  {'space':11s} {'affine R2':>10s} {'after |corr|':>13s}")
    fits = {}
    for space in CANDIDATES:
        T, c, r2, _ = fit_affine(SPACES[space](b), SPACES[space](a))
        corr = max_abs_correlation(SPACES[space](a))
        fits[space] = (T, r2, corr)
        print(f"  {space:11s} {r2:10.5f} {corr:13.3f}")

    suspect = False
    if max(v[1] for v in fits.values()) < 0.99:
        suspect = True
        print("\n  WARNING: no space fits as a global affine map. The pair has been through")
        print("  extra processing (resize, recompression, auto-contrast, colour adjust).")
    if min(v[2] for v in fits.values()) > 0.15:
        suspect = True
        print("\n  WARNING: the after image is not decorrelated in any candidate space, so it")
        print("  is not a plain decorrelation stretch.")
    if suspect:
        print("  Numbers below are not trustworthy."
              + ("" if args.diagnose else " Re-run with --diagnose to see why."))

    print("\nparameter recovery")
    best = None
    for space, (T, r2, corr) in fits.items():
        V = np.cov(SPACES[space](b), rowvar=False)
        for use_corr in (False, True):
            d, sigma, rel = recover(T, V, use_correlation=use_corr, restarts=12)
            tag = f"{space}/{'corr' if use_corr else 'cov'}"
            print(f"  {tag:16s} weights [{d[0]:.4f} {d[1]:.4f} {d[2]:.4f}]"
                  f"  sigma [{sigma[0]:.3f} {sigma[1]:.3f} {sigma[2]:.3f}]  residual {rel:.3e}")
            if best is None or rel < best[0]:
                best = (rel, space, use_corr, d, sigma)

    rel, space, use_corr, d, sigma = best
    print(f"\nbest model: {space} / {'correlation' if use_corr else 'covariance'} matrix, "
          f"residual {rel:.3e}")
    if rel > 1e-2:
        print("  residual is high - these numbers do not reproduce the fitted map")

    fractions = sigma / (NOMINAL[space] * d)
    print(f"\nas our parameters:  fraction per axis "
          f"{np.array2string(fractions, precision=4)}  (mean {fractions.mean():.4f})")
    print("\nSwift preset:\n")
    print("    static let recovered = DSColorSpace(")
    print(f"        family: {SWIFT_FAMILY[space]},")
    print(f"        weights: SIMD3({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})")
    print("    )")
    print(f"    // use with .uniform(fraction: {fractions.mean():.4f})")

    if args.diagnose:
        diagnose(before, after)


if __name__ == "__main__":
    main()
