"""Validate the fitter against parameters we already know.

Runs our own filter with a known colour space and stretch, feeds the before/after pair
back through the recovery, and checks the numbers come out. If this cannot recover our
own parameters there is no reason to trust it on anyone else's output.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from dsfit import SPACES, fit_affine, usable_mask, recover

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Samples", "_selftest.png")

if not os.path.exists(OUT):
    sys.exit(f"missing {OUT}\n\nGenerate it first with the app's headless mode:\n"
             f"  Decorrelate --process Samples/faded-panel.png Samples/_selftest.png "
             f"--space RedEmphasis --stretch 0.06")

before = np.asarray(Image.open(f"{ROOT}/Samples/faded-panel.png").convert("RGB")).reshape(-1, 3).astype(np.float64)
after = np.asarray(Image.open(OUT).convert("RGB")).reshape(-1, 3).astype(np.float64)

ok = usable_mask(before, after)
print(f"usable pixels: {ok.sum()} / {len(ok)}  ({100 * ok.mean():.1f}%)\n")
b, a = before[ok] / 255.0, after[ok] / 255.0

print("stage 1 - which space is the pipeline affine in?")
fits = {}
for name, fn in SPACES.items():
    T, c, r2, rmse = fit_affine(fn(b), fn(a))
    fits[name] = (T, c)
    print(f"  {name:11s} R2 = {r2:.6f}   rmse = {rmse:.4g}")

T, c = fits["lab"]
V = np.cov(SPACES["lab"](b), rowvar=False)
print("\nstage 2 - recover weights and target sigmas (LAB)")
for corr in (False, True):
    d, sigma, rel = recover(T, V, use_correlation=corr)
    label = "correlation" if corr else "covariance"
    print(f"  {label:11s} d = [{d[0]:.4f} {d[1]:.4f} {d[2]:.4f}]"
          f"  sigma = [{sigma[0]:.3f} {sigma[1]:.3f} {sigma[2]:.3f}]  residual = {rel:.3e}")

# RedEmphasis is lab(0.6, 1.6, 0.8) at fraction 0.06, and nominalScale already folds the
# weights in. The (D, Sigma_T) -> (kD, k.Sigma_T) degeneracy means only ratios survive,
# so rescale the truth to d[0] = 1 the same way the fitter does.
k = 1 / 0.6
print(f"\n  ground truth  d = [1.0000 {1.6 * k:.4f} {0.8 * k:.4f}]"
      f"  sigma = [{0.06 * 100 * 0.6 * k:.3f} {0.06 * 128 * 1.6 * k:.3f} {0.06 * 128 * 0.8 * k:.3f}]")
