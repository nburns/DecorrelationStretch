"""Recover decorrelation-stretch parameters from a before/after image pair.

Two stages, each of which independently reports whether its assumption held:

  Stage 1  The whole pipeline is affine in its own working space. Fit that affine map
           by least squares. Whichever candidate space yields a near-zero residual
           identifies the base family (and settles whether sRGB was linearised first).

  Stage 2  M is not free: it is determined by the before-image covariance. Given the
           fitted T, solve for the axis weights D and target sigmas.
             T = D^-1 . Sigma_T . (D V D)^-1/2 . D
           (D, Sigma_T) and (kD, k.Sigma_T) give identical T, so D[0] is pinned to 1.
"""
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------- colour spaces

D65 = np.array([0.95047, 1.0, 1.08883])
M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
M_YUV = np.array([[ 0.29900,  0.58700,  0.11400],
                  [-0.14713, -0.28886,  0.43600],
                  [ 0.61500, -0.51499, -0.10001]])


def srgb_to_linear(c):
    return np.where(c > 0.04045, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4, c / 12.92)


def to_lab(rgb01, linearise=True):
    lin = srgb_to_linear(rgb01) if linearise else rgb01
    xyz = lin @ M_RGB2XYZ.T / D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0)), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


SPACES = {
    "rgb01":      lambda p: p,
    "rgb255":     lambda p: p * 255.0,
    "yuv":        lambda p: p @ M_YUV.T,
    "lab":        lambda p: to_lab(p, linearise=True),
    "lab_nolin":  lambda p: to_lab(p, linearise=False),
}

# ---------------------------------------------------------------- stage 1

def fit_affine(X, Y):
    """Least-squares Y ~= X @ T.T + c."""
    A = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    T, c = sol[:3].T, sol[3]
    residual = Y - (X @ T.T + c)
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return T, c, 1 - ss_res / ss_tot, float(np.sqrt((residual ** 2).mean()))


def usable_mask(before_u8, after_u8, guard=3):
    """Clipped output pixels break the affine relation outright and must be dropped."""
    lo, hi = guard, 255 - guard
    ok = ((after_u8 > lo) & (after_u8 < hi)).all(axis=1)
    ok &= ((before_u8 > 0) & (before_u8 < 255)).all(axis=1)
    return ok

# ---------------------------------------------------------------- stage 2

def inv_sqrt(V, floor=1e-12):
    w, Q = np.linalg.eigh(V)
    w = np.maximum(w, w.max() * floor)
    return Q @ np.diag(w ** -0.5) @ Q.T


def predict_T(d, sigma, V, use_correlation=False):
    D = np.diag(d)
    Vw = D @ V @ D
    if use_correlation:
        s = np.sqrt(np.diag(Vw))
        C = Vw / np.outer(s, s)
        core = inv_sqrt(C) @ np.diag(1 / s)
    else:
        core = inv_sqrt(Vw)
    return np.linalg.inv(D) @ np.diag(sigma) @ core @ D


def recover(T, V, use_correlation=False, restarts=24, seed=0):
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt(np.trace(V) / 3))
    best = None
    for i in range(restarts):
        if i == 0:
            p0 = np.array([1.0, 1.0, np.log(scale), np.log(scale), np.log(scale)])
        else:
            p0 = np.concatenate([rng.uniform(0.2, 3.0, 2),
                                 np.log(scale) + rng.uniform(-2, 2, 3)])

        def cost(p):
            d = np.array([1.0, abs(p[0]) + 1e-6, abs(p[1]) + 1e-6])
            sigma = np.exp(np.clip(p[2:5], -20, 20))
            try:
                return float(((predict_T(d, sigma, V, use_correlation) - T) ** 2).sum())
            except np.linalg.LinAlgError:
                return 1e12

        r = minimize(cost, p0, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r

    d = np.array([1.0, abs(best.x[0]), abs(best.x[1])])
    sigma = np.exp(best.x[2:5])
    rel = float(np.sqrt(best.fun) / (np.linalg.norm(T) + 1e-30))
    return d, sigma, rel


# ---------------------------------------------------------------- side-by-side splitting

def find_seam(a):
    """Column range of a divider in a side-by-side composite, if there is one."""
    h, w, _ = a.shape
    column_spread = a.std(axis=(0, 2))
    centre = w // 2
    window = slice(max(0, centre - w // 8), min(w, centre + w // 8))
    quiet = np.where(column_spread[window] < 2.0)[0] + window.start
    if len(quiet) == 0:
        return centre, centre
    breaks = np.where(np.diff(quiet) > 1)[0]
    runs = np.split(quiet, breaks + 1)
    run = min(runs, key=lambda r: abs(r.mean() - centre))
    return int(run[0]), int(run[-1]) + 1


def _gradient_magnitude(rgb):
    g = rgb.mean(axis=2)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def register(left, right):
    """Integer (dy, dx) aligning `right` onto `left`, plus a confidence score.

    Correlates gradient magnitude rather than colour: the whole point of a before/after
    pair is that the colours differ wildly while the structure does not.
    """
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    a = _gradient_magnitude(left[:h, :w].astype(np.float64))
    b = _gradient_magnitude(right[:h, :w].astype(np.float64))
    a -= a.mean()
    b -= b.mean()
    spectrum = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    spectrum /= np.abs(spectrum) + 1e-9
    correlation = np.fft.irfft2(spectrum, s=(h, w))
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy = peak[0] if peak[0] <= h // 2 else peak[0] - h
    dx = peak[1] if peak[1] <= w // 2 else peak[1] - w
    score = float(correlation.max() / (correlation.std() + 1e-12))
    return int(dy), int(dx), score, h, w


def split_composite(path):
    """Split a side-by-side before/after composite and register the halves."""
    from PIL import Image
    image = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    start, end = find_seam(image)
    left, right = image[:, :start], image[:, end:]
    dy, dx, score, h, w = register(left, right)
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    return left[y0:y1, x0:x1], right[y0 - dy:y1 - dy, x0 - dx:x1 - dx], score


def box_downsample(a, factor):
    """Average over factor x factor blocks, to wash out compression artefacts."""
    if factor == 1:
        return a
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    return a[:h, :w].reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3))


def max_abs_correlation(pixels):
    """Largest absolute channel-pair correlation. Near zero means decorrelated."""
    V = np.cov(pixels, rowvar=False)
    s = np.sqrt(np.diag(V))
    if (s <= 1e-9).any():
        return 1.0
    C = V / np.outer(s, s)
    return max(abs(C[0, 1]), abs(C[0, 2]), abs(C[1, 2]))
