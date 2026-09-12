# DecorrelationStretch

[![CI](https://github.com/nburns/DecorrelationStretch/actions/workflows/ci.yml/badge.svg)](https://github.com/nburns/DecorrelationStretch/actions/workflows/ci.yml)
[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9%2B-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/platforms-macOS%2013%2B%20%7C%20iOS%2016%2B-lightgrey.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A real-time decorrelation stretch filter for macOS and iOS, in Metal.

Decorrelation stretch removes the correlation between colour channels, then re-inflates
each decorrelated axis to a target variance. It is not a contrast stretch — it remaps
the colours themselves, so near-identical hues separate dramatically. Developed at NASA
JPL in 1978 for satellite imagery, and the basis of DStretch, the standard tool for
reading faded rock art.

## Why it runs in real time

The expensive part — covariance and eigendecomposition — produces **one 3×3 matrix and a
3-vector offset**. Everything after that is a single affine multiply per pixel. So the
work splits into two passes that run at completely different cadences:

```
┌─ analysis: every N frames ───────────────────────────────────┐
│  pixels ──► working space ──► Σ moments ──► eigh ──► 3×3 + c │
└──────────────────────────────────────────────────────────────┘
                                                        │
┌─ apply: every frame ───────────────────────────────────▼─────┐
│  RGB ──► working space ──► M·w + c ──► back to RGB ──► clamp │
└──────────────────────────────────────────────────────────────┘
```

Measured at 3840×2160 on an Apple M2 Pro:

| pass | time |
|---|---|
| apply, RGB or YUV space | 0.47 ms |
| apply, LAB space | 1.34 ms |
| analysis, every pixel | 1.35 ms |
| analysis, stride 4 | 0.18 ms |

At the default 12-frame analysis interval that is **~1.35 ms per frame at 4K**, about 8%
of a 60fps budget. LAB costs roughly 3× RGB because of the `cbrt`/`pow` in both
directions; use a YUV-family space if you need the headroom.

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/nburns/DecorrelationStretch", from: "1.0.0")
]
```

Or in Xcode: **File → Add Package Dependencies** and paste the repository URL.

## Stills

```swift
import DecorrelationStretch

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let engine = try DSEngine(device: device)

var config = DSConfiguration()
config.colorSpace = .redEmphasis
config.target = .uniform(fraction: 0.0588)   // DStretch's "scale 15", in 0...255 units
engine.configuration = config

let stretched = try engine.process(image: sourceCGImage, device: device, queue: queue)
```

## Live camera

`encode` never blocks. It applies the most recent completed analysis and schedules a new
one when due, so the render loop never waits on a GPU readback.

```swift
let engine = try DSEngine(device: device)
let textures = try DSTextureCache(device: device)

// AVCaptureVideoDataOutputSampleBufferDelegate
func captureOutput(_ output: AVCaptureOutput,
                   didOutput sampleBuffer: CMSampleBuffer,
                   from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
          let drawable = metalLayer.nextDrawable(),
          let commandBuffer = queue.makeCommandBuffer() else { return }

    let source = try! textures.texture(from: pixelBuffer)
    try! engine.encode(source: source, destination: drawable.texture, in: commandBuffer)

    commandBuffer.addCompletedHandler { _ in textures.endFrame() }
    commandBuffer.present(drawable)
    commandBuffer.commit()
}
```

`DSTextureCache` wraps capture buffers zero-copy via `CVMetalTextureCache`. The drawable
texture needs `.shaderWrite` usage — set `metalLayer.framebufferOnly = false`.

## Configuration

| field | default | what it does |
|---|---|---|
| `colorSpace` | `.chromaBoost` | base space and axis weights the covariance is measured in — the biggest lever on the result |
| `target` | `.uniform(fraction: 0.0588)` | `.uniform` equalises every axis (the dramatic look); `.preserveOriginal` decorrelates without changing spread (subtle) |
| `amount` | `1` | blend against the original, 0...1 |
| `regionOfInterest` | `nil` | measure statistics inside this rect, stretch the whole frame |
| `samplingStride` | `1` | sample every Nth pixel during analysis |
| `analysisInterval` | `12` | frames between re-analysis in live mode |
| `temporalSmoothing` | `0.85` | damps matrix changes so colours don't pulse as a camera pans |

**`regionOfInterest` matters more than it looks.** Sky, foliage, and shadow otherwise
consume the variance budget. Restricting statistics to the panel and stretching the whole
frame is DStretch's selection feature, and it is often the difference between a useful
result and mud.

### Colour spaces

`DSColorSpace` is a base family (`.rgb`, `.yuv`, `.lab`) plus three axis weights, applied
before the covariance is measured and undone after. Scaling an axis changes the
covariance, which changes the eigenvectors, which changes the rotation the stretch
happens in — so the weights genuinely alter the output rather than merely rescaling it.

This mirrors DStretch's `YXX`/`LXX` custom-colorspace panel. The presets below are
starting points, **not** Harman's coefficients for `YDS`/`LRE`/`YBK` — those are not
public. Expect to dial in your own.

| preset | base | for |
|---|---|---|
| `.chromaBoost` | YUV | general purpose; luminance held back so the stretch spends its range on chroma |
| `.redEmphasis` | LAB | red and ochre pigment (weights a*) |
| `.yellowEmphasis` | LAB | faint yellows (weights b*) |
| `.tonalEmphasis` | LAB | dark pigment on dark rock, where the signal is tonal |
| `.rgb` / `.yuv` / `.lab` | — | unweighted baselines |

## The maths

Vectorise to `A` (p×n), mean `µ`, covariance `V`. Spectrally factorise `V = QΛQᵀ`, then

```
M = Σ_T · Q · Λ^(−1/2) · Qᵀ          S = M(Y − µuᵀ) + µ_T uᵀ
```

`QΛ^(−1/2)Qᵀ` is the inverse matrix square root of the covariance, so decorrelation
stretch is **ZCA whitening followed by a rescale** — which is why the whole thing
collapses to one affine map.

Implementation notes that matter:

- **Moments are accumulated about a reference centre**, not the origin. Forming the
  covariance from origin-centred raw moments is the numerically weak step in the classic
  algorithm — `Σx²` and `nµ²` are nearly equal and the subtraction cancels most of the
  significant digits. The centre tracks the measured mean frame to frame. It is exact
  bookkeeping, not an approximation.
- **Eigendecomposition is cyclic Jacobi in Double**, not a closed-form cubic solve, which
  stays accurate when eigenvalues are clustered — the normal case here, since highly
  correlated channels are exactly what this filter is for.
- **Near-zero eigenvalues are clamped, not pseudo-inverted.** A constant or linearly
  dependent channel makes `Λ^(−1/2)` diverge. MATLAB's `decorrstretch` falls back to the
  pseudo-inverse, which quietly fails to decorrelate at all (Crabu, Pes & Rodriguez 2025,
  §3). Clamping keeps the transform bounded and sets `DSTransform.wasRegularized` so the
  caller knows the result is degraded. Dropping the dependent plane would be better for
  offline work, but it changes the output rank mid-stream, which a live path cannot take.

## Caveats

- **Output is false colour.** Colours can be radically different from the original. DS
  results are not evidence of pigment colour.
- **It amplifies noise indiscriminately.** If scene variation is dominated by sensor
  noise or JPEG artefacts, the output will be colourful and unreadable. Feed it the best
  source you have.
- **Everything is assumed sRGB.** `DSImageIO` forces the draw through sRGB rather than
  trusting the image's profile, because the LAB conversion assumes sRGB primaries and
  transfer. Display P3 data passed in unconverted would skew the covariance.
- **Sub-sampling can alias periodic content.** `samplingStride` is safe on photographic
  material; a strongly periodic subject is the one case to leave it at 1.
- Clipping at high stretch is expected and is part of the look.

## Decorrelate (macOS app)

A SwiftUI app with a live preview and controls for every parameter. The Xcode project is
generated by [XcodeGen](https://github.com/yonaskolb/XcodeGen) from `project.yml`, which
is the source of truth — the `.xcodeproj` is gitignored.

```sh
brew install xcodegen
xcodegen generate
open Decorrelate.xcodeproj
```

The app consumes this package as a local SPM dependency, so the library still builds and
tests on its own with `swift build` / `swift test`.

Open an image, drag one in, or switch to Camera. **Drag on the preview** to restrict the
statistics to part of the frame. Presets, base space, per-axis weights, stretch amount,
blend, sampling stride, re-analysis interval, and smoothing are all live; the diagnostics
panel shows eigenvalues, the condition number, pixels sampled, and GPU frame time.

It also runs headless, which is handy for batch work and for checking a change without
clicking through the UI:

```sh
Decorrelate --process in.png out.png --space RedEmphasis --stretch 0.06
Decorrelate --open panel.png
```

`Samples/faded-panel.png` is a synthetic weathered panel whose pigment peaks at **4.8/255**
— invisible by eye, obvious after the stretch.

### Why the app is not a bare SPM executable

Camera access needs TCC, which needs a real `.app` bundle with `NSCameraUsageDescription`
and a camera entitlement. `swift build` emits a bare Mach-O with neither, so the capture
session is denied without ever prompting.

App Sandbox is deliberately **off**. It would block the `--process` file paths and force
security-scoped bookmarks for drag-and-drop, and for a local research tool it buys little.
Hardened runtime stays on, which is what makes `com.apple.security.device.camera`
required. Turn the sandbox back on in `project.yml` before distributing.

## Recovering parameters from a reference result (`Tools/`)

If you have a before/after pair produced by another implementation, `Tools/recover.py`
will tell you which base space and axis weights produced it.

```sh
pip install numpy scipy pillow

python3 Tools/recover.py before.png after.png
python3 Tools/recover.py --split composite.png          # one side-by-side image
python3 Tools/recover.py before.png after.png --diagnose  # when the checks fail
```

It works in two stages, each of which independently reports whether its own assumption
held:

1. **Fit the affine map.** The whole pipeline is affine *in its own working space*, so
   least-squares it from the pixel pairs. Whichever candidate space gives R² near 1
   identifies the base family, and settles whether sRGB was linearised before LAB.
2. **Recover the weights.** `M` is not free — it is fixed by the before-image covariance.
   With the fitted `T`, solve `T = D⁻¹·Σ_T·(D V D)^(−1/2)·D` for `D` and `Σ_T`. The
   `(D, Σ_T)` → `(kD, kΣ_T)` degeneracy means only ratios are identifiable, so `D₀` is
   pinned to 1.

`Tools/selftest.py` validates the whole thing against parameters we already know: it
recovers `weights [1, 2.665, 1.338]` against a true `[1, 2.667, 1.333]`, and the stretch
fraction to within 0.4%. It also correctly picks the covariance variant over the
correlation one (residual 1e-3 vs 6e-2).

**The input has to be a raw stretch.** Same crop, same size, PNG not JPEG, and no
auto-contrast or colour adjustment afterwards. Two health checks tell you when it isn't:

| check | meaning |
|---|---|
| `affine R2` | near 1.0 → the pair really is one global affine map |
| `after \|corr\|` | near 0 → the output is genuinely decorrelated in that space |

Published figures generally fail both, and `--diagnose` tells you which assumption broke.
Three NASA Spinoff side-by-sides were tested: affine R² only 0.35–0.68, and no space
showed the after image decorrelated (best max-correlation 0.38, against ~0.00 for a real
stretch). Downsampling lifted R² to 0.88–0.92 and then plateaued, so part of the mismatch
is recompression and resampling and part is a genuine non-affine stage. Adding an
auto-contrast stage did not help — the optimiser drove it to 0%. Those images are
composites that went through a display pipeline and web processing, and cannot be
inverted. Reference imagery is not redistributable and is not included in this
repository.

## Tests

`swift test` — 17 tests. The suite checks observable behaviour: that output is genuinely
decorrelated at the requested variance, that the shader and CPU colour conversions agree,
that sub-sampled analysis renders the same image as a full scan, that a rank-deficient
image is flagged and still renders finite pixels, and that the live path converges to the
same matrix as the synchronous one.

## Credits

The algorithm is not mine. This is an independent implementation, and the people who
actually did the work are:

- **Jim Soha and colleagues at NASA JPL** (1978) proposed applying the Karhunen–Loève
  transform to digital imagery, which is the basis of the technique.
- **Ronald Alley at NASA JPL** replaced the multi-pass formulation with a single matrix
  multiply, making it both faster and more accurate by removing the intermediate images
  that introduced rounding error. He led the ASTER AST06 decorrelation stretch product.
- **Jon Harman** wrote DStretch, the ImageJ plugin that made the technique the standard
  tool for reading faded rock art, and worked out the custom colour spaces that make it
  effective on pigment. The weighted-base-space parameterisation this library exposes
  follows the mechanism he documents. His specific tuned coefficients are his own work
  and are **not** reproduced here — the presets in this library are independent starting
  points.
- **Elisa Crabu, Federica Pes and Giuseppe Rodriguez** (2025) analysed the numerical
  failure modes of the standard algorithm. Their paper is why this implementation clamps
  near-zero eigenvalues instead of using the pseudo-inverse, and why moments are
  accumulated about a reference centre.

## Prior art

Other software implementing this technique, worth knowing about:

- **DStretch** — ImageJ plugin by Jon Harman, plus `iDStretch` and `aDStretch` for mobile.
  The reference implementation for rock art work.
- **Rock Art Enhancer** — BinaryEarth (Anthony Dunk), iOS and macOS, including a live
  video mode.
- **MATLAB** `decorrstretch`, and **ENVI**'s Decorrelation Stretch tool.

## License

MIT — see [LICENSE](LICENSE).

Note that this is a false-colour enhancement tool. Its output is not evidence of pigment
colour, and should not be presented as such in archaeological or scientific work without
saying how it was produced.

## References

- [NASA Spinoff: Technique for Manipulating Satellite Photos Now Reveals Ancient Images](https://spinoff.nasa.gov/Manipulating_Satellite_Photos_Now_Reveals_Ancient_Images)
- [ASTER AST06 Decorrelation Stretch (Ron Alley, JPL)](https://asterweb.jpl.nasa.gov/content/03_data/01_Data_Products/d-stretch.pdf)
- [DStretch algorithm description (Jon Harman)](http://www.dstretch.com/AlgorithmDescription.html)
- [Crabu, Pes & Rodriguez, *Numerical Methods for Decorrelation Stretch*, Mathematics 2025](https://doi.org/10.3390/math13203297)
- [MATLAB `decorrstretch`](https://www.mathworks.com/help/images/ref/decorrstretch.html)
