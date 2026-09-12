# Contributing

## Layout

| path | what |
|---|---|
| `Sources/DecorrelationStretch/` | the library. Pure SPM, no Xcode needed. |
| `Sources/DecorrelationStretch/Shaders/` | the Metal kernels. |
| `App/Decorrelate/` | the macOS app. |
| `project.yml` | XcodeGen spec. **Source of truth** — the `.xcodeproj` is generated and gitignored. |
| `Tools/` | Python parameter-recovery tools. |

```sh
swift build && swift test          # library
xcodegen generate                  # then open Decorrelate.xcodeproj
```

## Things worth knowing before changing them

**The shader and `DSColorSpaceMath.swift` are deliberately duplicated.** The CPU copy
exists because the recovery tools need to work in the base space without a GPU round
trip, and because having two independent implementations is what lets
`testShaderAndCPUConversionsAgree` catch shader bugs. **If you change one, change the
other** — that test will fail if you don't, which is the point.

**Moments are accumulated about a reference centre, not the origin.** This is not
incidental: forming the covariance from origin-centred raw moments is the numerically
weak step in the classic algorithm. Don't "simplify" it away.

**Near-zero eigenvalues are clamped, not pseudo-inverted.** MATLAB's `decorrstretch`
uses the pseudo-inverse, which quietly fails to decorrelate at all. See the note in
`DSSolver.makeTransform`.

**The transform is discarded when the colour-space *family* changes, but kept when only
the weights change.** Both halves matter and both are pinned by tests in
`DSColorSpaceSwitchTests`. Dropping it on weight changes would make every slider drag
flicker through an unfiltered frame; keeping it across a family change applies an
RGB-domain matrix to LAB values.

## Tests

Assert observable behaviour, not internals. The existing suite checks that output is
genuinely decorrelated at the requested variance, that the shader and CPU colour
conversions agree, that sub-sampled analysis renders the same image as a full scan, and
that a rank-deficient image is flagged and still renders finite pixels.

GPU tests call `XCTSkipIf(device == nil)` so the suite still runs without Metal.
