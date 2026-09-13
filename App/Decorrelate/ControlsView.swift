import SwiftUI
import DecorrelationStretch

struct ControlsView: View {
    @ObservedObject var model: FilterModel
    var onOpenImage: () -> Void
    var onExport: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                sourceSection
                Divider()
                colorSpaceSection
                Divider()
                stretchSection
                Divider()
                analysisSection
                Divider()
                diagnosticsSection
            }
            .padding(16)
        }
        #if os(macOS)
        .frame(width: 300)
        #else
        // In the iPhone sheet the controls should fill the width; a fixed 300pt would
        // leave a ragged margin that shifts with device size.
        .frame(maxWidth: .infinity)
        #endif
    }

    // MARK: Source

    private var sourceSection: some View {
        section("Source") {
            Picker("", selection: $model.sourceMode) {
                ForEach(SourceMode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            HStack {
                Button("Open Image…", action: onOpenImage)
                Spacer()
                Button("Export…", action: onExport)
                    .disabled(model.sourceSize == .zero)
            }
            Text(model.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Colour space

    private var colorSpaceSection: some View {
        section("Colour Space") {
            Menu("Presets") {
                ForEach(Preset.all) { preset in
                    Button(preset.name) { model.apply(preset: preset) }
                }
            }

            // Reset weights only when the user changes the base directly: a preset sets
            // family and weights together, and an onChange hook would wipe the weights it
            // had just applied.
            Picker("Base", selection: Binding(
                get: { model.family },
                set: { newValue in
                    guard newValue != model.family else { return }
                    model.family = newValue
                    model.resetWeights()
                }
            )) {
                Text("RGB").tag(DSColorSpaceFamily.rgb)
                Text("YUV").tag(DSColorSpaceFamily.yuv)
                Text("LAB").tag(DSColorSpaceFamily.lab)
            }
            .pickerStyle(.segmented)

            let labels = model.axisLabels
            slider(labels.0, value: $model.weightX, range: 0.1...3, format: "%.2f")
            slider(labels.1, value: $model.weightY, range: 0.1...3, format: "%.2f")
            slider(labels.2, value: $model.weightZ, range: 0.1...3, format: "%.2f")

            HStack {
                Button("Reset Weights") { model.resetWeights() }
                    .controlSize(.small)
                Spacer()
            }

            Text("Weights scale each axis before the covariance is measured, which changes "
                 + "the rotation the stretch happens in. These are starting points; expect "
                 + "to tune them for your own material.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Stretch

    private var stretchSection: some View {
        section("Stretch") {
            Toggle("Equalise all axes", isOn: $model.useUniformTarget)
                .help("Off: decorrelate without changing each axis's spread. Subtler, and "
                      + "closer to how MATLAB's decorrstretch behaves by default.")

            if model.useUniformTarget {
                slider("Amount", value: $model.stretchFraction, range: 0.01...0.30, format: "%.3f")
                Text("Target sigma as a fraction of the space's range. 0.059 is equivalent "
                     + "to a sigma of 15 in 0-255 units.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            slider("Blend", value: $model.amount, range: 0...1, format: "%.2f")
        }
    }

    // MARK: Analysis

    private var analysisSection: some View {
        section("Analysis") {
            HStack {
                Text("Region")
                Spacer()
                if let roi = model.regionOfInterest {
                    Text("\(Int(roi.width))×\(Int(roi.height))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Button("Clear") { model.regionOfInterest = nil }
                        .controlSize(.small)
                } else {
                    Text("Whole frame").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Drag on the preview to measure statistics from part of the frame only. "
                 + "Excluding sky or foliage stops them consuming the variance budget.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Stepper("Sampling stride: \(model.samplingStride)",
                    value: $model.samplingStride, in: 1...8)
            Stepper("Re-analyse every \(model.analysisInterval) frames",
                    value: $model.analysisInterval, in: 1...60)
                .disabled(model.sourceMode == .image)
            slider("Smoothing", value: $model.temporalSmoothing, range: 0...0.98, format: "%.2f")
                .disabled(model.sourceMode == .image)
        }
    }

    // MARK: Diagnostics

    private var diagnosticsSection: some View {
        section("Diagnostics") {
            if model.wasRegularized {
                Label("Colour planes are nearly dependent — one axis carries no signal and "
                      + "the stretch along it is amplifying noise.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            readout("Eigenvalues", String(format: "%.3g  %.3g  %.3g",
                                          model.eigenvalues.x, model.eigenvalues.y, model.eigenvalues.z))
            readout("Condition", String(format: "%.0f", model.conditionNumber))
            readout("Pixels sampled", "\(model.sampleCount)")
            readout("GPU frame", String(format: "%.2f ms", model.frameMilliseconds))
            if model.sourceSize != .zero {
                readout("Source", "\(Int(model.sourceSize.width))×\(Int(model.sourceSize.height))")
            }
        }
    }

    // MARK: Building blocks

    private func section<Content: View>(_ title: String,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
    }

    private func slider(_ label: String, value: Binding<Float>,
                        range: ClosedRange<Float>, format: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }

    private func readout(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospacedDigit())
        }
    }
}
