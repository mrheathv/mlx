import SwiftUI

struct PromptFormView: View {
    @EnvironmentObject private var generationViewModel: GenerationViewModel
    @EnvironmentObject private var modelManager: ModelManager

    var body: some View {
        Form {
            Section {
                ModelPickerView()
            }

            Section("Prompt") {
                TextEditor(text: $generationViewModel.prompt)
                    .frame(minHeight: 90)
                    .overlay(alignment: .topLeading) {
                        if generationViewModel.prompt.isEmpty {
                            Text("Describe the image you want…")
                                .foregroundStyle(.tertiary)
                                .padding(.top, 8)
                                .padding(.leading, 5)
                                .allowsHitTesting(false)
                        }
                    }
                TextField("Negative prompt (optional)", text: $generationViewModel.negativePrompt)
            }

            Section("Parameters") {
                Stepper(
                    "Steps: \(generationViewModel.parameters.steps)",
                    value: $generationViewModel.parameters.steps,
                    in: 1...(modelManager.selectedModel?.maxRecommendedSteps ?? 50)
                )

                if modelManager.selectedModel?.supportsGuidanceScale ?? true {
                    HStack {
                        Text("Guidance: \(generationViewModel.parameters.guidanceScale, specifier: "%.1f")")
                        Slider(value: $generationViewModel.parameters.guidanceScale, in: 0...15, step: 0.5)
                    }
                }

                Picker("Aspect ratio", selection: aspectRatioBinding) {
                    ForEach(AspectRatioPreset.allCases) { preset in
                        Text(preset.rawValue).tag(preset)
                    }
                }

                Stepper(
                    "Batch size: \(generationViewModel.parameters.batchSize)",
                    value: $generationViewModel.parameters.batchSize,
                    in: 1...8
                )

                HStack {
                    TextField("Seed", value: $generationViewModel.parameters.seed, format: .number)
                        .textFieldStyle(.roundedBorder)
                    Toggle("Lock", isOn: $generationViewModel.parameters.seedLocked)
                        .toggleStyle(.checkbox)
                    Button {
                        generationViewModel.parameters.seed = UInt32.random(in: 0...UInt32.max)
                    } label: {
                        Image(systemName: "shuffle")
                    }
                    .disabled(generationViewModel.parameters.seedLocked)
                    .help("Randomize seed")
                }
            }

            Section {
                if case let .failed(message) = generationViewModel.phase {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                }
                HStack {
                    Button(isGenerating ? "Cancel" : "Generate") {
                        if isGenerating {
                            generationViewModel.cancel()
                        } else {
                            generationViewModel.submit()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!isGenerating && !generationViewModel.canSubmit)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var isGenerating: Bool {
        switch generationViewModel.phase {
        case .loadingModel, .generating, .finishing: return true
        default: return false
        }
    }

    private var aspectRatioBinding: Binding<AspectRatioPreset> {
        Binding(
            get: {
                AspectRatioPreset.allCases.first {
                    let dims = $0.dimensions()
                    return dims.width == generationViewModel.parameters.width && dims.height == generationViewModel.parameters.height
                } ?? .square
            },
            set: { preset in
                let dims = preset.dimensions()
                generationViewModel.parameters.width = dims.width
                generationViewModel.parameters.height = dims.height
            }
        )
    }
}
