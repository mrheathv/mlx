import SwiftUI

struct MemoryWarningView: View {
    let assessment: MemoryGuard.Assessment
    let onCancel: () -> Void
    let onProceed: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("This model may not fit in available memory", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.orange)

            Text(
                "This model needs roughly \(String(format: "%.1f", assessment.estimatedRequiredGB)) GB " +
                "of unified memory (including headroom for macOS), but only about " +
                "\(String(format: "%.1f", assessment.estimatedAvailableGB)) GB currently looks free. " +
                "Generating anyway may cause heavy swapping or the app being terminated by macOS."
            )
            .fixedSize(horizontal: false, vertical: true)

            Text("Try closing other apps first, or continue anyway if you'd like to try.")
                .font(.callout)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel, action: onCancel)
                Button("Try Anyway", action: onProceed)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 420)
    }
}
