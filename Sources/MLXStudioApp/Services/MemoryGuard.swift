import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Warns before a generation is likely to exhaust unified memory, rather
/// than letting MLX/Metal thrash or the app get killed by the OS.
///
/// This is deliberately advisory, not a hard gate: the free-memory read is
/// a snapshot (macOS reclaims/compresses pages continuously) and the
/// per-model footprint in the registry is an estimate. `MemoryGuard` is
/// meant to catch the common case ("you have a browser and Xcode open on a
/// 24GB machine and picked a FLUX.1 model") — see ARCHITECTURE.md §4
/// for why this needs verification on real hardware.
struct MemoryGuard {
    struct Assessment {
        let isLikelyToFit: Bool
        let estimatedRequiredGB: Double
        let estimatedAvailableGB: Double
    }

    /// Fixed headroom reserved for macOS + the app's own (non-model) memory.
    /// Conservative on purpose for a fanless machine where swapping/thermal
    /// pressure is worse, not just slower.
    var reservedSystemGB: Double = 4.0

    func assess(model: ModelDescriptor) -> Assessment {
        let availableGB = Self.currentFreeMemoryGB()
        let requiredGB = model.approxRuntimeRAMGB + reservedSystemGB
        return Assessment(
            isLikelyToFit: availableGB >= requiredGB,
            estimatedRequiredGB: requiredGB,
            estimatedAvailableGB: availableGB
        )
    }

    /// Free + inactive (easily reclaimable) pages, in GB. Inactive pages
    /// are counted as "available" because macOS will reclaim them under
    /// pressure before it starts swapping — treating only `free` as
    /// available would under-report and trigger warnings constantly on a
    /// machine that's just been running normally.
    static func currentFreeMemoryGB() -> Double {
        #if canImport(Darwin)
        var pageSize: vm_size_t = 0
        host_page_size(mach_host_self(), &pageSize)

        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, reboundPointer, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            // Fall back to total physical memory as a conservative-ish
            // signal rather than crashing/asserting on a read failure.
            return Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
        }
        let freePages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
        let bytes = freePages * UInt64(pageSize)
        return Double(bytes) / 1_073_741_824.0
        #else
        return Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
        #endif
    }
}
