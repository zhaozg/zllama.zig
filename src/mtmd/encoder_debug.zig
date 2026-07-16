//! Shared encoder debug utilities
//!
//! Extracts the common markDebugOutputs / saveDebugTensors patterns
//! that are duplicated between VisionEncoder and AudioEncoder.

const std = @import("std");
const ggml = @import("ggml");
const debug_mod = @import("debug");

/// A single debug tensor entry: tensor name in graph → output filename.
pub const DebugTensorEntry = struct {
    /// Name set via ggml_set_name() in the graph
    tensor_name: []const u8,
    /// Output filename (relative to subdir)
    filename: []const u8,
};

/// Mark intermediate tensors with ggml.setOutput() so their data
/// is preserved after graph computation.
///
/// Call this BEFORE Gallocr.allocGraph or ggml_backend_graph_compute.
///
/// `log` is the scoped logger returned by `std.log.scoped(...)`.
/// In Zig 0.16.0, `std.log.scoped()` returns an anonymous struct type,
/// so we use `anytype` to accept any scoped logger.
pub fn markDebugOutputs(
    cgraph: *ggml.CGraph,
    entries: []const DebugTensorEntry,
    log: anytype,
) void {
    for (entries) |entry| {
        debug_mod.markTensorAsOutput(cgraph, entry.tensor_name) catch |err| {
            log.warn("markDebugOutputs: failed to mark '{s}': {}", .{ entry.tensor_name, err });
        };
    }
}

/// Save intermediate tensors from a computed graph to JSON files.
///
/// Uses the debug module's saveTensorFromGraph to write each entry.
/// Errors are logged but do not propagate (best-effort save).
///
/// `log` is the scoped logger returned by `std.log.scoped(...)`.
pub fn saveDebugTensors(
    io: std.Io,
    allocator: std.mem.Allocator,
    subdir: []const u8,
    entries: []const DebugTensorEntry,
    cgraph: *ggml.CGraph,
    log: anytype,
) void {
    for (entries) |entry| {
        debug_mod.saveTensorFromGraph(io, allocator, subdir, entry.filename, entry.tensor_name, cgraph) catch |err| {
            log.warn("saveDebugData: failed to save '{s}': {}", .{ entry.filename, err });
        };
    }
}
