//! Shared encoder debug utilities
//!
//! Extracts the common saveDebugTensors patterns
//! that are duplicated between VisionEncoder and AudioEncoder.

const std = @import("std");
const ggml = @import("ggml");
const debug_mod = @import("debug");

/// A single debug tensor entry: tensor name in graph → output filename.
/// `is_input` indicates whether this tensor is an input tensor (data set via setDataPtr)
/// or an output tensor (computed by the graph). Input tensors should not be marked
/// with setOutput, and their data type may differ from f32.
pub const DebugTensorEntry = struct {
    /// Name set via ggml_set_name() in the graph
    tensor_name: []const u8,
    /// Output filename (relative to subdir)
    filename: []const u8,
    /// Whether this is an input tensor (true) or output tensor (false, default)
    is_input: bool = false,
};

/// Save intermediate tensors from a computed graph to JSON files.
///
/// Uses the debug module's saveTensorFromGraph to write each entry.
/// For input tensors (is_input=true), reads data directly from the tensor's data pointer
/// using the correct element type. For output tensors, reads f32 data from the graph.
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
        if (entry.is_input) {
            // For input tensors, read data directly from the tensor's data pointer.
            // These tensors have data set via setDataPtr and may not be f32 type.
            saveInputTensor(io, allocator, subdir, entry.filename, entry.tensor_name, cgraph, log);
        } else {
            debug_mod.saveTensorFromGraph(io, allocator, subdir, entry.filename, entry.tensor_name, cgraph) catch |err| {
                log.warn("saveDebugData: failed to save '{s}': {}", .{ entry.filename, err });
            };
        }
    }
}

/// Save an input tensor's data to a JSON file.
/// Input tensors may have non-f32 types (e.g., i32 for position indices).
/// This function reads the raw data and converts to f32 for JSON output.
fn saveInputTensor(
    io: std.Io,
    allocator: std.mem.Allocator,
    subdir: []const u8,
    fname: []const u8,
    tensor_name: []const u8,
    cgraph: *ggml.CGraph,
    log: anytype,
) void {
    const c = @import("ggml").c;

    var name_buf: [256]u8 = undefined;
    if (tensor_name.len >= name_buf.len) {
        log.warn("saveInputTensor: tensor name too long ({d} >= {d})", .{ tensor_name.len, name_buf.len });
        return;
    }
    @memcpy(name_buf[0..tensor_name.len], tensor_name);
    name_buf[tensor_name.len] = 0;
    const t = c.ggml_graph_get_tensor(@ptrCast(cgraph), &name_buf);
    if (t == null) {
        log.warn("saveInputTensor: tensor '{s}' not found in graph", .{tensor_name});
        return;
    }

    const tensor = @as(*ggml.Tensor, @ptrCast(t));
    const nelements: usize = @intCast(tensor.nElems());
    const dtype = tensor.dataType();

    // Allocate buffer for f32 conversion
    const f32_data = allocator.alloc(f32, nelements) catch |err| {
        log.warn("saveInputTensor: failed to allocate f32 buffer: {}", .{err});
        return;
    };
    defer allocator.free(f32_data);

    // Read data based on type
    if (dtype == .i32) {
        const i32_data = tensor.dataGet(i32, allocator) catch |err| {
            log.warn("saveInputTensor: failed to read i32 data: {}", .{err});
            return;
        };
        defer allocator.free(i32_data);
        for (i32_data, 0..) |val, i| {
            f32_data[i] = @floatFromInt(val);
        }
    } else if (dtype == .f32) {
        const data = tensor.dataGet(f32, allocator) catch |err| {
            log.warn("saveInputTensor: failed to read f32 data: {}", .{err});
            return;
        };
        defer allocator.free(data);
        @memcpy(f32_data, data);
    } else {
        log.warn("saveInputTensor: unsupported type for '{s}': {}", .{ tensor_name, dtype });
        return;
    }

    debug_mod.saveData(io, subdir, fname, tensor_name, f32_data) catch |err| {
        log.warn("saveInputTensor: failed to save data: {}", .{err});
    };
}
