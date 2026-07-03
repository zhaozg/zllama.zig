const std = @import("std");
const gguf = @import("gguf");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: dump_tensors <model.gguf>\n", .{});
        return;
    }

    const cwd = std.fs.cwd();
    const file = try cwd.openFile(args[1], .{ .mode = .read_only });
    defer file.close();

    var gf = try gguf.GGUFFile.init(allocator, file, std.io.getStdIn());
    defer gf.deinit();

    std.debug.print("Tensors:\n", .{});
    for (gf.tensors.items, 0..) |t, i| {
        if (i >= 30) break;
        std.debug.print("  {s}: dims={any}\n", .{ t.name, t.dims[0..t.n_dims] });
    }
}
