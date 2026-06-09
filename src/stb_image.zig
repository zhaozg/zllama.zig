//! stb_image Zig wrapper — JPEG/PNG/GIF/BMP/TGA/PSD image loading
//!
//! Uses stb_image.h (public domain, single-header) for decoding.
//! The C implementation is compiled from vendor/stb/stb_image.c.
//!
//! API surface mirrors stb_image essentials:
//!   - load(): decode image into RGB(A) bytes, return width/height/channels
//!   - free(): release stb_image-allocated memory

/// Load an image file, returning raw pixel data + dimensions.
/// Returns null on failure — call failureReason() for details.
pub fn load(path: [*:0]const u8, w: *i32, h: *i32, comp: *i32, req_comp: i32) ?[*]u8 {
    return @as(?[*]u8, @ptrCast(stbi_load(path, @ptrCast(w), @ptrCast(h), @ptrCast(comp), @intCast(req_comp))));
}

/// Load from memory buffer.
pub fn loadFromMemory(buffer: [*]const u8, len: i32, w: *i32, h: *i32, comp: *i32, req_comp: i32) ?[*]u8 {
    return @as(?[*]u8, @ptrCast(stbi_load_from_memory(buffer, @intCast(len), @ptrCast(w), @ptrCast(h), @ptrCast(comp), @intCast(req_comp))));
}

/// Free pixel data allocated by stb_image.
pub fn free(data: ?[*]u8) void {
    stbi_image_free(@ptrCast(data));
}

/// Get a human-readable description of the last failure.
pub fn failureReason() [*:0]const u8 {
    return stbi_failure_reason();
}

/// Set flip vertically on load (useful for OpenGL-style bottom-left origin).
pub fn setFlipVertically(flip: i32) void {
    stbi_set_flip_vertically_on_load(@intCast(flip));
}

// ============================================================================
// C imports
// ============================================================================
const stb = @cImport({
    @cDefine("STB_IMAGE_STATIC", "");
    @cInclude("stb_image.h");
});

const stbi_load = stb.stbi_load;
const stbi_load_from_memory = stb.stbi_load_from_memory;
const stbi_image_free = stb.stbi_image_free;
const stbi_failure_reason = stb.stbi_failure_reason;
const stbi_set_flip_vertically_on_load = stb.stbi_set_flip_vertically_on_load;
