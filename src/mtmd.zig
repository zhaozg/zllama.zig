//! mtmd — Multi-modal decoding module
//! Reference: deps/llama.cpp/tools/mtmd/mtmd.h

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model = @import("model");
const mm = @import("mm");
const preprocess = @import("preprocess");
const tokenizer = @import("tokenizer");
const utils = @import("utils");

const log = std.log.scoped(.mtmd);

pub const audio = @import("audio");
pub const vision = @import("vision");
pub const helper = @import("helper");
pub const tokenize = @import("tokenize");
pub const manager = mm;
pub const preprocess_mod = preprocess;

pub const ChunkType = enum(u8) { text, image, audio };
pub const PosType = enum(u8) { normal, mrope, hunyuanvl };
pub const DecoderPos = struct { t: u32 = 0, x: u32 = 0, y: u32 = 0, z: u32 = 0 };

pub const ContextParams = struct {
    use_gpu: bool = true,
    print_timings: bool = true,
    n_threads: u32 = 4,
    media_marker: []const u8 = "<__media__>",
    warmup: bool = true,
    image_min_tokens: i32 = -1,
    image_max_tokens: i32 = -1,
};
pub fn defaultMarker() []const u8 {
    return "<__media__>";
}
pub fn contextParamsDefault() ContextParams {
    return .{};
}
pub const Caps = struct { inp_vision: bool = false, inp_audio: bool = false };

pub const Bitmap = struct {
    nx: u32,
    ny: u32,
    is_audio: bool = false,
    id: ?[]const u8 = null,
    data: ?[]const u8 = null,
    allocator: ?std.mem.Allocator = null,
    pub fn initImage(nx: u32, ny: u32, data: ?[]const u8) Bitmap {
        return .{ .nx = nx, .ny = ny, .data = data };
    }
    pub fn initAudio(n: u32, data: ?[]const u8) Bitmap {
        return .{ .nx = n, .ny = 1, .is_audio = true, .data = data };
    }
    pub fn initPlaceholderImage(nx: u32, ny: u32) Bitmap {
        return .{ .nx = nx, .ny = ny };
    }
    pub fn initPlaceholderAudio(n: u32) Bitmap {
        return .{ .nx = n, .ny = 1, .is_audio = true };
    }
    pub fn isPlaceholder(self: Bitmap) bool {
        return self.data == null;
    }
    pub fn nBytes(self: Bitmap) usize {
        return if (self.data) |d| d.len else 0;
    }
    pub fn deinit(self: *Bitmap) void {
        if (self.allocator) |a| {
            if (self.data) |d| a.free(d);
            self.data = null;
            self.allocator = null;
        }
    }
};

pub const ImageTokens = struct {
    nx: u32 = 0,
    ny: u32 = 0,
    pos: PosType = .normal,
    image_idx: u32 = 0,
    id: ?[]const u8 = null,
    patches: ?*anyopaque = null,
    patch_count: u32 = 0,
    pub fn nTokens(self: ImageTokens) u32 {
        return switch (self.pos) {
            .hunyuanvl => (self.nx + 1) * self.ny + 2,
            else => self.nx * self.ny,
        };
    }
    pub fn isPlaceholder(self: ImageTokens) bool {
        return self.patches == null;
    }
};

pub const InputChunk = struct {
    chunk_type: ChunkType,
    tokens_text: ?[]const i32 = null,
    tokens_image: ?ImageTokens = null,
    tokens_audio_n: u32 = 0,
    id: ?[]const u8 = null,
    pub fn nPos(self: InputChunk) u32 {
        return switch (self.chunk_type) {
            .text => @intCast(self.tokens_text.?.len),
            .image => {
                const img = &(self.tokens_image orelse return 0);
                return switch (img.pos) {
                    .mrope, .hunyuanvl => @max(@max(img.nx, img.ny), 1),
                    .normal => img.nTokens(),
                };
            },
            .audio => self.tokens_audio_n,
        };
    }
    pub fn nTokens(self: InputChunk) u32 {
        return switch (self.chunk_type) {
            .text => @intCast(self.tokens_text.?.len),
            .image => if (self.tokens_image) |img| img.nTokens() else 0,
            .audio => self.tokens_audio_n,
        };
    }
};

pub const InputChunks = struct {
    entries: std.ArrayList(InputChunk),
    allocator: std.mem.Allocator,
    pub fn init(a: std.mem.Allocator) InputChunks {
        return .{ .entries = std.ArrayList(InputChunk).initCapacity(a, 0) catch @panic("OOM"), .allocator = a };
    }
    pub fn deinit(self: *InputChunks) void {
        for (self.entries.items) |*chunk| {
            if (chunk.tokens_text) |t| self.allocator.free(t);
        }
        self.entries.clearAndFree(self.allocator);
    }
    pub fn size(self: InputChunks) usize {
        return self.entries.items.len;
    }
    pub fn get(self: InputChunks, idx: usize) *const InputChunk {
        return &self.entries.items[idx];
    }
    pub fn append(self: *InputChunks, chunk: InputChunk) !void {
        try self.entries.append(self.allocator, chunk);
    }
    pub fn totalTokens(self: InputChunks) usize {
        var t: usize = 0;
        for (self.entries.items) |c| t += c.nTokens();
        return t;
    }
    pub fn totalPos(self: InputChunks) usize {
        var t: usize = 0;
        for (self.entries.items) |c| t += c.nPos();
        return t;
    }
};

pub const InputText = struct { text: []const u8, add_special: bool = true, parse_special: bool = true };

pub const MtmdContext = struct {
    allocator: std.mem.Allocator,
    mm_manager: *mm.MultiModalManager,
    caps: model.ModelCapabilities,
    params: ContextParams,
    n_embd_text: i32,
    tok: ?*tokenizer.Tokenizer = null,
    media_marker: []const u8,
    img_beg: []const u8 = "",
    img_end: []const u8 = "",
    aud_beg: []const u8 = "",
    aud_end: []const u8 = "",
    pos_type: PosType = .normal,
    output_embd: ?[]f32 = null,

    pub fn init(allocator: std.mem.Allocator, mmproj_path: []const u8, io: std.Io, text_n_embd: i32, params: ContextParams, tok: ?*tokenizer.Tokenizer) !*MtmdContext {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
        defer file.close(io);
        var gf = try gguf.GGUFFile.init(allocator, file, io);
        defer gf.deinit();
        const caps = try detectCapabilities(&gf);
        const buf_size: usize = 512 * 1024 * 1024;
        const ctx = try ggml.Context.init(allocator, buf_size, false);
        const mgr = try allocator.create(mm.MultiModalManager);
        mgr.* = try mm.MultiModalManager.init(allocator, &gf, ctx, caps);
        const pos_type: PosType = .normal;
        var ib: []const u8 = "";
        var ie: []const u8 = "";
        var ab: []const u8 = "";
        var ae: []const u8 = "";
        if (caps.has_vision) {
            if (std.mem.eql(u8, caps.vision_encoder_type, "gemma4v") or std.mem.eql(u8, caps.vision_encoder_type, "gemma4uv")) {
                ib = "<|image>";
                ie = "<image|>";
            } else {
                ib = "<start_of_image>";
                ie = "<end_of_image>";
            }
        }
        if (caps.has_audio) {
            if (std.mem.eql(u8, caps.audio_encoder_type, "gemma4a") or std.mem.eql(u8, caps.audio_encoder_type, "gemma4ua")) {
                ab = "<|audio>";
                ae = "<audio|>";
            }
        }
        const self = try allocator.create(MtmdContext);
        self.* = .{ .allocator = allocator, .mm_manager = mgr, .caps = caps, .params = params, .n_embd_text = text_n_embd, .tok = tok, .media_marker = params.media_marker, .img_beg = ib, .img_end = ie, .aud_beg = ab, .aud_end = ae, .pos_type = pos_type };
        return self;
    }

    pub fn deinit(self: *MtmdContext) void {
        if (self.output_embd) |e| self.allocator.free(e);
        self.mm_manager.deinit();
        self.allocator.destroy(self.mm_manager);
        self.allocator.destroy(self);
    }

    pub fn supportVision(self: *const MtmdContext) bool {
        return self.caps.has_vision;
    }
    pub fn supportAudio(self: *const MtmdContext) bool {
        return self.caps.has_audio;
    }
    pub fn getAudioSampleRate(self: *const MtmdContext) i32 {
        return self.caps.audio_sample_rate;
    }
    pub fn decodeUseNonCausal(self: *const MtmdContext, chunk: ?*const InputChunk) bool {
        if (self.caps.vision_encoder_type.len > 0 and std.mem.startsWith(u8, self.caps.vision_encoder_type, "gemma")) {
            if (chunk) |c| return c.chunk_type == .image;
            return true;
        }
        return false;
    }
    pub fn decodeUseMRope(self: *const MtmdContext) bool {
        return self.pos_type == .mrope or self.pos_type == .hunyuanvl;
    }
    pub fn getMarker(self: *const MtmdContext) []const u8 {
        return self.media_marker;
    }
    pub fn getOutputEmbd(self: *MtmdContext) ?[]f32 {
        return self.output_embd;
    }
};

fn detectCapabilities(gf: *const gguf.GGUFFile) !model.ModelCapabilities {
    var caps = model.ModelCapabilities{};
    if (gf.findTensor("v.patch_embd.weight") != null or gf.findTensor("v.position_embd.weight") != null) {
        caps.has_vision = true;
        // 与 vision.zig 加载代码保持一致的命名前缀
        caps.vision_encoder_type = if (gf.findTensor("v.patch_norm.1.weight") != null or
            gf.findTensor("patch_norm_1.weight") != null)
            "gemma4uv"
        else
            "gemma4v";
    }
    if (gf.findTensor("a.conv1d.0.weight") != null or gf.findTensor("a.input_projection.weight") != null) {
        caps.has_audio = true;
        caps.audio_encoder_type = "gemma4a";
        caps.audio_sample_rate = 16000;
        if (gf.getU32("gemma4.audio.sample_rate")) |v| caps.audio_sample_rate = @intCast(v);
    }
    return caps;
}

pub fn getCapFromFile(allocator: std.mem.Allocator, mmproj_path: []const u8, io: std.Io) !Caps {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
    defer file.close(io);
    var gf = try gguf.GGUFFile.init(allocator, file, io);
    defer gf.deinit();
    const caps = try detectCapabilities(&gf);
    return .{ .inp_vision = caps.has_vision, .inp_audio = caps.has_audio };
}
