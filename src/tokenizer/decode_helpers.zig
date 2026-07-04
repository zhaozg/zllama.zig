const std = @import("std");
pub fn decodeTiktokenSingle(ts: []const u8, buf: []u8) usize {
    var written: usize = 0;
    var rem = ts;
    while (rem.len > 0 and written < buf.len) {
        if (rem.len >= 4 and rem[0] == '<' and rem[1] == '0' and rem[2] == 'x') {
            const end = std.mem.indexOfScalar(u8, rem[1..], '>') orelse {
                buf[written] = rem[0];
                written += 1;
                rem = rem[1..];
                continue;
            };
            const hex_str = rem[2 .. 2 + end - 1];
            if (hex_str.len == 2) {
                const byte = std.fmt.parseInt(u8, hex_str, 16) catch {
                    const copy_len = @min(end + 1, buf.len - written);
                    @memcpy(buf[written .. written + copy_len], rem[0..copy_len]);
                    written += copy_len;
                    rem = rem[end + 1 ..];
                    continue;
                };
                buf[written] = byte;
                written += 1;
                rem = rem[end + 1 ..];
                continue;
            }
        }
        buf[written] = rem[0];
        written += 1;
        rem = rem[1..];
    }
    return written;
}

pub fn decodeGpt2Single(ts: []const u8, unicode_to_byte: *const std.StringHashMap(u8), buf: []u8) usize {
    var written: usize = 0;
    var i: usize = 0;
    while (i < ts.len and written < buf.len) {
        const byte = ts[i];
        const cp_len = std.unicode.utf8ByteSequenceLength(byte) catch {
            i += 1;
            continue;
        };
        if (i + cp_len > ts.len) {
            i += 1;
            continue;
        }
        const cp_slice = ts[i .. i + cp_len];
        if (cp_len == 1 and byte < 0x80) {
            buf[written] = byte;
            written += 1;
            i += 1;
            continue;
        }
        if (unicode_to_byte.get(cp_slice)) |b| {
            buf[written] = b;
            written += 1;
            i += cp_len;
            continue;
        }
        const copy_len = @min(cp_len, buf.len - written);
        @memcpy(buf[written .. written + copy_len], cp_slice[0..copy_len]);
        written += copy_len;
        i += cp_len;
    }
    return written;
}

pub fn decodeSPMSingle(ts: []const u8, buf: []u8) usize {
    var written: usize = 0;
    var i: usize = 0;
    while (i < ts.len and written < buf.len) {
        if (ts[i] == '<' and i + 3 < ts.len and ts[i + 1] == '0' and ts[i + 2] == 'x') {
            const end = std.mem.indexOfScalar(u8, ts[i + 1 ..], '>') orelse {
                buf[written] = ts[i];
                written += 1;
                i += 1;
                continue;
            };
            const hex_str = ts[i + 3 .. i + 1 + end];
            if (hex_str.len == 2) {
                if (std.fmt.parseInt(u8, hex_str, 16)) |byte| {
                    buf[written] = byte;
                    written += 1;
                    i = i + 1 + end + 1;
                    continue;
                } else |_| {}
            }
            buf[written] = ts[i];
            written += 1;
            i += 1;
            continue;
        }
        if (i + 2 < ts.len and ts[i] == 0xE2 and ts[i + 1] == 0x96 and ts[i + 2] == 0x81) {
            buf[written] = ' ';
            written += 1;
            i += 3;
            continue;
        }
        if (i + 1 < ts.len and ts[i] == 0xC4 and ts[i + 1] == 0xA0) {
            buf[written] = ' ';
            written += 1;
            i += 2;
            continue;
        }
        buf[written] = ts[i];
        written += 1;
        i += 1;
    }
    return written;
}

// ============================================================================
