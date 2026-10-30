//! `ming-tts`: native Ming-3.0-tts-0.5b server and CLI (no Python).
//!
//!   ming-tts serve --model DIR [--host 127.0.0.1] [--port 8765] [--output-dir DIR]
//!   ming-tts say   --model DIR --out x.wav --json '{"task":"IP","text":"Hi.","voice":"UK_George"}'
//!   ming-tts bench --model DIR [--runs 3]
//!
//! `serve` implements the HTTP contract in ming/API.md (v1 + v2) with the
//! same JSON as the Python server. Env defaults match it too: MING_PORT,
//! MING_OUTPUT_DIR, MING_MAX_TEXT_CHARS. All MLX work runs on one worker
//! thread (MLX streams are per-thread); connection threads parse, validate
//! and wait.

const std = @import("std");
const mlx = @import("mlx.zig");
const wav = @import("wav.zig");
const service = @import("ming_service.zig");
const ming = service.engine;
const optStr = service.optStr;
const seedOf = service.seedOf;
const ApiError = service.ApiError;

const Allocator = std.mem.Allocator;

const DEFAULT_MODEL = service.DEFAULT_MODEL;
const DEFAULT_OUTPUT = service.DEFAULT_OUTPUT;

const Config = struct {
    model_dir: []const u8 = DEFAULT_MODEL,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8765,
    output_dir: []const u8 = DEFAULT_OUTPUT,
    max_text_chars: usize = ming.MAX_TEXT_CHARS,
};

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    mlx.installErrorHandler();
    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(a);
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer it.deinit();
    while (it.next()) |x| try args.append(a, x);
    if (args.items.len < 2) return usage();

    var cfg: Config = .{};
    if (std.c.getenv("MING_PORT")) |p| cfg.port = std.fmt.parseInt(u16, std.mem.span(p), 10) catch cfg.port;
    if (std.c.getenv("MING_OUTPUT_DIR")) |p| cfg.output_dir = std.mem.span(p);
    if (std.c.getenv("MING_MAX_TEXT_CHARS")) |p| cfg.max_text_chars = std.fmt.parseInt(usize, std.mem.span(p), 10) catch cfg.max_text_chars;
    var json_arg: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var runs: usize = 3;
    var i: usize = 2;
    while (i < args.items.len) : (i += 1) {
        const k = args.items[i];
        const v = if (i + 1 < args.items.len) args.items[i + 1] else return usage();
        i += 1;
        if (std.mem.eql(u8, k, "--model")) cfg.model_dir = v else if (std.mem.eql(u8, k, "--host")) cfg.host = v else if (std.mem.eql(u8, k, "--port")) {
            cfg.port = try std.fmt.parseInt(u16, v, 10);
        } else if (std.mem.eql(u8, k, "--output-dir")) cfg.output_dir = v else if (std.mem.eql(u8, k, "--json")) json_arg = v else if (std.mem.eql(u8, k, "--out")) out_path = v else if (std.mem.eql(u8, k, "--runs")) {
            runs = try std.fmt.parseInt(usize, v, 10);
        } else return usage();
    }
    const cmd = args.items[1];
    if (std.mem.eql(u8, cmd, "serve")) return serve(io, a, cfg);
    if (std.mem.eql(u8, cmd, "say")) return say(io, a, cfg, json_arg orelse return usage(), out_path orelse return usage());
    if (std.mem.eql(u8, cmd, "bench")) return bench(io, a, cfg, runs);
    if (std.mem.eql(u8, cmd, "frontend")) return frontend(io, a, cfg, json_arg orelse return usage());
    return usage();
}

fn usage() void {
    std.debug.print(
        \\usage:
        \\  ming-tts serve --model DIR [--host 127.0.0.1] [--port 8765] [--output-dir DIR]
        \\  ming-tts say   --model DIR --out x.wav --json '{{"task":"IP","text":"Hi.","voice":"UK_George"}}'
        \\  ming-tts bench --model DIR [--runs 3]
        \\  ming-tts frontend --model DIR --json '[{{"task":"TTS","text":"..."}}]'   (fragments + token ids, no model load)
        \\
    , .{});
}

// ─────────────────────────────────────────────────────────────────────────
// HTTP
// ─────────────────────────────────────────────────────────────────────────

fn serve(io: std.Io, a: Allocator, cfg: Config) !void {
    const st = try service.Service.start(io, a, .{ .model_dir = cfg.model_dir, .output_dir = cfg.output_dir, .max_text_chars = cfg.max_text_chars });
    const addr = try std.Io.net.IpAddress.parse(cfg.host, cfg.port);
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    std.debug.print("ming-tts listening on http://{s}:{d} (model {s}, outputs {s})\n", .{ cfg.host, cfg.port, cfg.model_dir, cfg.output_dir });
    while (true) {
        const stream = server.accept(io) catch |e| {
            std.log.err("accept: {s}", .{@errorName(e)});
            continue;
        };
        const t = std.Thread.spawn(.{}, handleConn, .{ st, stream }) catch {
            stream.close(io);
            continue;
        };
        t.detach();
    }
}

fn handleConn(st: *service.Service, stream: std.Io.net.Stream) void {
    const io = st.io;
    defer stream.close(io);
    var arena = std.heap.ArenaAllocator.init(st.a);
    defer arena.deinit();
    const a = arena.allocator();
    var rbuf: [16 * 1024]u8 = undefined;
    var wbuf: [16 * 1024]u8 = undefined;
    var rs = stream.reader(io, &rbuf);
    var ws = stream.writer(io, &wbuf);
    const req = readRequest(a, &rs.interface) catch return;
    var status: u16 = 200;
    const body = st.handle(a, req.method, req.path, req.body, &status) catch |e| blk: {
        status = 500;
        break :blk std.fmt.allocPrint(a, "{{\"error\":\"{s}\"}}", .{@errorName(e)}) catch return;
    };
    const reason = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        503 => "Service Unavailable",
        else => "Internal Server Error",
    };
    const w = &ws.interface;
    w.print("HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ status, reason, body.len }) catch return;
    w.writeAll(body) catch return;
    w.flush() catch return;
}

const HttpReq = struct { method: []const u8, path: []const u8, body: []const u8 };

fn readRequest(a: Allocator, r: *std.Io.Reader) !HttpReq {
    var buf: std.ArrayList(u8) = .empty;
    var head_end: ?usize = null;
    while (head_end == null) {
        if (buf.items.len > 64 * 1024) return error.HeadersTooLarge;
        var chunk: [4096]u8 = undefined;
        var bufs: [1][]u8 = .{&chunk};
        const n = r.readVec(&bufs) catch |e| switch (e) {
            error.EndOfStream => 0,
            else => return e,
        };
        if (n == 0) return error.EndOfStream;
        try buf.appendSlice(a, chunk[0..n]);
        head_end = std.mem.indexOf(u8, buf.items, "\r\n\r\n");
    }
    const head = buf.items[0..head_end.?];
    const line_end = std.mem.indexOf(u8, head, "\r\n") orelse head.len;
    var parts = std.mem.splitScalar(u8, head[0..line_end], ' ');
    const method = parts.next() orelse return error.BadRequest;
    var path = parts.next() orelse return error.BadRequest;
    if (std.mem.indexOfScalar(u8, path, '?')) |q| path = path[0..q];
    var clen: usize = 0;
    var lines = std.mem.splitSequence(u8, head[line_end..], "\r\n");
    while (lines.next()) |l| {
        const c = std.mem.indexOfScalar(u8, l, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, l[0..c], " "), "content-length"))
            clen = std.fmt.parseInt(usize, std.mem.trim(u8, l[c + 1 ..], " "), 10) catch 0;
    }
    if (clen > 16 * 1024 * 1024) return error.BodyTooLarge;
    const start = head_end.? + 4;
    while (buf.items.len < start + clen) {
        var chunk: [16384]u8 = undefined;
        var bufs: [1][]u8 = .{&chunk};
        const n = r.readVec(&bufs) catch |e| switch (e) {
            error.EndOfStream => 0,
            else => return e,
        };
        if (n == 0) break;
        try buf.appendSlice(a, chunk[0..n]);
    }
    const end = @min(buf.items.len, start + clen);
    return .{ .method = method, .path = path, .body = buf.items[start..end] };
}

// ─────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────

fn say(io: std.Io, a: Allocator, cfg: Config, json: []const u8, out: []const u8) !void {
    var eng = try ming.Engine.init(io, a, mlx.mlx_default_gpu_stream_new(), cfg.model_dir, .{});
    defer eng.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    var perr: ?ApiError = null;
    var r: ming.GenerateRequest = .{
        .task = o.get("task").?.string,
        .text = o.get("text").?.string,
        .language = optStr(o, "language", &perr, a),
        .prompt_wav_path = optStr(o, "prompt_wav_path", &perr, a),
        .prompt_text = optStr(o, "prompt_text", &perr, a),
        .voice = optStr(o, "voice", &perr, a),
        .instruction = optStr(o, "instruction", &perr, a),
        .seed = seedOf(o, &perr, a),
    };
    if (o.get("controls")) |cv| if (cv == .object) {
        var cit = cv.object.iterator();
        while (cit.next()) |kv| {
            const k = kv.key_ptr.*;
            const v = kv.value_ptr.string;
            if (std.mem.eql(u8, k, "speed")) r.controls.speed = v else if (std.mem.eql(u8, k, "pitch")) r.controls.pitch = v else if (std.mem.eql(u8, k, "volume")) r.controls.volume = v else if (std.mem.eql(u8, k, "emotion")) r.controls.emotion = v else if (std.mem.eql(u8, k, "dialect")) r.controls.dialect = v else if (std.mem.eql(u8, k, "style")) r.controls.style = v;
        }
    };
    var d: ming.Diag = .{};
    const t0 = std.Io.Timestamp.now(io, .awake);
    const samples = eng.generate(r, &d) catch |e| {
        if (e == error.BadRequest) std.debug.print("error: {s}\n", .{d.msg()});
        return e;
    };
    defer a.free(samples);
    const el = @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e9;
    const bytes = try wav.encodePcm16Mono(a, samples, ming.SAMPLE_RATE);
    defer a.free(bytes);
    const f = try std.Io.Dir.cwd().createFile(io, out, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
    const dur = @as(f64, @floatFromInt(samples.len)) / ming.SAMPLE_RATE;
    std.debug.print("{s}: {d:.2}s audio in {d:.2}s (rtf {d:.3}, includes first-call kernel compile)\n", .{ out, dur, el, el / dur });
}

/// Same requests as engine-mlx/bench.py; median RTF over `runs` seeds after warm-up.
fn bench(io: std.Io, a: Allocator, cfg: Config, runs: usize) !void {
    var eng = try ming.Engine.init(io, a, mlx.mlx_default_gpu_stream_new(), cfg.model_dir, .{});
    defer eng.deinit();
    var d: ming.Diag = .{};
    a.free(try eng.tts("Warm up.", "UK_George", null, 0, &d));
    const ref = "/Users/sbusso/Code/dev/ming/vendor/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav";
    const Case = struct { name: []const u8, clone: bool, txt: []const u8 };
    const cases = [_]Case{
        .{ .name = "tts-en", .clone = false, .txt = "The garden will remain open until sunset, provided the weather stays clear, and visitors are asked to leave by the north gate." },
        .{ .name = "tts-zh", .clone = false, .txt = "雨后的山谷很安静，远处的云慢慢变成了金色，我们沿着小路走回了村子。" },
        .{ .name = "clone", .clone = true, .txt = "I never thought a little machine could sound this much like me." },
    };
    for (cases) |c| {
        var rtfs: std.ArrayList(f64) = .empty;
        defer rtfs.deinit(a);
        var durs: std.ArrayList(f64) = .empty;
        defer durs.deinit(a);
        for (0..runs) |seed| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            const w = if (c.clone)
                try eng.clone(c.txt, ref, "诶，你今天有空吗？我们能不能聊聊这件事啊？", seed, &d)
            else
                try eng.tts(c.txt, null, null, seed, &d);
            defer a.free(w);
            const el = @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e9;
            const dur = @as(f64, @floatFromInt(w.len)) / ming.SAMPLE_RATE;
            try rtfs.append(a, el / dur);
            try durs.append(a, dur);
        }
        std.mem.sort(f64, rtfs.items, {}, std.sort.asc(f64));
        std.mem.sort(f64, durs.items, {}, std.sort.asc(f64));
        std.debug.print("zig        {s:<7} rtf={d:.3}  audio={d:.2}s\n", .{ c.name, rtfs.items[rtfs.items.len / 2], durs.items[durs.items.len / 2] });
    }
}

/// Text frontend dump for comparison with the Python one: for each request,
/// the prompt, the fragments `prepare_text` makes and each fragment's token
/// ids (no speaker slot, no prompt transcript). Loads only the tokenizer.
fn frontend(io: std.Io, a: Allocator, cfg: Config, json: []const u8) !void {
    if (std.c.getenv("MING_FR_TN")) |v| ming.text.french_frontend = !std.mem.eql(u8, std.mem.span(v), "0");
    var tok = try @import("tokenizer.zig").loadTokenizerAny(io, a, cfg.model_dir);
    defer tok.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try out.append(a, '[');
    for (parsed.value.array.items, 0..) |item, k| {
        const o = item.object;
        var perr: ?ApiError = null;
        const r: ming.GenerateRequest = .{
            .task = o.get("task").?.string,
            .text = o.get("text").?.string,
            .language = optStr(o, "language", &perr, a),
            .voice = optStr(o, "voice", &perr, a),
            .prompt_wav_path = optStr(o, "prompt_wav_path", &perr, a),
            .prompt_text = optStr(o, "prompt_text", &perr, a),
        };
        var d: ming.Diag = .{};
        // The upstream kwargs directly (no API.md reference checks).
        var in: ming.text.TaskInput = .{ .task = r.task, .text = r.text, .language_id = r.language };
        if (r.voice) |v| in.controls.ip = v;
        const req = ming.text.buildRequest(a, in, &d) catch |e| {
            std.debug.print("error: {s}\n", .{d.msg()});
            return e;
        };
        defer req.deinit(a);
        var frags = try ming.text.prepareText(a, req.text, req.text_language, 24);
        defer {
            for (frags.items) |f| a.free(f);
            frags.deinit(a);
        }
        if (k > 0) try out.append(a, ',');
        try out.appendSlice(a, "\n{\"text\":");
        try ming.text.jsonStr(a, &out, r.text);
        try out.appendSlice(a, ",\"prompt\":");
        try ming.text.jsonStr(a, &out, req.prompt);
        try out.appendSlice(a, ",\"fragments\":[");
        for (frags.items, 0..) |f, i| {
            if (i > 0) try out.append(a, ',');
            try ming.text.jsonStr(a, &out, f);
        }
        try out.appendSlice(a, "],\"ids\":[");
        for (frags.items, 0..) |f, i| {
            if (i > 0) try out.append(a, ',');
            var ii = try ming.text.buildInputIds(a, &tok, req.prompt, f, 0, req.instruction, null, 0);
            defer ii.deinit(a);
            try out.append(a, '[');
            for (ii.ids, 0..) |id, j| try out.print(a, "{s}{d}", .{ if (j > 0) "," else "", id });
            try out.append(a, ']');
        }
        try out.appendSlice(a, "]}");
    }
    try out.appendSlice(a, "]\n");
    std.debug.print("{s}", .{out.items});
}
