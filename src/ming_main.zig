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
const ming = @import("ming_tts.zig");
const text = ming.text;

const Allocator = std.mem.Allocator;

const DEFAULT_MODEL = "/Users/sbusso/Code/dev/ming/models/Ming-3.0-tts-0.5b";
const DEFAULT_OUTPUT = "/Users/sbusso/Code/dev/ming/outputs";

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
    return usage();
}

fn usage() void {
    std.debug.print(
        \\usage:
        \\  ming-tts serve --model DIR [--host 127.0.0.1] [--port 8765] [--output-dir DIR]
        \\  ming-tts say   --model DIR --out x.wav --json '{{"task":"IP","text":"Hi.","voice":"UK_George"}}'
        \\  ming-tts bench --model DIR [--runs 3]
        \\
    , .{});
}

// ─────────────────────────────────────────────────────────────────────────
// Request bodies → engine calls
// ─────────────────────────────────────────────────────────────────────────

const ApiError = struct {
    status: u16,
    msg: []u8,
};

const Body = std.json.ObjectMap;

/// Optional string field: wrong type → 400; blank → null (stripped).
fn optStr(o: Body, k: []const u8, err: *?ApiError, a: Allocator) ?[]const u8 {
    const v = o.get(k) orelse return null;
    switch (v) {
        .null => return null,
        .string => |s| {
            const t = text.strip(s);
            return if (t.len == 0) null else t;
        },
        else => {
            err.* = .{ .status = 400, .msg = std.fmt.allocPrint(a, "{s}: Input should be a valid string", .{k}) catch return null };
            return null;
        },
    }
}

fn seedOf(o: Body, err: *?ApiError, a: Allocator) u64 {
    const v = o.get("seed") orelse return 0;
    return switch (v) {
        .null => 0,
        .integer => |n| @bitCast(n),
        else => blk: {
            err.* = .{ .status = 400, .msg = std.fmt.allocPrint(a, "seed: Input should be a valid integer", .{}) catch break :blk 0 };
            break :blk 0;
        },
    };
}

/// A parsed, validated job for the inference thread. Strings live in `arena`.
const Job = struct {
    kind: enum { tts, clone, generate, warmup },
    req: ming.GenerateRequest = .{ .task = "", .text = "" },
    arena: std.heap.ArenaAllocator,
    // results (set by the worker)
    samples: ?[]f32 = null,
    elapsed_s: f64 = 0,
    err: ?ApiError = null,
    done: bool = false,
};

const State = struct {
    io: std.Io,
    a: Allocator,
    cfg: Config,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*Job) = .empty,
    loaded: std.atomic.Value(bool) = .init(false),
    load_error: ?[]u8 = null,
};

fn worker(st: *State) void {
    const a = st.a;
    const s = mlx.mlx_default_gpu_stream_new();
    var eng = ming.Engine.init(st.io, a, s, st.cfg.model_dir, .{}) catch |e| {
        st.load_error = std.fmt.allocPrint(a, "{s}", .{@errorName(e)}) catch null;
        std.log.err("[ming] model load failed: {s}", .{@errorName(e)});
        return;
    };
    defer eng.deinit();
    // Compile Metal kernels once so the first request is not slow.
    var d: ming.Diag = .{};
    if (eng.tts("Warm up.", "UK_George", null, 0, &d)) |w| a.free(w) else |_| {}
    st.loaded.store(true, .release);
    std.log.info("[ming] model loaded; serving", .{});
    while (true) {
        st.mu.lockUncancelable(st.io);
        while (st.queue.items.len == 0) st.cond.waitUncancelable(st.io, &st.mu);
        const job = st.queue.orderedRemove(0);
        st.mu.unlock(st.io);
        runJob(&eng, job);
        st.mu.lockUncancelable(st.io);
        job.done = true;
        st.cond.broadcast(st.io);
        st.mu.unlock(st.io);
    }
}

fn runJob(eng: *ming.Engine, job: *Job) void {
    const ja = job.arena.allocator();
    var d: ming.Diag = .{};
    const t0 = std.Io.Timestamp.now(eng.io, .awake);
    const r = job.req;
    const res = switch (job.kind) {
        .tts => eng.tts(r.text, r.voice, r.instruction, r.seed, &d),
        .clone => eng.clone(r.text, r.prompt_wav_path.?, r.prompt_text, r.seed, &d),
        .generate, .warmup => eng.generate(r, &d),
    };
    job.elapsed_s = @as(f64, @floatFromInt(t0.untilNow(eng.io, .awake).nanoseconds)) / 1e9;
    if (res) |samples| {
        job.samples = samples;
    } else |e| {
        job.err = switch (e) {
            error.BadRequest => .{ .status = 400, .msg = ja.dupe(u8, d.msg()) catch &.{} },
            error.NoFragments => .{ .status = 400, .msg = std.fmt.allocPrint(ja, "text produced no synthesizable fragments: '{s}'", .{r.text}) catch &.{} },
            error.PromptTooLong => .{ .status = 400, .msg = ja.dupe(u8, "prompt exhausts the 512-position cache; shorten the text, instruction, or reference audio") catch &.{} },
            else => .{ .status = 500, .msg = std.fmt.allocPrint(ja, "{s}", .{@errorName(e)}) catch &.{} },
        };
    }
}

fn submit(st: *State, job: *Job) void {
    st.mu.lockUncancelable(st.io);
    defer st.mu.unlock(st.io);
    st.queue.append(st.a, job) catch {
        job.err = .{ .status = 500, .msg = &.{} };
        job.done = true;
        return;
    };
    st.cond.broadcast(st.io);
    while (!job.done) st.cond.waitUncancelable(st.io, &st.mu);
}

// ─────────────────────────────────────────────────────────────────────────
// HTTP
// ─────────────────────────────────────────────────────────────────────────

fn serve(io: std.Io, a: Allocator, cfg: Config) !void {
    const st = try a.create(State);
    st.* = .{ .io = io, .a = a, .cfg = cfg };
    const th = try std.Thread.spawn(.{}, worker, .{st});
    th.detach();
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

fn handleConn(st: *State, stream: std.Io.net.Stream) void {
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
    const body = route(st, a, req.method, req.path, req.body, &status) catch |e| blk: {
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

fn errJson(a: Allocator, status: *u16, code: u16, msg: []const u8) ![]u8 {
    status.* = code;
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"error\":");
    try text.jsonStr(a, &out, msg);
    try out.append(a, '}');
    return out.toOwnedSlice(a);
}

fn route(st: *State, a: Allocator, method: []const u8, path: []const u8, body: []const u8, status: *u16) ![]u8 {
    const get = std.mem.eql(u8, method, "GET");
    const post = std.mem.eql(u8, method, "POST");
    if (std.mem.eql(u8, path, "/health")) {
        if (!get) return errJson(a, status, 405, "Method Not Allowed");
        var out: std.ArrayList(u8) = .empty;
        const name = std.fs.path.basename(st.cfg.model_dir);
        try out.appendSlice(a, if (st.load_error != null) "{\"status\":\"error\",\"engine\":\"mlx\",\"model\":" else "{\"status\":\"ok\",\"engine\":\"mlx\",\"model\":");
        try text.jsonStr(a, &out, name);
        try out.print(a, ",\"device\":\"gpu\",\"sample_rate\":{d},\"loaded\":{}", .{ ming.SAMPLE_RATE, st.loaded.load(.acquire) });
        if (st.load_error) |e| {
            try out.appendSlice(a, ",\"error\":");
            try text.jsonStr(a, &out, e);
        }
        try out.append(a, '}');
        return out.toOwnedSlice(a);
    }
    if (std.mem.eql(u8, path, "/voices")) {
        if (!get) return errJson(a, status, 405, "Method Not Allowed");
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(a, "{\"voices\":[");
        var first = true;
        for ([_][]const []const u8{ &text.EN_VOICES, &text.ZH_VOICES }, [_][]const u8{ "en", "zh" }) |list, lang| {
            for (list) |v| {
                if (!first) try out.append(a, ',');
                first = false;
                try out.appendSlice(a, "{\"id\":");
                try text.jsonStr(a, &out, v);
                try out.appendSlice(a, ",\"name\":");
                try text.jsonStr(a, &out, v);
                try out.print(a, ",\"language\":\"{s}\"}}", .{lang});
            }
        }
        try out.appendSlice(a, "]}");
        return out.toOwnedSlice(a);
    }
    if (std.mem.eql(u8, path, "/capabilities")) {
        if (!get) return errJson(a, status, 405, "Method Not Allowed");
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(a, "{\"tasks\":[");
        for (ming.V2_TASKS, 0..) |t, k| {
            if (k > 0) try out.append(a, ',');
            try text.jsonStr(a, &out, t);
        }
        try out.appendSlice(a, "],\"languages\":{\"zh\":\"Chinese\",\"en\":\"English\"");
        for (text.MULTILINGUAL) |p| try out.print(a, ",\"{s}\":\"{s}\"", .{ p[0], p[1] });
        try out.appendSlice(a, "},\"dialects\":");
        try strList(a, &out, &text.DIALECT_VALUES);
        try out.appendSlice(a, ",\"emotions\":");
        try strList(a, &out, &text.EMOTION_VALUES);
        try out.appendSlice(a, ",\"basic\":{\"speed\":");
        try strList(a, &out, &text.BASIC_SPEED);
        try out.appendSlice(a, ",\"pitch\":");
        try strList(a, &out, &text.BASIC_LEVEL);
        try out.appendSlice(a, ",\"volume\":");
        try strList(a, &out, &text.BASIC_LEVEL);
        try out.appendSlice(a, "},\"paralinguistic_tags\":");
        try strList(a, &out, &text.PARALINGUISTIC_TAGS);
        try out.print(a, ",\"max_prompt_s\":{d:.1},\"max_text_chars\":{d}}}", .{ ming.MAX_PROMPT_SECONDS, st.cfg.max_text_chars });
        return out.toOwnedSlice(a);
    }
    const synth = std.mem.eql(u8, path, "/tts") or std.mem.eql(u8, path, "/clone") or std.mem.eql(u8, path, "/generate");
    if (!synth) return errJson(a, status, 404, "Not Found");
    if (!post) return errJson(a, status, 405, "Method Not Allowed");

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return errJson(a, status, 400, "body: JSON decode error");
    if (parsed != .object) return errJson(a, status, 400, "body: Input should be a valid dictionary");
    const o = parsed.object;
    var perr: ?ApiError = null;
    const raw_text = o.get("text") orelse return errJson(a, status, 400, "text: Field required");
    if (raw_text != .string) return errJson(a, status, 400, "text: Input should be a valid string");
    // _check_text
    const txt = text.strip(raw_text.string);
    if (txt.len == 0) return errJson(a, status, 400, "text must not be empty");
    const nchars = std.unicode.utf8CountCodepoints(txt) catch txt.len;
    if (nchars > st.cfg.max_text_chars) return errJson(a, status, 400, try std.fmt.allocPrint(a, "text is {d} chars; limit is {d}", .{ nchars, st.cfg.max_text_chars }));

    const job = try st.a.create(Job);
    defer st.a.destroy(job);
    job.* = .{ .kind = .generate, .arena = std.heap.ArenaAllocator.init(st.a) };
    defer job.arena.deinit();
    const seed = seedOf(o, &perr, a);
    if (std.mem.eql(u8, path, "/tts")) {
        const voice = optStr(o, "voice", &perr, a);
        const instr = if (voice != null) null else optStr(o, "instruction", &perr, a);
        if (perr) |e| return errJson(a, status, e.status, e.msg);
        job.kind = .tts;
        job.req = .{ .task = "TTS", .text = txt, .voice = voice, .instruction = instr, .seed = seed };
    } else if (std.mem.eql(u8, path, "/clone")) {
        const p = o.get("prompt_wav_path") orelse return errJson(a, status, 400, "prompt_wav_path: Field required");
        if (p != .string) return errJson(a, status, 400, "prompt_wav_path: Input should be a valid string");
        const pth = checkPrompt(st.io, a, p.string, status) catch |e| switch (e) {
            error.Reported => return a.dupe(u8, last_prompt_error),
            else => return e,
        };
        const pt = optStr(o, "prompt_text", &perr, a);
        if (perr) |e| return errJson(a, status, e.status, e.msg);
        job.kind = .clone;
        job.req = .{ .task = "TTS", .text = txt, .prompt_wav_path = pth, .prompt_text = pt, .seed = seed };
    } else {
        const task_v = o.get("task") orelse return errJson(a, status, 400, "task: Field required");
        if (task_v != .string) return errJson(a, status, 400, "task: Input should be a valid string");
        var r: ming.GenerateRequest = .{
            .task = text.strip(task_v.string),
            .text = txt,
            .language = optStr(o, "language", &perr, a),
            .prompt_wav_path = optStr(o, "prompt_wav_path", &perr, a),
            .prompt_text = optStr(o, "prompt_text", &perr, a),
            .voice = optStr(o, "voice", &perr, a),
            .instruction = optStr(o, "instruction", &perr, a),
            .seed = seed,
        };
        if (perr) |e| return errJson(a, status, e.status, e.msg);
        if (o.get("controls")) |cv| switch (cv) {
            .null => {},
            .object => |co| {
                var unknown: std.ArrayList([]const u8) = .empty;
                var cit = co.iterator();
                while (cit.next()) |kv| {
                    if (kv.value_ptr.* != .string) return errJson(a, status, 400, try std.fmt.allocPrint(a, "controls.{s}: Input should be a valid string", .{kv.key_ptr.*}));
                    const v = text.strip(kv.value_ptr.string);
                    if (v.len == 0) continue;
                    const k = kv.key_ptr.*;
                    if (std.mem.eql(u8, k, "speed")) r.controls.speed = v else if (std.mem.eql(u8, k, "pitch")) r.controls.pitch = v else if (std.mem.eql(u8, k, "volume")) r.controls.volume = v else if (std.mem.eql(u8, k, "emotion")) r.controls.emotion = v else if (std.mem.eql(u8, k, "dialect")) r.controls.dialect = v else if (std.mem.eql(u8, k, "style")) r.controls.style = v else if (std.mem.eql(u8, k, "ip")) r.controls.ip = v else try unknown.append(a, k);
                }
                r.unknown_controls = unknown.items;
            },
            else => return errJson(a, status, 400, "controls: Input should be a valid dictionary"),
        };
        if (r.prompt_wav_path) |p| {
            r.prompt_wav_path = checkPrompt(st.io, a, p, status) catch |e| switch (e) {
                error.Reported => return a.dupe(u8, last_prompt_error),
                else => return e,
            };
        }
        var d: ming.Diag = .{};
        _ = ming.mapGenerate(r, &d) catch |e| switch (e) {
            error.BadRequest => return errJson(a, status, 400, d.msg()),
            else => return e,
        };
        job.req = r;
    }
    if (!st.loaded.load(.acquire)) return errJson(a, status, 503, st.load_error orelse "model is loading; poll /health");
    submit(st, job);
    if (job.err) |e| return errJson(a, status, e.status, e.msg);
    const samples = job.samples.?;
    defer st.a.free(samples);
    if (samples.len == 0) return errJson(a, status, 500, "model produced no audio");
    for (samples) |*x| x.* = std.math.clamp(x.*, -1.0, 1.0);
    const wav_path = try writeWav(st.io, a, st.cfg.output_dir, samples);
    const dur = @as(f64, @floatFromInt(samples.len)) / @as(f64, @floatFromInt(ming.SAMPLE_RATE));
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"wav_path\":");
    try text.jsonStr(a, &out, wav_path);
    try out.print(a, ",\"sample_rate\":{d},\"duration_s\":{d},\"elapsed_s\":{d},\"rtf\":{d}}}", .{ ming.SAMPLE_RATE, round3(dur), round3(job.elapsed_s), round3(job.elapsed_s / dur) });
    return out.toOwnedSlice(a);
}

fn round3(x: f64) f64 {
    return @round(x * 1000) / 1000;
}

fn strList(a: Allocator, out: *std.ArrayList(u8), items: []const []const u8) !void {
    try out.append(a, '[');
    for (items, 0..) |x, k| {
        if (k > 0) try out.append(a, ',');
        try text.jsonStr(a, out, x);
    }
    try out.append(a, ']');
}

threadlocal var last_prompt_error: []const u8 = "";

/// server/app.py `_check_prompt` for WAV files.
fn checkPrompt(io: std.Io, a: Allocator, path: []const u8, status: *u16) ![]const u8 {
    const fail = struct {
        fn f(al: Allocator, st: *u16, msg: []const u8) error{ Reported, OutOfMemory } {
            last_prompt_error = errJson(al, st, 400, msg) catch return error.OutOfMemory;
            return error.Reported;
        }
    }.f;
    if (!std.fs.path.isAbsolute(path)) return fail(a, status, "prompt_wav_path must be an absolute path");
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return fail(a, status, try std.fmt.allocPrint(a, "prompt_wav_path is not a file: {s}", .{path}));
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var r = f.reader(io, &rb);
    const bytes = r.interface.allocRemaining(a, .limited(512 << 20)) catch return fail(a, status, try std.fmt.allocPrint(a, "prompt_wav_path is not a file: {s}", .{path}));
    const d = @import("wav.zig").decode(a, bytes) catch |e| return fail(a, status, try std.fmt.allocPrint(a, "cannot read audio {s}: {s} (WAV PCM16/24 or float32 only)", .{ path, @errorName(e) }));
    const frames = d.pcm.len / d.channels;
    if (frames == 0) return fail(a, status, "prompt audio is empty");
    const dur = @as(f64, @floatFromInt(frames)) / @as(f64, @floatFromInt(d.sample_rate));
    if (dur > ming.MAX_PROMPT_SECONDS) return fail(a, status, try std.fmt.allocPrint(a, "prompt audio is {d:.1}s; limit is {d:.0}s", .{ dur, ming.MAX_PROMPT_SECONDS }));
    return path;
}

const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };
extern "c" fn localtime_r(t: *const c_long, out: *Tm) ?*Tm;
extern "c" fn time(t: ?*c_long) c_long;
extern "c" fn arc4random() u32;

/// outputs/YYYYmmdd-HHMMSS-xxxx.wav, 16-bit PCM mono (like the Python server).
fn writeWav(io: std.Io, a: Allocator, dir: []const u8, samples: []const f32) ![]u8 {
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    const now = time(null);
    var tm: Tm = undefined;
    _ = localtime_r(&now, &tm);
    const path = try std.fmt.allocPrint(a, "{s}/{d:0>4}{d:0>2}{d:0>2}-{d:0>2}{d:0>2}{d:0>2}-{x:0>4}.wav", .{
        dir, @as(u32, @intCast(tm.year + 1900)), @as(u32, @intCast(tm.mon + 1)), @as(u32, @intCast(tm.mday)), @as(u32, @intCast(tm.hour)), @as(u32, @intCast(tm.min)), @as(u32, @intCast(tm.sec)), arc4random() & 0xffff,
    });
    const bytes = try @import("wav.zig").encodePcm16Mono(a, samples, ming.SAMPLE_RATE);
    const f = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
    return path;
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
    const bytes = try @import("wav.zig").encodePcm16Mono(a, samples, ming.SAMPLE_RATE);
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
