//! Ming-3.0-tts as an in-process service: the API.md routes (/health,
//! /voices, /capabilities, /tts, /clone, /generate) as a function from
//! (method, path, JSON body) to (status, JSON body), backed by one engine
//! thread. `ming-tts serve` puts HTTP in front of it; an app can call
//! `Service.handle` directly and skip the network.
//!
//!     const svc = try ming_service.Service.start(io, gpa, .{ .model_dir = dir });
//!     var status: u16 = 0;
//!     const json = try svc.handle(arena, "POST", "/generate", body, &status);
//!
//! The model loads in the background; `/health` reports `"loaded"`, and
//! synthesis answers 503 until then. All MLX work runs on the engine thread
//! (MLX streams are per-thread); `handle` may be called from any thread and
//! blocks until its request is done. Requests run one at a time.
//!
//! For the typed API (no JSON), use `engine` (ming_tts.zig) directly.

const std = @import("std");
const mlx = @import("mlx.zig");
pub const engine = @import("ming_tts.zig");
const ming = engine;
const text = ming.text;
const wav = @import("wav.zig");

const Allocator = std.mem.Allocator;

/// Model directory used when neither `Config.model_dir` nor `MING_MODEL_DIR`
/// is set, relative to the current directory.
pub const DEFAULT_MODEL = "models/Ming-3.0-tts-0.5b";

/// `$MING_MODEL_DIR`, else `DEFAULT_MODEL`.
pub fn defaultModelDir() []const u8 {
    if (std.c.getenv("MING_MODEL_DIR")) |p| return std.mem.span(p);
    return DEFAULT_MODEL;
}

/// `$MING_OUTPUT_DIR`, else `$HOME/Library/Caches/ming-tts`, else
/// `$TMPDIR/ming-tts`, else `/tmp/ming-tts`.
pub fn defaultOutputDir(a: Allocator) ![]const u8 {
    if (std.c.getenv("MING_OUTPUT_DIR")) |p| return a.dupe(u8, std.mem.span(p));
    if (std.c.getenv("HOME")) |h| return std.fmt.allocPrint(a, "{s}/Library/Caches/ming-tts", .{std.mem.span(h)});
    const tmp: []const u8 = if (std.c.getenv("TMPDIR")) |t| std.mem.trimEnd(u8, std.mem.span(t), "/") else "/tmp";
    return std.fmt.allocPrint(a, "{s}/ming-tts", .{tmp});
}

pub const Config = struct {
    /// Empty: `defaultModelDir()`. A relative path is resolved against the
    /// current directory when the service starts.
    model_dir: []const u8 = "",
    /// Generated WAVs are written here (created if missing). Empty:
    /// `defaultOutputDir()`.
    output_dir: []const u8 = "",
    max_text_chars: usize = ming.MAX_TEXT_CHARS,
    /// `engine` field of /health.
    engine_name: []const u8 = "mlx",
};

pub const Service = struct {
    io: std.Io,
    a: Allocator,
    cfg: Config,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*Job) = .empty,
    loaded: std.atomic.Value(bool) = .init(false),
    load_error: ?[]u8 = null,
    /// Model load plus warm-up time, set when `loaded` turns true.
    load_s: f64 = 0,

    /// Allocate the service and start its engine thread (loads the model).
    /// The service lives for the rest of the process.
    pub fn start(io: std.Io, a: Allocator, cfg: Config) !*Service {
        mlx.installErrorHandler();
        var c = cfg;
        if (c.model_dir.len == 0) c.model_dir = defaultModelDir();
        // The engine opens files by absolute path; a missing dir is reported by /health.
        c.model_dir = std.Io.Dir.cwd().realPathFileAlloc(io, c.model_dir, a) catch c.model_dir;
        if (c.output_dir.len == 0) c.output_dir = try defaultOutputDir(a);
        // Responses carry absolute WAV paths, so resolve a relative dir once.
        std.Io.Dir.cwd().createDirPath(io, c.output_dir) catch {};
        c.output_dir = std.Io.Dir.cwd().realPathFileAlloc(io, c.output_dir, a) catch c.output_dir;
        const st = try a.create(Service);
        st.* = .{ .io = io, .a = a, .cfg = c };
        const th = try std.Thread.spawn(.{}, worker, .{st});
        th.detach();
        return st;
    }

    pub fn isLoaded(self: *Service) bool {
        return self.loaded.load(.acquire);
    }

    /// Route one API request. Returns the JSON response body (allocated in
    /// `a`) and sets `status` to the HTTP status code.
    pub fn handle(self: *Service, a: Allocator, method: []const u8, path: []const u8, body: []const u8, status: *u16) ![]u8 {
        return self.handleStream(a, method, path, body, status, null);
    }

    /// `handle`, plus streaming for /tts, /clone and /generate: `sink` gets
    /// each text fragment's audio on the engine thread as soon as it is
    /// generated, before the response (and WAV) for the whole text is ready.
    pub fn handleStream(self: *Service, a: Allocator, method: []const u8, path: []const u8, body: []const u8, status: *u16, sink: ?ming.Sink) ![]u8 {
        status.* = 200;
        return route(self, a, method, path, body, status, sink);
    }
};

const State = Service;

pub const ApiError = struct {
    status: u16,
    msg: []u8,
};

pub const Body = std.json.ObjectMap;

/// Optional string field: wrong type → 400; blank → null (stripped).
pub fn optStr(o: Body, k: []const u8, err: *?ApiError, a: Allocator) ?[]const u8 {
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

pub fn seedOf(o: Body, err: *?ApiError, a: Allocator) u64 {
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

pub const Longform = struct { pause_scale: ?f32 = null };

/// `pause_scale` (number, 0 to MAX_PAUSE_SCALE); null or absent: engine default.
pub fn longformOf(o: Body, err: *?ApiError, a: Allocator) Longform {
    var r: Longform = .{};
    if (o.get("pause_scale")) |v| switch (v) {
        .null => {},
        .integer => |n| r.pause_scale = @floatFromInt(n),
        .float => |x| r.pause_scale = @floatCast(x),
        else => err.* = .{ .status = 400, .msg = std.fmt.allocPrint(a, "pause_scale: Input should be a valid number", .{}) catch "" },
    };
    if (r.pause_scale) |x| if (!(x >= 0 and x <= ming.MAX_PAUSE_SCALE)) {
        err.* = .{ .status = 400, .msg = std.fmt.allocPrint(a, "pause_scale must be between 0 and {d}", .{ming.MAX_PAUSE_SCALE}) catch "" };
    };
    return r;
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
    sink: ?ming.Sink = null,
};

fn worker(st: *State) void {
    const a = st.a;
    const t0 = std.Io.Timestamp.now(st.io, .awake);
    const s = mlx.mlx_default_gpu_stream_new();
    var eng = ming.Engine.init(st.io, a, s, st.cfg.model_dir, .{}) catch |e| {
        st.load_error = std.fmt.allocPrint(a, "cannot load model from {s}: {s}", .{ st.cfg.model_dir, @errorName(e) }) catch null;
        std.log.err("[ming] model load failed: {s}", .{@errorName(e)});
        return;
    };
    defer eng.deinit();
    configure(&eng);
    // Compile Metal kernels once so the first request is not slow.
    var d: ming.Diag = .{};
    if (eng.tts("Warm up.", "UK_George", null, 0, &d)) |w| {
        defer a.free(w);
        warmClone(st.io, a, &eng, w);
    } else |_| {}
    st.load_s = @as(f64, @floatFromInt(t0.untilNow(st.io, .awake).nanoseconds)) / 1e9;
    st.loaded.store(true, .release);
    std.log.info("[ming] model loaded in {d:.2}s", .{st.load_s});
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

/// A/B switches for the deliberate departures from upstream ("0" turns one
/// off): MING_PROMPT_NORM, MING_SOFT_LIMIT, MING_GUARD, MING_FR_TN, MING_JOIN,
/// MING_CLEAR_CACHE, MING_LANG_DETECT.
pub fn configure(eng: *ming.Engine) void {
    const off = struct {
        fn f(name: [*:0]const u8) ?bool {
            const v = std.c.getenv(name) orelse return null;
            return std.mem.eql(u8, std.mem.span(v), "0");
        }
    }.f;
    if (off("MING_PROMPT_NORM")) |x| eng.normalize_prompt = !x;
    if (off("MING_SOFT_LIMIT")) |x| eng.soft_limit = !x;
    if (off("MING_GUARD")) |x| eng.guard = !x;
    if (off("MING_JOIN")) |x| eng.join = !x;
    if (off("MING_CLEAR_CACHE")) |x| eng.clear_cache = !x;
    configureText();
}

/// The text-only switches (also read by `ming-tts frontend`, which loads no
/// engine): MING_FR_TN, and MING_LANG_DETECT ("0", "text" or "prompt").
pub fn configureText() void {
    if (std.c.getenv("MING_FR_TN")) |v| ming.text.french_frontend = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("MING_LANG_DETECT")) |v| ming.text.detect_language = std.meta.stringToEnum(ming.text.DetectLanguage, std.mem.span(v)) orelse
        if (std.mem.eql(u8, std.mem.span(v), "0")) .off else ming.text.detect_language;
}

/// Run one clone on `samples` so the prompt path (resampler, CAM++, VAE
/// encoder kernels) is compiled before the first real clone request.
fn warmClone(io: std.Io, a: Allocator, eng: *ming.Engine, samples: []const f32) void {
    const dir = if (std.c.getenv("TMPDIR")) |t| std.mem.span(t) else "/tmp";
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/ming-warmup-{d}.wav", .{ std.mem.trimEnd(u8, dir, "/"), std.c.getpid() }) catch return;
    const bytes = wav.encodePcm16Mono(a, samples, ming.SAMPLE_RATE) catch return;
    defer a.free(bytes);
    const f = std.Io.Dir.createFileAbsolute(io, path, .{}) catch return;
    f.writeStreamingAll(io, bytes) catch {};
    f.close(io);
    defer std.Io.Dir.deleteFileAbsolute(io, path) catch {};
    var d: ming.Diag = .{};
    if (eng.clone("Warm up.", path, null, 0, &d)) |w| a.free(w) else |_| {}
}

fn runJob(eng: *ming.Engine, job: *Job) void {
    const ja = job.arena.allocator();
    var d: ming.Diag = .{};
    const t0 = std.Io.Timestamp.now(eng.io, .awake);
    const r = job.req;
    eng.on_chunk = job.sink;
    defer eng.on_chunk = null;
    const saved = eng.useOptions(r);
    defer eng.restoreOptions(saved);
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

fn errJson(a: Allocator, status: *u16, code: u16, msg: []const u8) ![]u8 {
    status.* = code;
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"error\":");
    try text.jsonStr(a, &out, msg);
    try out.append(a, '}');
    return out.toOwnedSlice(a);
}

fn route(st: *State, a: Allocator, method: []const u8, path: []const u8, body: []const u8, status: *u16, sink: ?ming.Sink) ![]u8 {
    const get = std.mem.eql(u8, method, "GET");
    const post = std.mem.eql(u8, method, "POST");
    if (std.mem.eql(u8, path, "/health")) {
        if (!get) return errJson(a, status, 405, "Method Not Allowed");
        var out: std.ArrayList(u8) = .empty;
        const name = std.fs.path.basename(st.cfg.model_dir);
        try out.appendSlice(a, if (st.load_error != null) "{\"status\":\"error\",\"engine\":" else "{\"status\":\"ok\",\"engine\":");
        try text.jsonStr(a, &out, st.cfg.engine_name);
        try out.appendSlice(a, ",\"model\":");
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
        try out.print(a, ",\"max_prompt_s\":{d:.1},\"max_text_chars\":{d}", .{ ming.MAX_PROMPT_SECONDS, st.cfg.max_text_chars });
        try out.print(a, ",\"pauses_s\":{{\"clause\":{d},\"sentence\":{d},\"paragraph\":{d}}},\"max_pause_scale\":{d}}}", .{ ming.PAUSE_CLAUSE_S, ming.PAUSE_SENTENCE_S, ming.PAUSE_PARAGRAPH_S, ming.MAX_PAUSE_SCALE });
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
    job.* = .{ .kind = .generate, .arena = std.heap.ArenaAllocator.init(st.a), .sink = sink };
    defer job.arena.deinit();
    const seed = seedOf(o, &perr, a);
    const lf = longformOf(o, &perr, a);
    if (perr) |e| return errJson(a, status, e.status, e.msg);
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
    job.req.pause_scale = lf.pause_scale;
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
    const d = wav.decode(a, bytes) catch |e| return fail(a, status, try std.fmt.allocPrint(a, "cannot read audio {s}: {s} (WAV PCM16/24 or float32 only)", .{ path, @errorName(e) }));
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
    const bytes = try wav.encodePcm16Mono(a, samples, ming.SAMPLE_RATE);
    const f = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
    return path;
}
