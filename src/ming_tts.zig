//! Ming-3.0-tts-0.5b engine: the full request pipeline and the in-process
//! Zig API. Mirrors `ming_engine_mlx/core.py` (`MingMLX.synthesize`) and the
//! request mapping of the Python server (`server/engine.py`).
//!
//! In-process use (no HTTP):
//!
//!     var eng = try ming_tts.Engine.init(io, gpa, mlx.mlx_default_gpu_stream_new(),
//!         "/path/to/Ming-3.0-tts-0.5b", .{});
//!     defer eng.deinit();
//!     var diag: ming_tts.Diag = .{};
//!     const pcm = try eng.generate(.{ .task = "IP", .text = "Hello.", .voice = "UK_George", .seed = 0 }, &diag);
//!     defer gpa.free(pcm); // mono f32 at ming_tts.SAMPLE_RATE (44.1 kHz)
//!
//! `error.BadRequest` carries the upstream message in `diag.msg()`. An Engine
//! is not thread-safe: call it from one thread (MLX work must stay on it).

const std = @import("std");
const mlx = @import("mlx.zig");
pub const model = @import("ming_model.zig");
pub const text = @import("ming_text.zig");
pub const audio = @import("ming_audio.zig");
pub const campplus = @import("ming_campplus.zig");
const tokenizer_mod = @import("tokenizer.zig");
const wav_mod = @import("wav.zig");

pub const Diag = text.Diag;
pub const SAMPLE_RATE = model.SAMPLE_RATE;
const Arr = model.Arr;
const G = model.G;

pub const MAX_PROMPT_SECONDS: f32 = 20.0;
pub const MAX_TEXT_CHARS: usize = 1000;
pub const V2_TASKS = [_][]const u8{ "TTS", "MULTILINGUAL", "BASIC", "EMOTION", "DIALECT", "IP", "FREEFORM", "VOICE_DESIGN" };

/// One /generate request (API.md v2). Strings are borrowed.
pub const GenerateRequest = struct {
    task: []const u8,
    text: []const u8,
    language: ?[]const u8 = null,
    prompt_wav_path: ?[]const u8 = null,
    prompt_text: ?[]const u8 = null,
    voice: ?[]const u8 = null,
    instruction: ?[]const u8 = null,
    controls: text.Controls = .{},
    /// Control keys the request named that `Controls` has no field for.
    unknown_controls: []const []const u8 = &.{},
    seed: ?u64 = 0,
};

pub const Engine = struct {
    a: std.mem.Allocator,
    io: std.Io,
    s: model.S,
    model: *model.Model,
    tok: tokenizer_mod.Tokenizer,
    speaker: *campplus.Campplus,

    pub fn init(io: std.Io, a: std.mem.Allocator, s: model.S, model_dir: []const u8, opts: model.Options) !Engine {
        const m = try model.Model.load(a, s, model_dir, opts);
        errdefer m.deinit();
        var tok = try tokenizer_mod.loadTokenizerAny(io, a, model_dir);
        errdefer tok.deinit();
        try validateTokenizer(a, &tok);
        const cp_path = try std.fmt.allocPrint(a, "{s}/campplus.safetensors", .{model_dir});
        defer a.free(cp_path);
        // CAM++ runs on the CPU stream: Metal fp32 convolutions drift ~3e-3
        // from onnxruntime, enough to change a bf16 clone's trajectory.
        const sp = try campplus.Campplus.load(a, mlx.mlx_default_cpu_stream_new(), cp_path);
        return .{ .a = a, .io = io, .s = s, .model = m, .tok = tok, .speaker = sp };
    }

    pub fn deinit(e: *Engine) void {
        e.speaker.deinit();
        e.tok.deinit();
        e.model.deinit();
    }

    // ── API.md request mapping (server/engine.py synthesize_kwargs) ──

    /// Map a /generate body to synthesize() and run it. Returns mono f32
    /// samples at 44.1 kHz (owned). Field rules: API.md v2 table.
    pub fn generate(e: *Engine, r: GenerateRequest, d: *Diag) ![]f32 {
        const in = try mapGenerate(r, d);
        return e.synthesize(in, r.seed);
    }

    /// POST /tts: built-in voice, voice design, or the default voice.
    pub fn tts(e: *Engine, txt: []const u8, voice: ?[]const u8, instruction: ?[]const u8, seed: ?u64, d: *Diag) ![]f32 {
        var in: text.TaskInput = .{ .task = "IP", .text = txt, .use_zero_spk_emb = true };
        if (voice) |v| {
            if (!text.isVoice(v)) return d.set("unknown voice '{s}'; see /voices", .{v});
            in.controls.ip = v;
        } else if (instruction) |ins| {
            in.task = "FREEFORM";
            in.instruction = ins;
        } else {
            // The model has no neutral speaker; plain TTS uses a default built-in voice.
            in.controls.ip = if (isChineseUtf8(txt)) "灵小柔" else "UK_George";
        }
        return e.synthesizeChecked(in, seed, d);
    }

    /// POST /clone: zero-shot clone from a reference clip (+ transcript).
    pub fn clone(e: *Engine, txt: []const u8, prompt_wav_path: []const u8, prompt_text: ?[]const u8, seed: ?u64, d: *Diag) ![]f32 {
        return e.synthesizeChecked(.{
            .task = "TTS",
            .text = txt,
            .language_id = if (isChineseUtf8(txt)) "Chinese" else "English",
            .prompt_wav_path = prompt_wav_path,
            .prompt_text = prompt_text,
            .use_spk_emb = true,
        }, seed, d);
    }

    fn synthesizeChecked(e: *Engine, in: text.TaskInput, seed: ?u64, d: *Diag) ![]f32 {
        // Validate first so the 400 carries the upstream message.
        const req = try text.buildRequest(e.a, in, d);
        req.deinit(e.a);
        return e.synthesize(in, seed);
    }

    // ── MingMLX.synthesize ──

    /// Vendor `MingTTS.synthesize` kwargs → waveform (owned, 44.1 kHz).
    /// Call `text.buildRequest` first if you need the 400 message; here a bad
    /// request is `error.BadRequest`.
    pub fn synthesize(e: *Engine, in: text.TaskInput, seed: ?u64) ![]f32 {
        const a = e.a;
        if (seed) |sd| try mlx.check(mlx.mlx_random_seed(sd));
        var d: Diag = .{};
        const req = try text.buildRequest(a, in, &d);
        defer req.deinit(a);

        var prompt_latent: ?Arr = null;
        defer if (prompt_latent) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var spk: ?[]f32 = null; // 192 floats
        defer if (spk) |x| a.free(x);
        if (req.use_zero_spk_emb) {
            spk = try a.alloc(f32, model.SPK_DIM);
            @memset(spk.?, 0);
        } else if (req.prompt_wav_path) |p| {
            const loaded = try audio.loadMono(e.io, a, p);
            defer a.free(loaded.pcm);
            if (loaded.pcm.len == 0) return error.EmptyPrompt;
            if (req.use_spk_emb) {
                const w16 = try audio.resample(a, loaded.pcm, loaded.sample_rate, 16000);
                defer a.free(w16);
                spk = try e.speaker.embed(a, w16);
            }
            const w44 = try audio.resample(a, loaded.pcm, loaded.sample_rate, SAMPLE_RATE);
            defer a.free(w44);
            prompt_latent = try e.encodePromptLatent(w44, .rng);
        }
        var prompt_text: ?[]const u8 = if (prompt_latent != null) req.prompt_text else null;
        if (prompt_text == null) if (prompt_latent) |x| {
            _ = mlx.mlx_array_free(x);
            prompt_latent = null;
        };

        var frags = try text.prepareText(a, req.text, req.text_language, 24);
        defer {
            for (frags.items) |f| a.free(f);
            frags.deinit(a);
        }
        var zero = req.use_zero_spk_emb;
        const reuse_first = req.taskname == .STYLE and frags.items.len > 1;
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(a);
        var reused_text: ?[]u8 = null;
        defer if (reused_text) |x| a.free(x);
        for (frags.items, 0..) |frag, i| {
            const chunk = try e.generateFragment(req, frag, spk, prompt_latent, prompt_text, zero);
            defer a.free(chunk);
            const trimmed = audio.trimTrailingSilence(chunk, SAMPLE_RATE);
            try out.appendSlice(a, trimmed);
            if (reuse_first and i == 0) {
                // VOICE_DESIGN/FREEFORM: later fragments clone the first one's voice.
                const w16 = try audio.resample(a, trimmed, SAMPLE_RATE, 16000);
                defer a.free(w16);
                if (spk) |x| a.free(x);
                spk = null;
                spk = try e.speaker.embed(a, w16);
                if (prompt_latent) |x| _ = mlx.mlx_array_free(x);
                prompt_latent = null;
                prompt_latent = try e.encodePromptLatent(trimmed, .rng);
                reused_text = try a.dupe(u8, frag);
                prompt_text = reused_text;
                zero = false;
            }
        }
        return out.toOwnedSlice(a);
    }

    /// core.generate: one fragment → untrimmed waveform (owned).
    fn generateFragment(e: *Engine, req: text.Request, frag: []const u8, spk: ?[]const f32, prompt_latent_in: ?Arr, prompt_text: ?[]const u8, zero: bool) ![]f32 {
        const a = e.a;
        const max_steps = @min(@as(usize, 1000), model.BASE_STEPS + model.STEPS_PER_UNIT * text.spokenUnits(frag));
        const pl: ?Arr = if (prompt_text == null) null else prompt_latent_in;
        var g = G.init(a, e.s);
        defer g.deinit();
        const spk_arr: ?Arr = if (spk) |x| try g.fromF32(x, &.{ 1, @intCast(model.SPK_DIM) }) else null;
        const view: RequestView = .{ .prompt = req.prompt, .text = frag, .instruction = req.instruction, .prompt_text = prompt_text, .use_zero_spk_emb = zero };
        const built = try e.buildInputs(&g, view, spk_arr, pl);
        defer a.free(built.ids);
        const lat = try model.sampleLatents(e.model, built.embeds, pl, .{ .max_steps = max_steps, .cfg = req.cfg, .sigma = req.sigma, .temperature = req.temperature }, .rng, null);
        _ = try g.keep(lat);
        return e.model.vaeDecode(&g, lat);
    }

    /// pad_prompt + encode_prompt_latent: 44.1 kHz audio → latents [1, T, 64] (owned).
    pub fn encodePromptLatent(e: *Engine, wav44: []const f32, noise: union(enum) { rng, given: Arr }) !Arr {
        const padded = try padPrompt(e.a, wav44);
        defer e.a.free(padded);
        var g = G.init(e.a, e.s);
        defer g.deinit();
        const x = try promptInput(&g, e.model, try g.fromF32(padded, &.{@intCast(padded.len)}));
        const h = try e.model.vaeEncodeH(&g, x);
        const half = @divExact(mlx.getShape(h)[2], 2);
        const shape = [_]c_int{ 1, mlx.getShape(h)[1], half };
        const nz = switch (noise) {
            .rng => try g.normal(&shape, mlx.mlx_array_dtype(h)),
            .given => |n| n,
        };
        const lat = try model.Model.vaePosterior(&g, h, nz);
        try model.evalAll(&.{lat});
        return model.own(lat);
    }

    pub const Built = struct { ids: []i32, embeds: Arr };

    /// core.build_inputs: token ids + embeddings with the speaker slot and
    /// prompt latents spliced in. `embeds` lives in `g`.
    pub fn buildInputs(e: *Engine, g: *G, r: RequestView, spk_emb: ?Arr, prompt_latent: ?Arr) !Built {
        const a = e.a;
        const m = e.model;
        var proj_spk: ?Arr = null;
        if (spk_emb) |se| {
            const p = try m.spk_head.f(g, try g.astype(try g.reshape(se, &.{ 1, -1 }), m.dtype));
            proj_spk = if (r.use_zero_spk_emb) try g.zeros(mlx.getShape(p), mlx.mlx_array_dtype(p)) else p;
        }
        var proj_lat: ?Arr = null;
        if (prompt_latent != null and r.prompt_text != null) {
            const pl = try g.astype(try g.reshape(prompt_latent.?, &.{ -1, 1, 64 }), m.dtype);
            const agg = try m.aggregate(g, pl); // [T, 1, 896]
            proj_lat = try g.reshape(agg, &.{ 1, -1, mlx.getShape(agg)[2] });
        }
        const steps: usize = if (proj_lat) |x| @intCast(mlx.getShape(x)[1]) else 0;
        const ii = try text.buildInputIds(a, &e.tok, r.prompt, r.text, if (proj_spk != null) 1 else 0, r.instruction, r.prompt_text, steps);
        defer a.free(ii.speaker_positions);
        errdefer a.free(ii.ids);
        var emb = try m.embedIds(g, ii.ids);
        const edt = mlx.mlx_array_dtype(emb);
        const L: c_int = @intCast(ii.ids.len);
        if (proj_spk) |ps| {
            const pos: c_int = @intCast(ii.speaker_positions[0]);
            const row = try g.astype(try g.reshape(ps, &.{ 1, 1, -1 }), edt);
            emb = try g.concat(&.{ try g.sliceAxis(emb, 1, 0, pos, 1), row, try g.sliceAxis(emb, 1, pos + 1, L, 1) }, 1);
        }
        if (proj_lat) |pl| {
            const lp: c_int = @intCast(ii.latent_position.?);
            emb = try g.concat(&.{ try g.sliceAxis(emb, 1, 0, lp, 1), try g.astype(pl, edt) }, 1);
        }
        return .{ .ids = ii.ids, .embeds = emb };
    }
};

/// The request fields `buildInputs` reads.
pub const RequestView = struct {
    prompt: []const u8,
    text: []const u8,
    instruction: ?[]const u8,
    prompt_text: ?[]const u8,
    use_zero_spk_emb: bool,
};

/// Front-pad to a multiple of 3528 samples (hop × VAE patch), owned.
pub fn padPrompt(a: std.mem.Allocator, w: []const f32) ![]f32 {
    const r = w.len % model.PAD_ALIGN;
    const pad = if (r == 0) 0 else model.PAD_ALIGN - r;
    const out = try a.alloc(f32, w.len + pad);
    @memset(out[0..pad], 0);
    @memcpy(out[pad..], w);
    return out;
}

/// Round to the model dtype like the reference (waveform.to(bf16)), then f32: [1, N].
pub fn promptInput(g: *G, m: *model.Model, w: Arr) !Arr {
    return g.astype(try g.astype(try g.reshape(w, &.{ 1, -1 }), m.dtype), .float32);
}

fn isChineseUtf8(s: []const u8) bool {
    var it = (std.unicode.Utf8View.init(s) catch return false).iterator();
    while (it.nextCodepoint()) |c| if (text.isCjk(c)) return true;
    return false;
}

/// prompt_contract.validate_tokenizer: the trained ids, each a single token.
fn validateTokenizer(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer) !void {
    const want = [_]struct { []const u8, u32 }{
        .{ "<|endoftext|>", 151643 },    .{ "<|im_start|>", 151644 },   .{ "<|im_end|>", 151645 },
        .{ "<|vision_start|>", 151652 }, .{ "<|vision_end|>", 151653 }, .{ "<|vision_pad|>", 151654 },
        .{ "<audioPatch>", 151705 },     .{ "<audio>", 151706 },
    };
    for (want) |w| {
        const ids = try tok.encode(a, w[0]);
        defer a.free(ids);
        if (ids.len != 1 or ids[0] != w[1]) {
            std.log.err("[ming] tokenizer does not match the Ming-3.0-tts dense contract: {s} -> {any} (want {d})", .{ w[0], ids, w[1] });
            return error.TokenizerMismatch;
        }
    }
}

/// server/engine.py synthesize_kwargs + the API.md v2 per-task field rules.
pub fn mapGenerate(r: GenerateRequest, d: *Diag) !text.TaskInput {
    var tb: [32]u8 = undefined;
    const t = text.strip(r.task);
    const n = @min(t.len, tb.len);
    for (t[0..n], 0..) |c, i| tb[i] = std.ascii.toUpper(c);
    const task = tb[0..n];
    var known = false;
    for (V2_TASKS) |x| if (std.mem.eql(u8, x, task)) {
        known = true;
    };
    if (!known) return d.set("task must be one of ['TTS', 'MULTILINGUAL', 'BASIC', 'EMOTION', 'DIALECT', 'IP', 'FREEFORM', 'VOICE_DESIGN']", .{});
    const has_controls = !r.controls.isEmpty() or r.unknown_controls.len > 0;
    const Field = struct { name: []const u8, present: bool };
    const fields = [_]Field{
        .{ .name = "prompt_wav_path", .present = r.prompt_wav_path != null },
        .{ .name = "prompt_text", .present = r.prompt_text != null },
        .{ .name = "language", .present = r.language != null },
        .{ .name = "voice", .present = r.voice != null },
        .{ .name = "instruction", .present = r.instruction != null },
        .{ .name = "controls", .present = has_controls },
    };
    const Rule = struct { task: []const u8, required: []const []const u8, forbidden: []const []const u8 };
    const rules = [_]Rule{
        .{ .task = "TTS", .required = &.{}, .forbidden = &.{ "voice", "instruction", "controls" } },
        .{ .task = "MULTILINGUAL", .required = &.{ "prompt_wav_path", "prompt_text", "language" }, .forbidden = &.{ "voice", "instruction", "controls" } },
        .{ .task = "BASIC", .required = &.{ "prompt_wav_path", "controls" }, .forbidden = &.{ "prompt_text", "language", "voice", "instruction" } },
        .{ .task = "EMOTION", .required = &.{ "prompt_wav_path", "controls" }, .forbidden = &.{ "prompt_text", "language", "voice", "instruction" } },
        .{ .task = "DIALECT", .required = &.{ "prompt_wav_path", "controls" }, .forbidden = &.{ "prompt_text", "language", "voice", "instruction" } },
        .{ .task = "IP", .required = &.{"voice"}, .forbidden = &.{ "prompt_wav_path", "prompt_text", "language", "instruction", "controls" } },
        .{ .task = "FREEFORM", .required = &.{"instruction"}, .forbidden = &.{ "prompt_wav_path", "prompt_text", "language", "voice", "controls" } },
        .{ .task = "VOICE_DESIGN", .required = &.{}, .forbidden = &.{ "prompt_wav_path", "prompt_text", "language", "voice" } },
    };
    for (rules) |rule| {
        if (!std.mem.eql(u8, rule.task, task)) continue;
        for (rule.required) |f| for (fields) |fd| if (std.mem.eql(u8, fd.name, f) and !fd.present) return d.set("{s} requires {s}", .{ rule.task, f });
        for (rule.forbidden) |f| for (fields) |fd| if (std.mem.eql(u8, fd.name, f) and fd.present) return d.set("{s} does not accept {s}", .{ rule.task, f });
    }
    const task_name: []const u8 = for (V2_TASKS) |x| {
        if (std.mem.eql(u8, x, task)) break x;
    } else unreachable;
    var in: text.TaskInput = .{ .task = task_name, .text = r.text, .language_id = r.language, .controls = r.controls, .unknown_controls = r.unknown_controls };
    if (std.mem.eql(u8, task, "IP") or std.mem.eql(u8, task, "FREEFORM") or std.mem.eql(u8, task, "VOICE_DESIGN")) in.use_zero_spk_emb = true;
    if (std.mem.eql(u8, task, "IP")) {
        if (!text.isVoice(r.voice.?)) return d.set("unknown voice '{s}'; see /voices", .{r.voice.?});
        in.controls = .{ .ip = r.voice };
    } else if (std.mem.eql(u8, task, "FREEFORM")) {
        in.instruction = r.instruction;
    } else if (std.mem.eql(u8, task, "VOICE_DESIGN")) {
        if (r.instruction != null and has_controls) return d.set("VOICE_DESIGN takes controls.style or instruction, not both", .{});
        if (r.instruction) |ins| in.controls = .{ .style = ins };
    } else if (r.prompt_wav_path) |p| {
        in.prompt_wav_path = p;
        in.prompt_text = r.prompt_text;
        in.use_spk_emb = true;
    }
    // Run the upstream validation now so callers get its message.
    var tmp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer tmp_arena.deinit();
    _ = try text.buildRequest(tmp_arena.allocator(), in, d);
    return in;
}
