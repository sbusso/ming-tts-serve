//! Parity check: native Ming-3.0-tts vs the Python MLX engine.
//!
//!   ming-parity <model_dir> <ref_dir> [f32|bf16]
//!
//! `ref_dir` comes from `engine-mlx/zig_ref_dump.py` (f32 = fp32 weights on
//! the CPU stream, bf16 = shipped dtypes on the GPU). Each module is fed the
//! reference input of that module, so errors do not compound, except in the
//! free-running rows. Exit status is non-zero when a gated row fails.
//! MING_PARITY_STEPS=1 also prints the free run's z diff for steps 0-5.

const std = @import("std");
const mlx = @import("mlx.zig");
const ming = @import("ming_model.zig");
const engine = @import("ming_tts.zig");

const Arr = ming.Arr;
var failed: usize = 0;

fn maxAbs(a: std.mem.Allocator, s: ming.S, x: Arr) !f32 {
    const v = try ming.toF32(a, x, s);
    defer a.free(v);
    var m: f32 = 0;
    for (v) |e| m = @max(m, @abs(e));
    return m;
}

fn report(a: std.mem.Allocator, s: ming.S, name: []const u8, got: Arr, want: Arr, tol: f32) !void {
    const gs = mlx.getShape(got);
    const ws = mlx.getShape(want);
    var gn: usize = 1;
    for (gs) |d| gn *= @intCast(d);
    var wn: usize = 1;
    for (ws) |d| wn *= @intCast(d);
    if (gn != wn) {
        std.debug.print("{s:<36} SHAPE got {any} want {any}  FAIL\n", .{ name, gs, ws });
        failed += 1;
        return;
    }
    const gv = try ming.toF32(a, got, s);
    defer a.free(gv);
    const wv = try ming.toF32(a, want, s);
    defer a.free(wv);
    var d: f32 = 0;
    var r: f32 = 0;
    for (gv, wv) |x, y| {
        d = @max(d, @abs(x - y));
        r = @max(r, @abs(y));
    }
    const scale = if (r == 0) 1 else r;
    const ok = d <= tol * scale;
    if (!ok) failed += 1;
    std.debug.print("{s:<36} max|d|={e:.3}  max|ref|={e:.3}  rel={e:.2}  {s}\n", .{ name, d, r, d / scale, if (ok) "ok" else "FAIL" });
}

fn corr(x: []const f32, y: []const f32) f64 {
    const n = @min(x.len, y.len);
    var mx_: f64 = 0;
    var my: f64 = 0;
    for (0..n) |i| {
        mx_ += x[i];
        my += y[i];
    }
    mx_ /= @floatFromInt(n);
    my /= @floatFromInt(n);
    var sxy: f64 = 0;
    var sxx: f64 = 0;
    var syy: f64 = 0;
    for (0..n) |i| {
        const dx = x[i] - mx_;
        const dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    return sxy / @sqrt(sxx * syy);
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer it.deinit();
    _ = it.next();
    const model_dir = it.next() orelse return usage();
    const ref_dir = it.next() orelse return usage();
    const mode = it.next() orelse "f32";
    ref_wav = it.next() orelse if (std.c.getenv("MING_REF_WAV")) |p| std.mem.span(p) else return usage();
    const f32_mode = std.mem.eql(u8, mode, "f32");
    mlx.installErrorHandler();

    const s = if (f32_mode) mlx.mlx_default_cpu_stream_new() else mlx.mlx_default_gpu_stream_new();
    if (f32_mode) {
        const dev = mlx.mlx_device_new_type(.cpu, 0);
        _ = mlx.mlx_set_default_device(dev);
    }
    // Relative to max|ref|: fp32 CPU agrees to rounding; bf16 keeps ~3 digits per op.
    const tol: f32 = if (f32_mode) 1e-4 else 1e-1;
    std.debug.print("mode={s} tol={e}\n", .{ mode, tol });

    // The fused kernels are not bit-exact (ming_fused.zig); with them the
    // free run can stop at a different step than the reference dump.
    @import("ming_fused.zig").force_off = true;
    var eng = try engine.Engine.init(init.io, a, s, model_dir, .{ .cast_to = if (f32_mode) .float32 else null });
    // Parity is against upstream, which neither normalizes the reference nor limits the output.
    eng.normalize_prompt = false;
    eng.soft_limit = false;
    eng.guard = false;
    engine.text.french_frontend = false;
    defer eng.deinit();
    const m = eng.model;

    // Text frontend + token ids (frontend.json).
    try checkFrontend(init.io, a, &eng, ref_dir);

    for ([_][]const u8{ "ip", "clone" }) |case| {
        std.debug.print("== {s}\n", .{case});
        const path = try std.fmt.allocPrint(a, "{s}/{s}.safetensors", .{ ref_dir, case });
        defer a.free(path);
        var ref = try ming.Tensors.load(a, path, null, s);
        defer ref.deinit();
        var g = ming.G.init(a, s);
        defer g.deinit();

        // Request → token ids → embeddings (speaker slot / prompt latents spliced).
        const input = caseInput(case);
        var diag: engine.Diag = .{};
        const req = try engine.text.buildRequest(a, input, &diag);
        defer req.deinit(a);
        if (ref.has("prompt_wav_44k", .{})) {
            // Prompt audio: resample, pad, CAM++, VAE encoder, posterior.
            const raw = try ming.toF32(a, try ref.get("prompt_raw", .{}), s);
            defer a.free(raw);
            const sr_arr = try ming.toF32(a, try ref.get("prompt_sr", .{}), s);
            defer a.free(sr_arr);
            const sr: u32 = @intFromFloat(sr_arr[0]);
            const w16 = try engine.audio.resample(a, raw, sr, 16000);
            defer a.free(w16);
            try report(a, s, "resample ->16k", try g.fromF32(w16, &.{@intCast(w16.len)}), try ref.get("wav_16k", .{}), 1e-4);
            const fb = try engine.audio.fbank(a, w16);
            defer a.free(fb);
            try report(a, s, "kaldi fbank (80 bins)", try g.fromF32(fb, &.{ @intCast(fb.len / 80), 80 }), try ref.get("fbank", .{}), 1e-4);
            const spk = try eng.speaker.embed(a, w16);
            defer a.free(spk);
            const spk_arr = try g.fromF32(spk, &.{ 1, 192 });
            try report(a, s, "CAM++ speaker embedding", spk_arr, try ref.get("spk_emb", .{}), 1e-3);
            // The same row after the model-dtype cast and spk_head (what the LM sees).
            const p_own = try m.spk_head.f(&g, try g.astype(spk_arr, m.dtype));
            const p_ref = try m.spk_head.f(&g, try g.astype(try ref.get("spk_emb", .{}), m.dtype));
            try report(a, s, "spk_head(own CAM++) vs spk_head(ref)", p_own, p_ref, 1e10);
            const w44 = try engine.audio.resample(a, raw, sr, ming.SAMPLE_RATE);
            defer a.free(w44);
            const padded = try engine.padPrompt(a, w44);
            defer a.free(padded);
            try report(a, s, "resample ->44.1k + front pad", try g.fromF32(padded, &.{@intCast(padded.len)}), try ref.get("prompt_wav_44k", .{}), 1e-4);
            const wav_in = try engine.promptInput(&g, m, try ref.get("prompt_wav_44k", .{}));
            const h = try m.vaeEncodeH(&g, wav_in);
            try report(a, s, "VAE encoder h", h, try ref.get("enc_h", .{}), tol);
            const lat = try ming.Model.vaePosterior(&g, h, try ref.get("prompt_noise", .{}));
            try report(a, s, "VAE posterior latent", lat, try ref.get("prompt_latent", .{}), tol);
        }
        const view: engine.RequestView = .{ .prompt = req.prompt, .text = req.text, .instruction = req.instruction, .prompt_text = req.prompt_text, .use_zero_spk_emb = req.use_zero_spk_emb };
        const built = try eng.buildInputs(&g, view, if (ref.has("spk_emb", .{})) try ref.get("spk_emb", .{}) else null, if (ref.has("prompt_latent", .{})) try ref.get("prompt_latent", .{}) else null);
        defer a.free(built.ids);
        {
            const want_ids = try ming.toF32(a, try ref.get("input_ids", .{}), s);
            defer a.free(want_ids);
            var same = want_ids.len == built.ids.len;
            if (same) for (built.ids, want_ids) |x, y| {
                if (@as(f32, @floatFromInt(x)) != y) same = false;
            };
            if (!same) failed += 1;
            std.debug.print("{s:<36} {s} ({d} tokens, ref {d})\n", .{ "token ids", if (same) "exact" else "FAIL", built.ids.len, want_ids.len });
        }
        try report(a, s, "input embeds (spk/latent spliced)", built.embeds, try ref.get("inputs_embeds", .{}), tol);

        const ref_embeds = try ref.get("inputs_embeds", .{});
        const z_ref = try ref.get("z", .{});
        const bank = try ref.get("flow_noise", .{});
        const lat_ref = try ref.get("latents", .{});
        const dt = m.dtype;
        // Teacher-forced LM prefill.
        var cache = try ming.KvCache.init(a, m.lm.layers.len);
        defer cache.deinit();
        const hidden = try m.lmForward(&g, try g.astype(ref_embeds, dt), &cache);
        const Lh = mlx.getShape(hidden)[1];
        try report(a, s, "LM prefill hidden (step 0)", try g.sliceAxis(hidden, 1, Lh - 1, Lh, 1), try g.sliceAxis(z_ref, 0, 0, 1, 1), tol);
        const z0 = try g.astype(try g.reshape(try g.sliceAxis(z_ref, 0, 0, 1, 1), &.{ 1, 1, 896 }), dt);
        const hist0 = if (ref.has("prompt_latent", .{})) blk: {
            const pl = try ref.get("prompt_latent", .{});
            const T = mlx.getShape(pl)[1];
            break :blk try g.astype(try g.sliceAxis(pl, 1, T - 1, T, 1), dt);
        } else try g.zeros(&.{ 1, 1, 64 }, dt);
        const row0 = try g.sliceAxis(bank, 0, 0, 1, 1);
        const n0 = try g.astype(try g.reshape(try g.sliceAxis(row0, 1, 0, 1, 1), &.{ 1, 1, 64 }), dt);
        const t0 = try g.scalar(0, dt);
        try report(a, s, "DiT forward_with_cfg (step 0)", try m.ditCfg(&g, n0, t0, z0, hist0), try ref.get("dit0", .{}), tol);
        const sde0 = try g.astype(try g.reshape(try g.sliceAxis(row0, 1, 1, 11, 1), &.{ 10, 1, 1, 64 }), dt);
        try report(a, s, "CFM sample, 10 steps (step 0)", try m.cfmSample(&g, z0, hist0, n0, sde0, 2.0, 0.25, 0.0), try ref.get("cfm0", .{}), tol);
        const l0 = try g.astype(try g.sliceAxis(lat_ref, 1, 0, 1, 1), dt);
        try report(a, s, "aggregator (linear_proj_audio)", try m.aggregate(&g, l0), try ref.get("agg0", .{}), tol);
        const wav = try m.vaeDecode(&g, try g.astype(lat_ref, dt));
        defer a.free(wav);
        try report(a, s, "VAE decoder waveform", try g.fromF32(wav, &.{@intCast(wav.len)}), try ref.get("audio", .{}), tol);

        // Free-running, replaying the reference noise.
        var trace: ming.Trace = .{};
        defer trace.deinit(a);
        const pl_ref: ?Arr = if (ref.has("prompt_latent", .{})) try g.astype(try ref.get("prompt_latent", .{}), dt) else null;
        const lat = try ming.sampleLatents(m, try g.astype(ref_embeds, dt), pl_ref, .{}, .{ .replay = bank }, &trace);
        _ = try g.keep(lat);
        const n_ref: usize = @intCast(mlx.getShape(lat_ref)[1]);
        const n_got: usize = @intCast(mlx.getShape(lat)[1]);
        std.debug.print("{s:<36} ref={d} zig={d}  {s}\n", .{ "decode steps (stop head)", n_ref, n_got, if (n_ref == n_got) "ok" else "DIFF" });
        const n = @min(n_ref, n_got);
        var zmax: f32 = 0;
        for (0..n) |i| {
            const zi = try g.reshape(trace.z.items[i], &.{ 1, 896 });
            const zr = try g.sliceAxis(z_ref, 0, @intCast(i), @intCast(i + 1), 1);
            const d = try g.abs(try g.sub(try g.astype(zi, .float32), zr));
            const zd = try maxAbs(a, s, d);
            if (std.c.getenv("MING_PARITY_STEPS") != null and i < 6) std.debug.print("  z step {d}: max|d|={e:.3}\n", .{ i, zd });
            zmax = @max(zmax, zd);
        }
        const stop_ref = try ming.toF32(a, try ref.get("stop_prob", .{}), s);
        defer a.free(stop_ref);
        var smax: f32 = 0;
        for (0..n) |i| smax = @max(smax, @abs(trace.stop.items[i] - stop_ref[i]));
        const lmax = try maxAbs(a, s, try g.sub(try g.astype(try g.sliceAxis(lat, 1, 0, @intCast(n), 1), .float32), try g.sliceAxis(lat_ref, 1, 0, @intCast(n), 1)));
        std.debug.print("{s:<36} z max|d|={e:.3}  stop max|d|={e:.3}  latents max|d|={e:.3}\n", .{ "free-run per-step", zmax, smax, lmax });
        const wav2 = try m.vaeDecode(&g, lat);
        defer a.free(wav2);
        const aref = try ming.toF32(a, try ref.get("audio", .{}), s);
        defer a.free(aref);
        const c = corr(wav2, aref);
        std.debug.print("{s:<36} len ref={d} zig={d}  corr={d:.5}\n", .{ "free-run audio", aref.len, wav2.len, c });
        if (f32_mode and (n_ref != n_got or c < 0.999)) failed += 1;

        // Whole request with the engine's own RNG (same seed as the dump).
        const synth = try eng.synthesize(input, 0);
        defer a.free(synth);
        const sref = try ming.toF32(a, try ref.get("synth_audio", .{}), s);
        defer a.free(sref);
        std.debug.print("{s:<36} len ref={d} zig={d}  corr={d:.5}\n", .{ "synthesize() seeded RNG", sref.len, synth.len, corr(synth, sref) });
    }
    std.debug.print("\n{s}\n", .{if (failed == 0) "PASS" else "FAILED"});
    if (failed != 0) std.process.exit(1);
}

/// The voice-clone reference zig_ref_dump.py used (upstream's
/// data/voice_presets/voice_audio/lingguang.wav).
var ref_wav: []const u8 = "";
const REF_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？";

/// The requests zig_ref_dump.py CASES dumped.
fn caseInput(case: []const u8) engine.text.TaskInput {
    if (std.mem.eql(u8, case, "ip")) return .{ .task = "IP", .text = "Good evening, and welcome to the late news.", .controls = .{ .ip = "UK_George" }, .use_zero_spk_emb = true };
    return .{ .task = "TTS", .language_id = "English", .text = "The evening wind blew in from the mountain pass, and the clouds turned gold.", .prompt_wav_path = ref_wav, .prompt_text = REF_TEXT, .use_spk_emb = true };
}

/// frontend.json: prepare_text fragments and build_input_ids for every task.
fn checkFrontend(io: std.Io, a: std.mem.Allocator, eng: *engine.Engine, ref_dir: []const u8) !void {
    const path = try std.fmt.allocPrint(a, "{s}/frontend.json", .{ref_dir});
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var r = f.reader(io, &rb);
    const body = try r.interface.allocRemaining(a, .limited(16 << 20));
    defer a.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer parsed.deinit();
    std.debug.print("== text frontend\n", .{});
    var it = parsed.value.object.get("fragments").?.object.iterator();
    var ok_n: usize = 0;
    var total: usize = 0;
    while (it.next()) |kv| {
        total += 1;
        var frags = engine.text.prepareText(a, kv.key_ptr.*, null, 24) catch |err| {
            std.debug.print("prepare_text FAIL ({s}) {s}\n", .{ @errorName(err), kv.key_ptr.* });
            failed += 1;
            continue;
        };
        defer {
            for (frags.items) |x| a.free(x);
            frags.deinit(a);
        }
        const want = kv.value_ptr.array.items;
        var same = want.len == frags.items.len;
        if (same) for (want, frags.items) |w, g| {
            if (!std.mem.eql(u8, w.string, g)) same = false;
        };
        if (same) ok_n += 1 else {
            failed += 1;
            std.debug.print("prepare_text FAIL\n  in:   {s}\n  want:", .{kv.key_ptr.*});
            for (want) |w| std.debug.print(" [{s}]", .{w.string});
            std.debug.print("\n  got: ", .{});
            for (frags.items) |g| std.debug.print(" [{s}]", .{g});
            std.debug.print("\n", .{});
        }
    }
    std.debug.print("{s:<36} {d}/{d} texts identical\n", .{ "prepare_text fragments", ok_n, total });
    var tit = parsed.value.object.get("tokens").?.object.iterator();
    while (tit.next()) |kv| {
        const o = kv.value_ptr.object;
        const instr: ?[]const u8 = if (o.get("instruction").? == .string) o.get("instruction").?.string else null;
        const pt: ?[]const u8 = if (o.get("prompt_text")) |v| v.string else null;
        const steps: usize = if (o.get("latent_steps")) |v| @intCast(v.integer) else 0;
        const ii = try engine.text.buildInputIds(a, &eng.tok, o.get("prompt").?.string, o.get("text").?.string, @intCast(o.get("num_speakers").?.integer), instr, pt, steps);
        defer ii.deinit(a);
        const want = o.get("ids").?.array.items;
        var same = want.len == ii.ids.len;
        if (same) for (want, ii.ids) |w, g| {
            if (w.integer != g) same = false;
        };
        if (!same) failed += 1;
        std.debug.print("token ids {s:<26} {s} ({d} tokens)\n", .{ kv.key_ptr.*, if (same) "exact" else "FAIL", ii.ids.len });
        // The task builder must produce the same prompt and caption.
        if (tokenTaskInput(kv.key_ptr.*)) |in| {
            var diag: engine.Diag = .{};
            const req = try engine.text.buildRequest(a, in, &diag);
            defer req.deinit(a);
            const same_p = std.mem.eql(u8, req.prompt, o.get("prompt").?.string);
            const same_i = if (instr) |w| (req.instruction != null and std.mem.eql(u8, w, req.instruction.?)) else req.instruction == null;
            if (!(same_p and same_i)) {
                failed += 1;
                std.debug.print("  caption FAIL: got {?s}\n", .{req.instruction});
            }
        }
    }
}

/// zig_ref_dump.py TOKEN_REQUESTS
fn tokenTaskInput(name: []const u8) ?engine.text.TaskInput {
    const T = engine.text.TaskInput;
    const table = [_]struct { []const u8, T }{
        .{ "TTS", .{ .task = "TTS", .text = "Hello world.", .language_id = "en" } },
        .{ "MULTILINGUAL", .{ .task = "MULTILINGUAL", .text = "こんにちは、元気ですか。", .language_id = "ja" } },
        .{ "BASIC", .{ .task = "BASIC", .text = "今天天气很好。", .controls = .{ .speed = "快速", .pitch = "高", .volume = "中" } } },
        .{ "EMOTION", .{ .task = "EMOTION", .text = "I can't believe we won!", .controls = .{ .emotion = "高兴" } } },
        .{ "DIALECT", .{ .task = "DIALECT", .text = "今天天气很好。", .controls = .{ .dialect = "cantonese" } } },
        .{ "IP", .{ .task = "IP", .text = "Good evening.", .controls = .{ .ip = "灵小柔" } } },
        .{ "FREEFORM", .{ .task = "FREEFORM", .text = "Once upon a time.", .instruction = "A calm, low female voice." } },
        .{ "VOICE_DESIGN", .{ .task = "VOICE_DESIGN", .text = "Once upon a time.", .controls = .{ .style = "温柔的女声" } } },
    };
    for (table) |t| if (std.mem.eql(u8, t[0], name)) return t[1];
    return null;
}

fn usage() void {
    std.debug.print("usage: ming-parity <model_dir> <ref_dir> [f32|bf16] [ref.wav]  (or MING_REF_WAV=ref.wav)\n", .{});
}
