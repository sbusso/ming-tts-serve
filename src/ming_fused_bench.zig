//! GPU timing for the fused DiT kernels vs the MLX graph.
//!
//!   ming-fused-bench <model_dir> [iters]
//!   ming-fused-bench spk <model_dir> <ref.wav> <a.wav>...
//!
//! `spk` prints, per WAV: CAM++ cosine similarity to ref.wav, duration, RMS
//! and peak (dBFS), and the count of non-finite samples.
//!
//! Times one compiled CFM sample (10 Euler steps × CFG DiT) per iteration
//! with MING_FUSED on and off, then each fused kernel alone (80 dispatches
//! per graph, as in one LM step), and prints the max |diff| of the CFM output.
const std = @import("std");
const mlx = @import("mlx.zig");
const ming = @import("ming_model.zig");
const fused = @import("ming_fused.zig");
const audio = @import("ming_audio.zig");
const campplus = @import("ming_campplus.zig");

const Arr = ming.Arr;

fn now(io: std.Io) std.Io.Timestamp {
    return std.Io.Timestamp.now(io, .awake);
}

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e6;
}

fn cfm(m: *ming.Model, z: Arr, hist: Arr, noise: Arr, sde: Arr) !Arr {
    var g = ming.G.init(m.a, m.s);
    defer g.deinit();
    const y = try m.cfmSample(&g, z, hist, noise, sde, 2.0, 0.25, 0.0);
    try ming.evalAll(&.{y});
    return ming.own(y);
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer it.deinit();
    _ = it.next();
    var dir = it.next() orelse return error.Usage;
    if (std.mem.eql(u8, dir, "lmcheck")) {
        // bf16 fused and bf16 MLX LM decode vs an f32 copy of the model.
        dir = it.next() orelse return error.Usage;
        mlx.installErrorHandler();
        const s = mlx.mlx_default_gpu_stream_new();
        const mb = try ming.Model.load(a, s, dir, .{});
        defer mb.deinit();
        const mf = try ming.Model.load(a, s, dir, .{ .cast_to = .float32 });
        defer mf.deinit();
        var g = ming.G.init(a, s);
        defer g.deinit();
        const pre = try g.normal(&.{ 1, 200, 896 }, .float32);
        const e1 = try g.normal(&.{ 1, 1, 896 }, .float32);
        const steps: usize = if (std.c.getenv("LMCHECK_STEPS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 8;
        var outs: [3]Arr = undefined;
        for (0..3) |k| {
            const m = if (k == 2) mf else mb;
            const dt: mlx.mlx_dtype = if (k == 2) .float32 else .bfloat16;
            _ = setenv("MING_FUSED", if (k == 0) "1" else "0", 1);
            var cache = try ming.KvCache.init(a, m.lm.layers.len);
            defer cache.deinit();
            try ming.evalAll(&.{try m.lmForward(&g, try g.astype(pre, dt), &cache)});
            var x = try g.astype(e1, dt);
            for (0..steps) |_| x = try m.lmForward(&g, x, &cache); // decode steps, fed back
            outs[k] = try g.astype(x, .float32);
            try ming.evalAll(&.{outs[k]});
        }
        for ([_][]const u8{ "fused bf16", "MLX bf16" }, 0..) |name, k| {
            const dv = try ming.toF32(a, try g.abs(try g.sub(outs[k], outs[2])), s);
            defer a.free(dv);
            const rv = try ming.toF32(a, outs[2], s);
            defer a.free(rv);
            var mx: f32 = 0;
            var at: usize = 0;
            var sum: f64 = 0;
            var big: usize = 0;
            for (dv, 0..) |e, i| {
                if (e > mx) {
                    mx = e;
                    at = i;
                }
                sum += e;
                if (e > 0.05) big += 1;
            }
            std.debug.print("{s} vs f32 after {d} steps: max|d| {e:.3} at {d} (ref {d:.3}), mean|d| {e:.3}, >0.05: {d}\n", .{ name, steps, mx, at, rv[at], sum / @as(f64, @floatFromInt(dv.len)), big });
        }
        return;
    }
    if (std.mem.eql(u8, dir, "spk")) {
        dir = it.next() orelse return error.Usage;
        const cp_path = try std.fmt.allocPrint(a, "{s}/campplus.safetensors", .{dir});
        defer a.free(cp_path);
        const cp = try campplus.Campplus.load(a, mlx.mlx_default_cpu_stream_new(), cp_path);
        defer cp.deinit();
        var ref: ?[]f32 = null;
        defer if (ref) |r| a.free(r);
        while (it.next()) |path| {
            const w = try audio.loadMono(io, a, path);
            defer a.free(w.pcm);
            var bad: usize = 0;
            var ss: f64 = 0;
            var peak: f32 = 0;
            for (w.pcm) |v| {
                if (!std.math.isFinite(v)) {
                    bad += 1;
                    continue;
                }
                ss += v * v;
                peak = @max(peak, @abs(v));
            }
            const w16 = try audio.resample(a, w.pcm, w.sample_rate, 16000);
            defer a.free(w16);
            const e = try cp.embed(a, w16);
            if (ref == null) {
                ref = e;
                continue;
            }
            defer a.free(e);
            var dot: f64 = 0;
            var na: f64 = 0;
            var nb: f64 = 0;
            for (ref.?, e) |x, y| {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            const n: f64 = @floatFromInt(w.pcm.len);
            std.debug.print("{s}\tsim {d:.4}\tdur {d:.2}s\trms {d:.1} dBFS\tpeak {d:.1} dBFS\tnonfinite {d}\n", .{
                path,                                dot / @sqrt(na * nb),                        n / @as(f64, @floatFromInt(w.sample_rate)),
                10 * std.math.log10(ss / n + 1e-20), 20 * std.math.log10(@as(f64, peak) + 1e-20), bad,
            });
        }
        return;
    }
    const iters: usize = if (it.next()) |n| try std.fmt.parseInt(usize, n, 10) else 50;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    const m = try ming.Model.load(a, s, dir, .{});
    defer m.deinit();

    var g0 = ming.G.init(a, s);
    defer g0.deinit();
    const z = try g0.normal(&.{ 1, 1, 896 }, .bfloat16);
    const hist = try g0.normal(&.{ 1, 1, 64 }, .bfloat16);
    const noise = try g0.normal(&.{ 1, 1, 64 }, .bfloat16);
    const sde = try g0.normal(&.{ 10, 1, 1, 64 }, .bfloat16);
    try ming.evalAll(&.{ z, hist, noise, sde });

    var outs: [2]Arr = undefined;
    for ([_][]const u8{ "1", "0" }, 0..) |flag, k| {
        _ = setenv("MING_FUSED", @ptrCast(flag.ptr), 1);
        if (m.cfm_compiled) |c| _ = mlx.mlx_closure_free(c);
        m.cfm_compiled = null;
        outs[k] = try cfm(m, z, hist, noise, sde); // trace + warm
        for (0..3) |_| _ = mlx.mlx_array_free(try cfm(m, z, hist, noise, sde));
        const t0 = now(io);
        for (0..iters) |_| _ = mlx.mlx_array_free(try cfm(m, z, hist, noise, sde));
        std.debug.print("cfm MING_FUSED={s}: {d:.3} ms/sample\n", .{ flag, ms(io, t0) / @as(f64, @floatFromInt(iters)) });
        // CPU side: graph build + async_eval (encode) vs waiting for the GPU.
        var enc: f64 = 0;
        var wait: f64 = 0;
        for (0..iters) |_| {
            var g = ming.G.init(m.a, m.s);
            defer g.deinit();
            const ta = now(io);
            const y = try m.cfmSample(&g, z, hist, noise, sde, 2.0, 0.25, 0.0);
            try ming.asyncEvalAll(&.{y});
            enc += ms(io, ta);
            const tb = now(io);
            try ming.evalAll(&.{y});
            wait += ms(io, tb);
        }
        const fi: f64 = @floatFromInt(iters);
        std.debug.print("   encode {d:.3} ms, then wait {d:.3} ms\n", .{ enc / fi, wait / fi });
    }
    const d = try g0.abs(try g0.sub(try g0.astype(outs[0], .float32), try g0.astype(outs[1], .float32)));
    const v = try ming.toF32(a, d, s);
    defer a.free(v);
    var mx: f32 = 0;
    for (v) |e| mx = @max(mx, e);
    const r = try ming.toF32(a, outs[1], s);
    defer a.free(r);
    var rm: f32 = 0;
    for (r) |e| rm = @max(rm, @abs(e));
    std.debug.print("cfm out max|fused-mlx| = {e:.3} (max|mlx| {e:.3})\n", .{ mx, rm });

    // LM decode step (MLX graph) at a 256-position cache, and the aggregator.
    {
        var cache = try ming.KvCache.init(a, m.lm.layers.len);
        defer cache.deinit();
        {
            var g = ming.G.init(a, s);
            defer g.deinit();
            const pre = try g.normal(&.{ 1, 256, 896 }, .bfloat16);
            try ming.evalAll(&.{try m.lmForward(&g, pre, &cache)});
        }
        const e1 = try g0.normal(&.{ 1, 1, 896 }, .bfloat16);
        const lat = try g0.normal(&.{ 1, 1, 64 }, .bfloat16);
        try ming.evalAll(&.{ e1, lat });
        // Fused vs MLX decode of one token on the same cache.
        var dec: [2]Arr = undefined;
        for ([_][*:0]const u8{ "1", "0" }, 0..) |flag, k| {
            _ = setenv("MING_FUSED", flag, 1);
            var g = ming.G.init(a, s);
            defer g.deinit();
            dec[k] = try ming.own(try m.lmForward(&g, e1, &cache));
            try ming.evalAll(&.{dec[k]});
            cache.len -= 1;
            for (cache.k, cache.v) |*kk, *vv| {
                const k2 = try g.sliceAxis(kk.*, 2, 0, cache.len, 1);
                const v2 = try g.sliceAxis(vv.*, 2, 0, cache.len, 1);
                _ = mlx.mlx_array_free(kk.*);
                _ = mlx.mlx_array_free(vv.*);
                kk.* = try ming.own(k2);
                vv.* = try ming.own(v2);
            }
        }
        {
            const dd = try g0.abs(try g0.sub(try g0.astype(dec[0], .float32), try g0.astype(dec[1], .float32)));
            const dv = try ming.toF32(a, dd, s);
            defer a.free(dv);
            const rv = try ming.toF32(a, dec[1], s);
            defer a.free(rv);
            var dmx: f32 = 0;
            var drm: f32 = 0;
            for (dv, rv) |e, rr| {
                dmx = @max(dmx, e);
                drm = @max(drm, @abs(rr));
            }
            std.debug.print("LM decode max|fused-mlx| = {e:.3} (max|mlx| {e:.3})\n", .{ dmx, drm });
            for (dec) |dx| _ = mlx.mlx_array_free(dx);
        }
        for ([_][*:0]const u8{ "1", "0" }) |flag| for ([_][]const u8{ "LM decode", "aggregate" }, 0..) |name, wi| {
            _ = setenv("MING_FUSED", flag, 1);
            var best: f64 = 1e9;
            var enc: f64 = 1e9;
            for (0..20) |_| {
                var g = ming.G.init(a, s);
                defer g.deinit();
                const t0 = now(io);
                const y = if (wi == 0) try m.lmForward(&g, e1, &cache) else try m.aggregate(&g, lat);
                try ming.asyncEvalAll(&.{y});
                enc = @min(enc, ms(io, t0));
                try ming.evalAll(&.{y});
                best = @min(best, ms(io, t0));
                if (wi == 0) { // drop the appended position again
                    cache.len -= 1;
                    for (cache.k, cache.v) |*kk, *vv| {
                        const k2 = try g.sliceAxis(kk.*, 2, 0, cache.len, 1);
                        const v2 = try g.sliceAxis(vv.*, 2, 0, cache.len, 1);
                        _ = mlx.mlx_array_free(kk.*);
                        _ = mlx.mlx_array_free(vv.*);
                        kk.* = try ming.own(k2);
                        vv.* = try ming.own(v2);
                    }
                }
            }
            std.debug.print("{s:<12} MING_FUSED={s} {d:.3} ms (encode {d:.3})\n", .{ name, flag, best, enc });
        };
        _ = setenv("MING_FUSED", "1", 1);
    }
    // Per-kernel GPU time: N dispatches of one kernel on fixed inputs, one eval.
    _ = setenv("MING_FUSED", "1", 1);
    const b = &m.dit.fb[0];
    const x = try g0.normal(&.{ 2, 3, 1024 }, .bfloat16);
    const blk = try fused.block(b, x, m.dit.rot.tab[0], m.dit.rot.tab[1], s);
    const parts = [_]Arr{ blk.qkv, blk.x1, blk.hh, blk.x2 };
    try ming.evalAll(&parts);
    defer for (parts) |q| {
        _ = mlx.mlx_array_free(q);
    };
    const names = [_][]const u8{ "A qkv 6MB", "B attn+out 2MB", "C ff1 8MB", "D ff2 8MB" };
    const mb = [_]f64{ 6, 2, 8, 8 };
    const N = 200;
    for (names, 0..) |name, ki| {
        var best: f64 = 1e9;
        for (0..3) |_| {
            var g = ming.G.init(a, s);
            defer g.deinit();
            var ys: [N]Arr = undefined;
            // Round-robin over all 16 DiT blocks (384 MB): weights come from DRAM.
            for (&ys, 0..) |*y, yi| {
                const bb = if (yi % 16 < 8) &m.dit.fb[yi % 8] else &m.agg.fb[yi % 8];
                y.* = try g.keep(try fused.single(ki, bb, x, blk.qkv, blk.hh, m.dit.rot.tab[0], m.dit.rot.tab[1], s));
            }
            const t0 = now(io);
            try ming.evalAll(&ys);
            best = @min(best, ms(io, t0) * 1000.0 / N);
        }
        std.debug.print("{s:<16} {d:6.1} us  {d:5.0} GB/s\n", .{ name, best, mb[ki] * 1.048576e6 / (best * 1e3) });
    }
    // Dependent chains (each dispatch waits for the previous): 0 = whole
    // block, 1 = B alone (x = B(qkv, x)), 3 = D alone (x = D(hh, x)).
    for ([_]usize{ 0, 1, 3, 4, 5 }) |ki| {
        var best: f64 = 1e9;
        for (0..3) |_| {
            var g = ming.G.init(a, s);
            defer g.deinit();
            var y = x;
            for (0..80) |yi| {
                const hot = std.c.getenv("BENCH_HOT") != null;
                const bb = if (hot) &m.dit.fb[0] else if (yi % 16 < 8) &m.dit.fb[yi % 8] else &m.agg.fb[yi % 8];
                if (ki == 0) {
                    const o = try fused.block(bb, y, m.dit.rot.tab[0], m.dit.rot.tab[1], s);
                    for ([_]Arr{ o.qkv, o.x1, o.hh }) |q| _ = try g.keep(q);
                    y = try g.keep(o.x2);
                } else if (ki == 4) { // A then B
                    const q = try g.keep(try fused.single(0, bb, y, blk.qkv, blk.hh, m.dit.rot.tab[0], m.dit.rot.tab[1], s));
                    y = try g.keep(try fused.single(1, bb, y, q, blk.hh, m.dit.rot.tab[0], m.dit.rot.tab[1], s));
                } else if (ki == 5) { // C then D
                    const h = try g.keep(try fused.single(2, bb, y, blk.qkv, blk.hh, m.dit.rot.tab[0], m.dit.rot.tab[1], s));
                    y = try g.keep(try fused.single(3, bb, y, blk.qkv, h, m.dit.rot.tab[0], m.dit.rot.tab[1], s));
                } else y = try g.keep(try fused.single(ki, bb, y, blk.qkv, blk.hh, m.dit.rot.tab[0], m.dit.rot.tab[1], s));
            }
            const t0 = now(io);
            try ming.evalAll(&.{y});
            best = @min(best, ms(io, t0) * 1000.0 / 80);
        }
        std.debug.print("chain {s:<10} {d:6.1} us\n", .{ ([_][]const u8{ "block", "B", "", "D", "AB", "CD" })[ki], best });
    }
    // Floor: a chain of 80 dependent tiny MLX adds.
    {
        var best: f64 = 1e9;
        for (0..3) |_| {
            var g = ming.G.init(a, s);
            defer g.deinit();
            var y = try g.keep(try ming.own(x));
            const one = try g.scalar(1, .bfloat16);
            for (0..80) |_| y = try g.add(y, one);
            const t0 = now(io);
            try ming.evalAll(&.{y});
            best = @min(best, ms(io, t0) * 1000.0 / 80);
        }
        std.debug.print("chain {s:<10} {d:6.1} us\n", .{ "tiny add", best });
    }
    // Reference: MLX addmm for the same ff1 shape.
    {
        const h = try g0.normal(&.{ 6, 1024 }, .bfloat16);
        try ming.evalAll(&.{h});
        var best: f64 = 1e9;
        for (0..3) |_| {
            var g = ming.G.init(a, s);
            defer g.deinit();
            var ys: [N]Arr = undefined;
            for (&ys, 0..) |*y, yi| y.* = try (if (yi % 16 < 8) m.dit.blocks[yi % 8] else m.agg.blocks[yi % 8]).ff1.f(&g, h);
            const t0 = now(io);
            try ming.evalAll(&ys);
            best = @min(best, ms(io, t0) * 1000.0 / N);
        }
        std.debug.print("{s:<16} {d:6.1} us  {d:5.0} GB/s\n", .{ "MLX addmm ff1", best, 8 * 1.048576e6 / (best * 1e3) });
    }
    for (outs) |o| _ = mlx.mlx_array_free(o);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
