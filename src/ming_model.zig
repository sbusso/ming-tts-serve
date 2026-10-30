//! Native Ming-3.0-tts-0.5b (inclusionAI) on mlx-c. No Python.
//!
//! Port of mlx-audio's `tts/models/dense` + `bailingmm` modules and the
//! request layer of `ming_engine_mlx/core.py` (vendor `ming_tts.py`):
//!
//!   prompt ids → Qwen2.5-0.5B (24 layers, KV cache, 512 positions)
//!     per step: last hidden z ─┬─ stop head (softmax[1] > 0.5 after step 3)
//!                              └─ CFM: 10 Euler steps of an 8-layer DiT with CFG 2
//!                                  over [t+c, history latent, x] → one 64-d latent
//!     next input = aggregator(latent) (8-layer DiT stack, CLS row → 896)
//!   latents (12.5 Hz) → VAE decoder (×4 upsample, Qwen2 24 layers, sliding
//!     window 64, ISTFT head n_fft 3528 / hop 882) → 44.1 kHz waveform
//!
//! Prompt audio for cloning goes through the VAE encoder (Qwen2 24 + 4 layer
//! patch aggregator, posterior std = softplus(scale) + 1e-4) and CAM++
//! (`ming_campplus.zig`). Text frontend and prompt layout: `ming_text.zig`.
//!
//! Every op mirrors the Python graph (same op order, same weak-scalar dtypes,
//! `addmm` for biased linears), so fp32-on-CPU runs match the Python MLX engine
//! to rounding and bf16 runs stay within bf16 noise. `src/ming_parity.zig`
//! holds the check.

const std = @import("std");
const mlx = @import("mlx.zig");

pub const Arr = mlx.mlx_array;
pub const S = mlx.mlx_stream;
pub const Dtype = mlx.mlx_dtype;

extern "c" fn mlx_addmm(res: *Arr, c: Arr, a: Arr, b: Arr, alpha: f32, beta: f32, s: S) c_int;
extern "c" fn mlx_power(res: *Arr, a: Arr, b: Arr, s: S) c_int;

pub const SAMPLE_RATE: u32 = 44100;
const HIDDEN: c_int = 896; // LM / VAE backbone width
const LATENT: c_int = 64;
const DIT_HIDDEN: c_int = 1024;
const DIT_HEADS: c_int = 16;
const DIT_HEAD_DIM: c_int = 64;
const HOP: usize = 882;
const N_FFT: usize = 3528;
const VAE_PATCH: usize = 4;
pub const SPK_DIM: usize = 192;
/// Vendor modeling_bailingmm.MAX_CACHE_LEN: prompt + generated latents.
pub const MAX_CACHE_LEN: usize = 512;
pub const FLOW_STEPS: usize = 10;
/// Runaway-decode cap (mlx-audio PR #1059): 2 s plus ~1 s per spoken unit.
pub const BASE_STEPS: usize = 25;
pub const STEPS_PER_UNIT: usize = 12;
/// Front-pad alignment for prompt audio: hop × VAE patch × DiT patch.
pub const PAD_ALIGN: usize = HOP * VAE_PATCH;

// ─────────────────────────────────────────────────────────────────────────
// Graph scope: every op result is parked here and freed together, so model
// code reads like the Python it mirrors. `own` copies a handle out of scope.
// ─────────────────────────────────────────────────────────────────────────

pub const G = struct {
    s: S,
    a: std.mem.Allocator,
    live: std.ArrayList(Arr) = .empty,
    /// Inside an mlx_compile trace: no nested compiled closures.
    in_trace: bool = false,

    pub fn init(a: std.mem.Allocator, s: S) G {
        return .{ .a = a, .s = s };
    }

    pub fn deinit(g: *G) void {
        for (g.live.items) |x| _ = mlx.mlx_array_free(x);
        g.live.deinit(g.a);
    }

    pub fn keep(g: *G, x: Arr) !Arr {
        g.live.append(g.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    fn ret(g: *G, rc: c_int, o: Arr) !Arr {
        if (rc != 0) {
            _ = mlx.mlx_array_free(o);
            try mlx.check(rc);
        }
        return g.keep(o);
    }

    pub fn add(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_add(&o, x, y, g.s), o);
    }
    pub fn sub(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_subtract(&o, x, y, g.s), o);
    }
    pub fn mul(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_multiply(&o, x, y, g.s), o);
    }
    pub fn div(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_divide(&o, x, y, g.s), o);
    }
    pub fn minimum(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_minimum(&o, x, y, g.s), o);
    }
    pub fn matmul(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_matmul(&o, x, y, g.s), o);
    }
    pub fn addmm(g: *G, c: Arr, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx_addmm(&o, c, x, y, 1.0, 1.0, g.s), o);
    }
    pub fn power(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx_power(&o, x, y, g.s), o);
    }
    pub fn exp(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_exp(&o, x, g.s), o);
    }
    pub fn cos(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_cos(&o, x, g.s), o);
    }
    pub fn sin(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_sin(&o, x, g.s), o);
    }
    pub fn tanh(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_tanh(&o, x, g.s), o);
    }
    pub fn abs(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_abs(&o, x, g.s), o);
    }
    pub fn sigmoid(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_sigmoid(&o, x, g.s), o);
    }
    pub fn logaddexp(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_logaddexp(&o, x, y, g.s), o);
    }
    pub fn floor(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_floor(&o, x, g.s), o);
    }
    pub fn softmax(g: *G, x: Arr, axis: c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_softmax_axis(&o, x, axis, false, g.s), o);
    }
    pub fn astype(g: *G, x: Arr, dt: Dtype) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_astype(&o, x, dt, g.s), o);
    }
    pub fn reshape(g: *G, x: Arr, shape: []const c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, g.s), o);
    }
    pub fn transpose(g: *G, x: Arr, axes: []const c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, g.s), o);
    }
    pub fn take(g: *G, x: Arr, idx: Arr, axis: c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_take_axis(&o, x, idx, axis, g.s), o);
    }
    pub fn rms(g: *G, x: Arr, w: Arr, eps: f32) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_fast_rms_norm(&o, x, w, eps, g.s), o);
    }
    pub fn clip(g: *G, x: Arr, lo: Arr, hi: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_clip(&o, x, lo, hi, g.s), o);
    }
    pub fn concat(g: *G, xs: []const Arr, axis: c_int) !Arr {
        const v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_concatenate_axis(&o, v, axis, g.s), o);
    }
    pub fn stack(g: *G, xs: []const Arr, axis: c_int) !Arr {
        const v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_stack_axis(&o, v, axis, g.s), o);
    }
    /// x[start:stop:step] along one axis, everything else whole.
    pub fn sliceAxis(g: *G, x: Arr, axis: usize, start: c_int, stop: c_int, step: c_int) !Arr {
        const sh = mlx.getShape(x);
        var st: [8]c_int = @splat(0);
        var sp: [8]c_int = undefined;
        var str: [8]c_int = @splat(1);
        for (sh, 0..) |d, i| sp[i] = d;
        st[axis] = start;
        sp[axis] = stop;
        str[axis] = step;
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_slice(&o, x, &st, sh.len, &sp, sh.len, &str, sh.len, g.s), o);
    }
    /// Python-style weak scalar: a constant in `dt`, so `x * 0.5` keeps x's dtype.
    pub fn scalar(g: *G, v: f32, dt: Dtype) !Arr {
        const f = mlx.mlx_array_new_float(v);
        defer _ = mlx.mlx_array_free(f);
        return g.astype(f, dt);
    }
    pub fn zeros(g: *G, shape: []const c_int, dt: Dtype) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, g.s), o);
    }
    pub fn fromF32(g: *G, data: []const f32, shape: []const c_int) !Arr {
        return g.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
    }
    pub fn fromI32(g: *G, data: []const i32, shape: []const c_int) !Arr {
        return g.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .int32));
    }
    pub fn normal(g: *G, shape: []const c_int, dt: Dtype) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, 1, .{ .ctx = null }, g.s), o);
    }
    pub fn split2(g: *G, x: Arr, axis: c_int) ![2]Arr {
        var v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_split(&v, x, 2, axis, g.s));
        var out: [2]Arr = undefined;
        for (0..2) |i| {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&o, v, i));
            out[i] = try g.keep(o);
        }
        return out;
    }
    pub fn sdpa(g: *G, q: Arr, k: Arr, v: Arr, scale: f32, mode: [*:0]const u8, mask: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, mask, .{ .ctx = null }, false, g.s), o);
    }
    pub fn rope(g: *G, x: Arr, dims: c_int, traditional: bool, base: f32, offset: c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_fast_rope(&o, x, dims, traditional, mlx.mlx_optional_float.some(base), 1.0, offset, .{ .ctx = null }, g.s), o);
    }
    pub fn silu(g: *G, x: Arr) !Arr {
        return g.mul(x, try g.sigmoid(x));
    }
};

pub fn own(x: Arr) !Arr {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, x));
    return o;
}

pub fn evalAll(xs: []const Arr) !void {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
    try mlx.check(mlx.mlx_eval(v));
}

/// Evaluate `x` and copy it out as f32 (caller frees).
pub fn toF32(a: std.mem.Allocator, x: Arr, s: S) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, x, .float32, s));
    var cc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cc);
    try mlx.check(mlx.mlx_contiguous(&cc, c, false, s));
    try mlx.check(mlx.mlx_array_eval(cc));
    const n = mlx.mlx_array_size(cc);
    const p = mlx.mlx_array_data_float32(cc) orelse return error.NoData;
    return a.dupe(f32, p[0..n]);
}

fn itemF32(x: Arr, s: S) !f32 {
    if (mlx.mlx_array_dtype(x) == .float32) {
        try mlx.check(mlx.mlx_array_eval(x));
        var v: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&v, x));
        return v;
    }
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(c));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, c));
    return v;
}

// ─────────────────────────────────────────────────────────────────────────
// Weights
// ─────────────────────────────────────────────────────────────────────────

/// All tensors of one safetensors file, keyed by name. Owns the handles.
pub const Tensors = struct {
    map: std.StringHashMap(Arr),
    a: std.mem.Allocator,
    extra: std.ArrayList(Arr) = .empty, // derived views (transposed linears)

    pub fn load(a: std.mem.Allocator, path: []const u8, cast_to: ?Dtype, s: S) !Tensors {
        var self: Tensors = .{ .map = std.StringHashMap(Arr).init(a), .a = a };
        errdefer self.deinit();
        const pz = try a.dupeSentinel(u8, path, 0);
        defer a.free(pz);
        var tm = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tm);
        var mm = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(mm);
        // Loading is a CPU-only primitive; casts and views then run on `s`.
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        try mlx.check(mlx.mlx_load_safetensors(&tm, &mm, pz.ptr, cpu));
        const it = mlx.mlx_map_string_to_array_iterator_new(tm);
        defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
        while (true) {
            var key: ?[*:0]const u8 = null;
            var v = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, it) != 0 or key == null) {
                _ = mlx.mlx_array_free(v);
                break;
            }
            if (cast_to) |dt| {
                if (mlx.mlx_array_dtype(v) != dt and mlx.mlx_array_dtype(v) != .int32) {
                    var c = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&c, v, dt, s));
                    _ = mlx.mlx_array_free(v);
                    v = c;
                }
            }
            const k = try a.dupe(u8, std.mem.span(key.?));
            try self.map.put(k, v);
        }
        return self;
    }

    pub fn deinit(self: *Tensors) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            self.a.free(e.key_ptr.*);
        }
        self.map.deinit();
        for (self.extra.items) |x| _ = mlx.mlx_array_free(x);
        self.extra.deinit(self.a);
    }

    pub fn get(self: *const Tensors, comptime fmt: []const u8, args: anytype) !Arr {
        var buf: [256]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, fmt, args);
        return self.map.get(k) orelse {
            std.log.err("[ming] missing weight {s}", .{k});
            return error.MissingWeight;
        };
    }

    pub fn has(self: *const Tensors, comptime fmt: []const u8, args: anytype) bool {
        var buf: [256]u8 = undefined;
        const k = std.fmt.bufPrint(&buf, fmt, args) catch return false;
        return self.map.contains(k);
    }

    /// Linear `<prefix>.weight` [out, in] (+ optional `.bias`). The weight is
    /// kept as a transposed VIEW, like `x @ W.T` in mlx.nn.Linear.
    pub fn lin(self: *Tensors, s: S, comptime fmt: []const u8, args: anytype) !Lin {
        const w = try self.get(fmt ++ ".weight", args);
        var wt = mlx.mlx_array_new();
        const perm = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wt, w, &perm, 2, s));
        try self.extra.append(self.a, wt);
        const b: ?Arr = if (self.has(fmt ++ ".bias", args)) try self.get(fmt ++ ".bias", args) else null;
        return .{ .wt = wt, .b = b };
    }
};

pub const Lin = struct {
    wt: Arr,
    b: ?Arr,

    pub fn f(l: Lin, g: *G, x: Arr) !Arr {
        if (l.b) |b| return g.addmm(b, x, l.wt);
        return g.matmul(x, l.wt);
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Qwen2 stack (LM backbone, VAE encoder/aggregator/decoder)
// ─────────────────────────────────────────────────────────────────────────

const QLayer = struct {
    ln1: Arr,
    ln2: Arr,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
};

const QStack = struct {
    layers: []QLayer,
    norm: Arr,
    heads: c_int = 14,
    kv_heads: c_int = 2,
    head_dim: c_int = 64,
    eps: f32 = 1e-6,
    theta: f32 = 1e6,
    /// Sliding-window size (VAE backbones); null = plain causal (the LM).
    window: ?c_int = null,

    fn load(t: *Tensors, a: std.mem.Allocator, s: S, comptime prefix: []const u8, n: usize, window: ?c_int) !QStack {
        const layers = try a.alloc(QLayer, n);
        for (layers, 0..) |*l, i| {
            l.* = .{
                .ln1 = try t.get(prefix ++ ".layers.{d}.input_layernorm.weight", .{i}),
                .ln2 = try t.get(prefix ++ ".layers.{d}.post_attention_layernorm.weight", .{i}),
                .q = try t.lin(s, prefix ++ ".layers.{d}.self_attn.q_proj", .{i}),
                .k = try t.lin(s, prefix ++ ".layers.{d}.self_attn.k_proj", .{i}),
                .v = try t.lin(s, prefix ++ ".layers.{d}.self_attn.v_proj", .{i}),
                .o = try t.lin(s, prefix ++ ".layers.{d}.self_attn.o_proj", .{i}),
                .gate = try t.lin(s, prefix ++ ".layers.{d}.mlp.gate_proj", .{i}),
                .up = try t.lin(s, prefix ++ ".layers.{d}.mlp.up_proj", .{i}),
                .down = try t.lin(s, prefix ++ ".layers.{d}.mlp.down_proj", .{i}),
            };
        }
        return .{ .layers = layers, .norm = try t.get(prefix ++ ".norm.weight", .{}), .window = window };
    }

    /// `x` [B, L, 896] → normed hidden [B, L, 896]. With `cache`, keys/values
    /// are appended (the LM decode path).
    fn forward(st: *const QStack, g: *G, x_in: Arr, cache: ?*KvCache) !Arr {
        const sh = mlx.getShape(x_in);
        const B = sh[0];
        const L = sh[1];
        const offset: c_int = if (cache) |c| c.len else 0;
        // Sliding mask (min(causal, window) additive, in the activation dtype),
        // or SDPA's causal mode for a multi-token LM prefill.
        var mask: Arr = .{ .ctx = null };
        var mode: [*:0]const u8 = "";
        if (st.window) |w| {
            mask = try slidingMask(g, L, offset + L, offset, w, mlx.mlx_array_dtype(x_in));
            mode = "array";
        } else if (L > 1) {
            mode = "causal";
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(st.head_dim)));
        var x = x_in;
        for (st.layers, 0..) |*l, li| {
            const h = try g.rms(x, l.ln1, st.eps);
            var q = try g.transpose(try g.reshape(try l.q.f(g, h), &.{ B, L, st.heads, st.head_dim }), &.{ 0, 2, 1, 3 });
            var k = try g.transpose(try g.reshape(try l.k.f(g, h), &.{ B, L, st.kv_heads, st.head_dim }), &.{ 0, 2, 1, 3 });
            var v = try g.transpose(try g.reshape(try l.v.f(g, h), &.{ B, L, st.kv_heads, st.head_dim }), &.{ 0, 2, 1, 3 });
            q = try g.rope(q, st.head_dim, false, st.theta, offset);
            k = try g.rope(k, st.head_dim, false, st.theta, offset);
            if (cache) |c| {
                if (c.k[li].ctx != null) {
                    k = try g.concat(&.{ c.k[li], k }, 2);
                    v = try g.concat(&.{ c.v[li], v }, 2);
                    _ = mlx.mlx_array_free(c.k[li]);
                    _ = mlx.mlx_array_free(c.v[li]);
                }
                c.k[li] = try own(k);
                c.v[li] = try own(v);
            }
            var o = try g.sdpa(q, k, v, scale, mode, mask);
            o = try g.reshape(try g.transpose(o, &.{ 0, 2, 1, 3 }), &.{ B, L, st.heads * st.head_dim });
            const hres = try g.add(x, try l.o.f(g, o));
            const h2 = try g.rms(hres, l.ln2, st.eps);
            const m = try l.down.f(g, try compiledAct(g, .swiglu, &.{ try l.gate.f(g, h2), try l.up.f(g, h2) }));
            x = try g.add(hres, m);
        }
        if (cache) |c| c.len += L;
        return g.rms(x, st.norm, st.eps);
    }
};

/// mlx-audio `MingQwen2Attention._build_sliding_additive_mask` for mask="causal".
fn slidingMask(g: *G, q_len: c_int, kv_len: c_int, offset: c_int, window: c_int, dt: Dtype) !Arr {
    const n: usize = @intCast(q_len * kv_len);
    const buf = try g.a.alloc(f32, n);
    defer g.a.free(buf);
    const neg: f32 = -3.4028235e38;
    for (0..@intCast(q_len)) |i| {
        const qp: c_int = offset + @as(c_int, @intCast(i));
        for (0..@intCast(kv_len)) |j| {
            const kp: c_int = @intCast(j);
            const ok = kp > qp - window and qp >= kp;
            buf[i * @as(usize, @intCast(kv_len)) + j] = if (ok) 0 else neg;
        }
    }
    const m = try g.fromF32(buf, &.{ 1, 1, q_len, kv_len });
    // fromF32 does not copy until eval; materialize before `buf` is freed.
    const c = try g.astype(m, dt);
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

pub const KvCache = struct {
    k: []Arr,
    v: []Arr,
    len: c_int = 0,
    a: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, n: usize) !KvCache {
        const k = try a.alloc(Arr, n);
        const v = try a.alloc(Arr, n);
        for (k, v) |*x, *y| {
            x.* = .{ .ctx = null };
            y.* = .{ .ctx = null };
        }
        return .{ .k = k, .v = v, .a = a };
    }

    pub fn deinit(c: *KvCache) void {
        for (c.k, c.v) |x, y| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
            if (y.ctx != null) _ = mlx.mlx_array_free(y);
        }
        c.a.free(c.k);
        c.a.free(c.v);
    }
};

// ─────────────────────────────────────────────────────────────────────────
// DiT block stack (flow head + aggregator)
// ─────────────────────────────────────────────────────────────────────────

const DitBlock = struct {
    norm1: Arr,
    norm2: Arr,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    ff1: Lin,
    ff2: Lin,
};

/// x_transformers-style interleaved-pair rotary, tabulated per sequence
/// length from the checkpoint's `inv_freq` exactly as `RotaryEmbedding` does.
const Rotary = struct {
    inv_freq: Arr,

    fn apply(r: Rotary, g: *G, x: Arr) !Arr {
        // x: [B, H, L, D]
        const sh = mlx.getShape(x);
        const L = sh[2];
        var pa = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_arange(&pa, 0, @floatFromInt(L), 1, mlx.mlx_array_dtype(r.inv_freq), g.s));
        const pos = try g.keep(pa);
        // outer(pos, inv) = pos[:, None] * inv[None, :]
        const freqs = try g.mul(try g.reshape(pos, &.{ L, 1 }), try g.reshape(r.inv_freq, &.{ 1, -1 }));
        const c = try g.reshape(try g.cos(freqs), &.{ 1, 1, L, -1 });
        const s = try g.reshape(try g.sin(freqs), &.{ 1, 1, L, -1 });
        const D: c_int = sh[3];
        const xe = try g.sliceAxis(x, 3, 0, D, 2);
        const xo = try g.sliceAxis(x, 3, 1, D, 2);
        const oe = try g.sub(try g.mul(xe, c), try g.mul(xo, s));
        const oo = try g.add(try g.mul(xe, s), try g.mul(xo, c));
        const st = try g.stack(&.{ oe, oo }, -1);
        return g.astype(try g.reshape(st, sh), mlx.mlx_array_dtype(x));
    }
};

const DitStack = struct {
    blocks: []DitBlock,
    rot: Rotary,

    fn load(t: *Tensors, a: std.mem.Allocator, s: S, comptime prefix: []const u8, n: usize) !DitStack {
        const blocks = try a.alloc(DitBlock, n);
        for (blocks, 0..) |*b, i| {
            b.* = .{
                .norm1 = try t.get(prefix ++ ".blocks.{d}.norm1.weight", .{i}),
                .norm2 = try t.get(prefix ++ ".blocks.{d}.norm2.weight", .{i}),
                .q = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_q", .{i}),
                .k = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_k", .{i}),
                .v = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_v", .{i}),
                .out = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_out.0", .{i}),
                .ff1 = try t.lin(s, prefix ++ ".blocks.{d}.mlp.ff.0.0", .{i}),
                .ff2 = try t.lin(s, prefix ++ ".blocks.{d}.mlp.ff.2", .{i}),
            };
        }
        return .{ .blocks = blocks, .rot = .{ .inv_freq = try t.get(prefix ++ ".rotary_embed.inv_freq", .{}) } };
    }

    fn forward(st: *const DitStack, g: *G, x_in: Arr) !Arr {
        const sh = mlx.getShape(x_in);
        const B = sh[0];
        const L = sh[1];
        var x = x_in;
        for (st.blocks) |*b| {
            const h = try g.rms(x, b.norm1, 1e-6);
            var q = try g.transpose(try g.reshape(try b.q.f(g, h), &.{ B, L, DIT_HEADS, DIT_HEAD_DIM }), &.{ 0, 2, 1, 3 });
            var k = try g.transpose(try g.reshape(try b.k.f(g, h), &.{ B, L, DIT_HEADS, DIT_HEAD_DIM }), &.{ 0, 2, 1, 3 });
            const v = try g.transpose(try g.reshape(try b.v.f(g, h), &.{ B, L, DIT_HEADS, DIT_HEAD_DIM }), &.{ 0, 2, 1, 3 });
            q = try st.rot.apply(g, q);
            k = try st.rot.apply(g, k);
            var o = try g.sdpa(q, k, v, 1.0 / 8.0, "", .{ .ctx = null });
            o = try g.reshape(try g.transpose(o, &.{ 0, 2, 1, 3 }), &.{ B, L, DIT_HIDDEN });
            x = try g.add(x, try b.out.f(g, o));
            const h2 = try g.rms(x, b.norm2, 1e-6);
            x = try g.add(x, try b.ff2.f(g, try compiledAct(g, .gelu, &.{try b.ff1.f(g, h2)})));
        }
        return x;
    }
};

// ── mx.compile'd activations ──
// mlx.nn.gelu_approx / silu and mlx-audio's swiglu are `mx.compile(shapeless=True)`
// in Python: one fused kernel each. Same here, built once per process.

const Act = enum { gelu, silu, swiglu };
var act_cls: [3]?mlx.mlx_closure = .{ null, null, null };
/// Stream the traces run on (the model's; one model stream per process).
var act_stream: S = .{ .ctx = null };

fn actTrace(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const which: Act = @fromBackingInt(@intCast(@intFromPtr(payload) - 1));
    var g = G.init(std.heap.c_allocator, act_stream);
    defer g.deinit();
    var x = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
    _ = g.keep(x) catch return -1;
    const out = switch (which) {
        .gelu => geluApprox(&g, x),
        .silu => g.silu(x),
        .swiglu => blk: {
            var up = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&up, input, 1) != 0) return -1;
            _ = g.keep(up) catch return -1;
            break :blk g.mul(g.silu(x) catch return -1, up);
        },
    } catch return -1;
    res.* = mlx.mlx_vector_array_new_value(out);
    return 0;
}

fn compiledAct(g: *G, which: Act, inputs: []const Arr) !Arr {
    if (g.in_trace) return switch (which) {
        .gelu => geluApprox(g, inputs[0]),
        .silu => g.silu(inputs[0]),
        .swiglu => g.mul(try g.silu(inputs[0]), inputs[1]),
    };
    const i = @backingInt(which);
    if (act_cls[i] == null) {
        const raw = mlx.mlx_closure_new_func_payload(&actTrace, @ptrFromInt(i + 1), null);
        defer _ = mlx.mlx_closure_free(raw);
        var c = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&c, raw, true));
        act_cls[i] = c;
    }
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (inputs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
    var r = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(r);
    try mlx.check(mlx.mlx_closure_apply(&r, act_cls[i].?, v));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&o, r, 0));
    return g.keep(o);
}

/// mlx.nn.gelu_approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3))).
fn geluApprox(g: *G, x: Arr) !Arr {
    const dt = mlx.mlx_array_dtype(x);
    const x3 = try g.power(x, try g.scalar(3, dt));
    const inner = try g.add(x, try g.mul(try g.scalar(0.044715, dt), x3));
    const t = try g.tanh(try g.mul(try g.scalar(@sqrt(2.0 / std.math.pi), dt), inner));
    return g.mul(try g.mul(try g.scalar(0.5, dt), x), try g.add(try g.scalar(1, dt), t));
}

// ─────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────

pub const Options = struct {
    /// Cast every weight (fp32 parity mode); null keeps stored dtypes (bf16 +
    /// two fp32 inv_freq buffers), which is how the Python engine ships.
    cast_to: ?Dtype = null,
};

pub const Model = struct {
    a: std.mem.Allocator,
    s: S,
    t: Tensors,
    /// Activation dtype for prompt rounding and the Python `self.dtype`.
    dtype: Dtype,

    embed: Arr,
    lm: QStack,
    stop_head: Lin,
    spk_head: Lin,
    // flow head
    dit: DitStack,
    t_mlp0: Lin,
    t_mlp2: Lin,
    x_emb: Lin,
    c_emb: Lin,
    dit_final_norm: Arr,
    dit_final: Lin,
    // aggregator
    agg: DitStack,
    agg_x_emb: Lin,
    agg_cls: Arr,
    agg_final_norm: Arr,
    agg_final: Lin,
    // VAE
    enc: QStack,
    enc_agg: QStack,
    enc_fc1: Lin,
    enc_fc2: Lin,
    enc_fc3: Lin,
    enc_cls: Arr,
    dec: QStack,
    dec_fc1: Lin,
    dec_head: Lin,
    window: Arr, // ISTFT window (checkpoint buffer), f32
    cfm_compiled: ?mlx.mlx_closure = null,
    cfm_params: [3]f32 = .{ 0, 0, 0 },
    agg_compiled: ?mlx.mlx_closure = null,
    /// SinusPositionEmbedding frequencies: exp(arange(128, f32) * -log(1e4)/127).
    t_freq: Arr,

    pub fn load(a: std.mem.Allocator, s: S, model_dir: []const u8, opts: Options) !*Model {
        const path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{model_dir});
        defer a.free(path);
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.* = undefined;
        m.cfm_compiled = null;
        m.agg_compiled = null;
        m.a = a;
        m.s = s;
        act_stream = s;
        m.dtype = opts.cast_to orelse .bfloat16;
        m.t = try Tensors.load(a, path, opts.cast_to, s);
        errdefer m.t.deinit();
        const t = &m.t;
        m.embed = try t.get("model.model.embed_tokens.weight", .{});
        m.lm = try QStack.load(t, a, s, "model.model", 24, null);
        m.stop_head = try t.lin(s, "stop_head", .{});
        m.spk_head = try t.lin(s, "spk_head", .{});
        const F = "flowloss.cfm.model";
        m.dit = try DitStack.load(t, a, s, F, 8);
        m.t_mlp0 = try t.lin(s, F ++ ".t_embedder.time_mlp.0", .{});
        m.t_mlp2 = try t.lin(s, F ++ ".t_embedder.time_mlp.2", .{});
        m.x_emb = try t.lin(s, F ++ ".x_embedder", .{});
        m.c_emb = try t.lin(s, F ++ ".c_embedder.cond_embedder", .{});
        m.dit_final_norm = try t.get(F ++ ".final_layer.norm_final.weight", .{});
        m.dit_final = try t.lin(s, F ++ ".final_layer.linear", .{});
        const A = "linear_proj_audio";
        m.agg = try DitStack.load(t, a, s, A, 8);
        m.agg_x_emb = try t.lin(s, A ++ ".x_embedder", .{});
        m.agg_cls = try t.get(A ++ ".word_embedder.weight", .{});
        m.agg_final_norm = try t.get(A ++ ".final_layer.norm_final.weight", .{});
        m.agg_final = try t.lin(s, A ++ ".final_layer.linear", .{});
        m.enc = try QStack.load(t, a, s, "audio.encoder.encoder", 24, 64);
        m.enc_agg = try QStack.load(t, a, s, "audio.encoder.aggregator", 4, 64);
        m.enc_fc1 = try t.lin(s, "audio.encoder.fc1", .{});
        m.enc_fc2 = try t.lin(s, "audio.encoder.fc2", .{});
        m.enc_fc3 = try t.lin(s, "audio.encoder.fc3", .{});
        m.enc_cls = try t.get("audio.encoder.cls_embed", .{});
        m.dec = try QStack.load(t, a, s, "audio.decoder.decoder", 24, 64);
        m.dec_fc1 = try t.lin(s, "audio.decoder.fc1", .{});
        m.dec_head = try t.lin(s, "audio.decoder.head.out", .{});
        {
            var g = G.init(a, s);
            defer g.deinit();
            var ar = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_arange(&ar, 0, 128, 1, .float32, s));
            const e: f32 = @floatCast(@log(@as(f64, 10000.0)) / 127.0);
            const f = try g.exp(try g.mul(try g.keep(ar), try g.scalar(-e, .float32)));
            m.t_freq = try own(f);
            try t.extra.append(a, m.t_freq);
        }
        var w32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&w32, try t.get("audio.decoder.head.istft.window", .{}), .float32, s));
        try t.extra.append(a, w32);
        m.window = w32;
        // Materialize everything now so the first request does not pay for it.
        var all: std.ArrayList(Arr) = .empty;
        defer all.deinit(a);
        var it = t.map.valueIterator();
        while (it.next()) |v| try all.append(a, v.*);
        try all.appendSlice(a, t.extra.items);
        try evalAll(all.items);
        return m;
    }

    pub fn deinit(m: *Model) void {
        const a = m.a;
        if (m.cfm_compiled) |c| _ = mlx.mlx_closure_free(c);
        if (m.agg_compiled) |c| _ = mlx.mlx_closure_free(c);
        for ([_][]QLayer{ m.lm.layers, m.enc.layers, m.enc_agg.layers, m.dec.layers }) |ls| a.free(ls);
        a.free(m.dit.blocks);
        a.free(m.agg.blocks);
        m.t.deinit();
        a.destroy(m);
    }

    // ── LM ──

    pub fn embedIds(m: *Model, g: *G, ids: []const i32) !Arr {
        const idx = try g.fromI32(ids, &.{@intCast(ids.len)});
        const e = try g.take(m.embed, idx, 0);
        return g.reshape(e, &.{ 1, @intCast(ids.len), -1 });
    }

    pub fn lmForward(m: *Model, g: *G, embeds: Arr, cache: *KvCache) !Arr {
        return m.lm.forward(g, embeds, cache);
    }

    /// softmax(stop_head(z).astype(f32))[0, 0, 1]
    pub fn stopProb(m: *Model, g: *G, z: Arr) !Arr {
        const logits = try g.astype(try m.stop_head.f(g, z), .float32);
        const p = try g.softmax(logits, -1);
        return g.reshape(try g.sliceAxis(p, 2, 1, 2, 1), &.{});
    }

    // ── Aggregator: latent [B, 1, 64] → LM input rows [B, 1, 896] ──

    pub fn aggregateRaw(m: *Model, g: *G, latent: Arr) !Arr {
        const x = try m.agg_x_emb.f(g, latent);
        const B = mlx.getShape(x)[0];
        var cb = mlx.mlx_array_new();
        const csh = [_]c_int{ B, 1, DIT_HIDDEN };
        try mlx.check(mlx.mlx_broadcast_to(&cb, try g.reshape(m.agg_cls, &.{ 1, 1, DIT_HIDDEN }), &csh, 3, g.s));
        const cls_b = try g.keep(cb);
        var h = try g.concat(&.{ cls_b, x }, 1);
        h = try m.agg.forward(g, h);
        h = try m.agg_final.f(g, try g.rms(h, m.agg_final_norm, 1e-6));
        return g.sliceAxis(h, 1, 0, 1, 1);
    }

    // ── Flow head ──

    /// DiT.forward_with_cfg: x [1,1,64], t scalar (solver dtype), c [1,1,896],
    /// hist [1,1,64] → [2, 1, 64] (cond, uncond).
    pub fn ditCfg(m: *Model, g: *G, x1: Arr, t: Arr, c1: Arr, hist1: Arr) !Arr {
        const x = try g.concat(&.{ x1, x1 }, 0);
        const hist = try g.concat(&.{ hist1, hist1 }, 0);
        const c = try g.concat(&.{ c1, try g.zeros(mlx.getShape(c1), mlx.mlx_array_dtype(c1)) }, 0);
        const xdt = mlx.mlx_array_dtype(x);
        var tt = mlx.mlx_array_new();
        const two = [_]c_int{2};
        try mlx.check(mlx.mlx_full(&tt, &two, 1, t, xdt, g.s));
        const tb = try g.keep(tt);
        // TimestepEmbedder: sinusoid in f32 (x * 1000 rounds in t's dtype first).
        const ts = try g.mul(try g.scalar(1000.0, xdt), tb);
        const emb = try g.mul(try g.reshape(ts, &.{ 2, 1 }), try g.reshape(m.t_freq, &.{ 1, 128 }));
        const sincos = try g.astype(try g.concat(&.{ try g.sin(emb), try g.cos(emb) }, -1), xdt);
        var te = try m.t_mlp0.f(g, sincos);
        te = try m.t_mlp2.f(g, try compiledAct(g, .silu, &.{te}));
        const t_embed = try g.reshape(te, &.{ 2, 1, DIT_HIDDEN });
        const x_now = try m.x_emb.f(g, x);
        const x_hist = try m.x_emb.f(g, hist);
        const c_embed = try m.c_emb.f(g, c);
        const y = try g.add(t_embed, c_embed);
        var h = try g.concat(&.{ y, x_hist, x_now }, 1);
        h = try m.dit.forward(g, h);
        h = try m.dit_final.f(g, try g.rms(h, m.dit_final_norm, 1e-6));
        const L = mlx.getShape(h)[1];
        return g.sliceAxis(h, 1, L - 1, L, 1);
    }

    /// Run `body(m, g, inputs)` through an mlx_compile'd closure cached in
    /// `slot` (shapes are fixed per call site, so each traces once).
    /// MING_COMPILE=0 runs the op-by-op graph instead.
    fn compiled(m: *Model, g: *G, slot: *?mlx.mlx_closure, comptime body: fn (*Model, *G, []const Arr) anyerror!Arr, inputs: []const Arr) !Arr {
        if (g.in_trace or !compileEnabled()) return body(m, g, inputs);
        if (slot.* == null) {
            const Tr = struct {
                fn trace(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                    const mm: *Model = @ptrCast(@alignCast(payload.?));
                    var tg = G.init(std.heap.c_allocator, mm.s);
                    tg.in_trace = true;
                    defer tg.deinit();
                    var xs: [8]Arr = undefined;
                    const n = mlx.mlx_vector_array_size(input);
                    for (0..n) |i| {
                        xs[i] = mlx.mlx_array_new();
                        if (mlx.mlx_vector_array_get(&xs[i], input, i) != 0) return -1;
                        _ = tg.keep(xs[i]) catch return -1;
                    }
                    const out = body(mm, &tg, xs[0..n]) catch return -1;
                    res.* = mlx.mlx_vector_array_new_value(out);
                    return 0;
                }
            };
            const raw = mlx.mlx_closure_new_func_payload(&Tr.trace, m, null);
            defer _ = mlx.mlx_closure_free(raw);
            var c = mlx.mlx_closure{ .ctx = null };
            try mlx.check(mlx.mlx_compile(&c, raw, false));
            slot.* = c;
        }
        const v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        for (inputs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
        var r = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(r);
        try mlx.check(mlx.mlx_closure_apply(&r, slot.*.?, v));
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, r, 0));
        return g.keep(o);
    }

    fn cfmBody(m: *Model, g: *G, in: []const Arr) !Arr {
        return m.cfmSampleRaw(g, in[0], in[1], in[2], in[3], m.cfm_params[0], m.cfm_params[1], m.cfm_params[2]);
    }

    fn aggBody(m: *Model, g: *G, in: []const Arr) !Arr {
        return m.aggregateRaw(g, in[0]);
    }

    /// CFM.sample for one LM step, compiled whole (10 Euler steps × CFG DiT).
    pub fn cfmSample(m: *Model, g: *G, z: Arr, hist: Arr, noise: Arr, sde: Arr, cfg: f32, sigma: f32, temperature: f32) !Arr {
        const p = [3]f32{ cfg, sigma, temperature };
        if (m.cfm_compiled != null and !std.mem.eql(f32, &p, &m.cfm_params)) {
            _ = mlx.mlx_closure_free(m.cfm_compiled.?);
            m.cfm_compiled = null;
        }
        m.cfm_params = p;
        return m.compiled(g, &m.cfm_compiled, cfmBody, &.{ z, hist, noise, sde });
    }

    /// Aggregator for one latent [1, 1, 64] (compiled); batched prompt rows go uncompiled.
    pub fn aggregate(m: *Model, g: *G, latent: Arr) !Arr {
        if (mlx.getShape(latent)[0] != 1) return m.aggregateRaw(g, latent);
        return m.compiled(g, &m.agg_compiled, aggBody, &.{latent});
    }

    /// CFM.sample (ming3 sampling): Euler over the EPSS + sway schedule in the
    /// noise dtype. `noise` [1,1,64], `sde` [10,1,1,64].
    pub fn cfmSampleRaw(m: *Model, g: *G, z: Arr, hist: Arr, noise: Arr, sde: Arr, cfg: f32, sigma: f32, temperature: f32) !Arr {
        const dt = mlx.mlx_array_dtype(noise);
        const sched = [_]f32{ 0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32 };
        // t = (1/32) * array(sched, dt); t = t + (-1) * (cos(pi/2 * t) - 1 + t)
        var t = try g.astype(try g.fromF32(&sched, &.{11}), dt);
        t = try g.mul(try g.scalar(1.0 / 32.0, dt), t);
        const ct = try g.cos(try g.mul(try g.scalar(std.math.pi / 2.0, dt), t));
        const sw = try g.add(try g.sub(ct, try g.scalar(1, dt)), t);
        t = try g.add(t, try g.mul(try g.scalar(-1, dt), sw));
        var y = noise;
        const shift_k = sigma * std.math.sqrt(temperature);
        for (1..sched.len) |i| {
            const t0 = try g.reshape(try g.sliceAxis(t, 0, @intCast(i - 1), @intCast(i), 1), &.{});
            const t1 = try g.reshape(try g.sliceAxis(t, 0, @intCast(i), @intCast(i + 1), 1), &.{});
            const dtt = try g.sub(t1, t0);
            const out = try m.ditCfg(g, y, t0, z, hist);
            const pn = try g.split2(out, 0);
            const cfg_a = try g.scalar(cfg, mlx.mlx_array_dtype(out));
            var f0 = try g.add(pn[0], try g.mul(try g.sub(pn[0], pn[1]), cfg_a));
            f0 = try g.astype(f0, dt);
            y = try g.add(y, try g.mul(try g.astype(dtt, dt), f0));
            if (shift_k != 0) {
                const nz = try g.reshape(try g.sliceAxis(sde, 0, @intCast(i - 1), @intCast(i), 1), mlx.getShape(y));
                const sq = try g.power(try g.abs(dtt), try g.scalar(0.5, dt));
                y = try g.add(y, try g.mul(try g.mul(try g.scalar(shift_k, dt), sq), nz));
            }
        }
        return y;
    }

    // ── VAE ──

    /// Encoder: waveform [1, N] f32 (N a multiple of 882) → h [1, T, 128].
    pub fn vaeEncodeH(m: *Model, g: *G, wav: Arr) !Arr {
        const N = mlx.getShape(wav)[1];
        const nf = @divFloor(N + @as(c_int, HOP) - 1, @as(c_int, HOP));
        var x = wav;
        const need = nf * @as(c_int, HOP);
        if (need > N) x = try g.concat(&.{ x, try g.zeros(&.{ 1, need - N }, .float32) }, 1);
        x = try g.reshape(x, &.{ 1, nf, @as(c_int, HOP) });
        x = try g.astype(x, mlx.mlx_array_dtype(m.enc_fc1.wt));
        x = try m.enc_fc2.f(g, try m.enc_fc1.f(g, x));
        x = try m.enc.forward(g, x, null);
        // pad_patch_insert_cls
        const T = mlx.getShape(x)[1];
        const r = @mod(T, @as(c_int, VAE_PATCH));
        if (r != 0) x = try g.concat(&.{ x, try g.zeros(&.{ 1, @as(c_int, VAE_PATCH) - r, HIDDEN }, mlx.mlx_array_dtype(x)) }, 1);
        x = try g.reshape(x, &.{ -1, @as(c_int, VAE_PATCH), HIDDEN });
        const np_ = mlx.getShape(x)[0];
        const cls = try g.astype(m.enc_cls, mlx.mlx_array_dtype(x));
        var cb = mlx.mlx_array_new();
        const csh = [_]c_int{ np_, 1, HIDDEN };
        try mlx.check(mlx.mlx_broadcast_to(&cb, cls, &csh, 3, g.s));
        x = try g.concat(&.{ x, try g.keep(cb) }, 1);
        x = try g.reshape(x, &.{ 1, -1, HIDDEN });
        x = try m.enc_agg.forward(g, x, null);
        x = try g.reshape(x, &.{ -1, @as(c_int, VAE_PATCH) + 1, HIDDEN });
        x = try g.reshape(try g.sliceAxis(x, 1, @as(c_int, VAE_PATCH), @as(c_int, VAE_PATCH) + 1, 1), &.{ 1, -1, HIDDEN });
        return m.enc_fc3.f(g, x);
    }

    /// Posterior sample: mean + (softplus(scale) + 1e-4) * noise.
    pub fn vaePosterior(g: *G, h: Arr, noise: Arr) !Arr {
        const ms = try g.split2(h, -1);
        const dt = mlx.mlx_array_dtype(ms[0]);
        const sp = try g.logaddexp(ms[1], try g.scalar(0, dt));
        const std_ = try g.add(sp, try g.scalar(1e-4, dt));
        return g.add(ms[0], try g.mul(std_, try g.astype(noise, dt)));
    }

    /// Decoder: latents [1, T, 64] → waveform samples (T * 4 * 882, f32, owned).
    pub fn vaeDecode(m: *Model, g: *G, latents: Arr) ![]f32 {
        var x = try m.dec_fc1.f(g, latents);
        x = try linearUpsample(g, x, VAE_PATCH);
        x = try m.dec.forward(g, x, null);
        // ISTFT head
        var p = try m.dec_head.f(g, x); // [1, F, 3530]
        p = try g.transpose(p, &.{ 0, 2, 1 });
        const mp = try g.split2(p, 1);
        var mag = try g.exp(mp[0]);
        const big = try g.scalar(1e2, mlx.mlx_array_dtype(mag));
        mag = try g.minimum(mag, big);
        const ph = mp[1];
        const i_unit = try g.keep(mlx.mlx_array_new_complex(0, 1));
        const spec = try g.mul(mag, try g.add(try g.cos(ph), try g.mul(i_unit, try g.sin(ph))));
        var fr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fft_irfft(&fr, spec, @intCast(N_FFT), 1, mlx.MLX_FFT_NORM_BACKWARD, g.s));
        var frames = try g.keep(fr); // [1, n_fft, F] f32
        frames = try g.mul(frames, try g.reshape(m.window, &.{ 1, @intCast(N_FFT), 1 }));
        const F: usize = @intCast(mlx.getShape(frames)[2]);
        // Overlap-add with n_fft = 4 * hop: block b sums frame (b - j) part j,
        // added in frame order like the Python loop.
        const fT = try g.reshape(try g.transpose(frames, &.{ 0, 2, 1 }), &.{ @intCast(F), 4, @intCast(HOP) });
        var acc: ?Arr = null;
        var j: usize = 4;
        while (j > 0) {
            j -= 1;
            const part = try g.reshape(try g.sliceAxis(fT, 1, @intCast(j), @intCast(j + 1), 1), &.{ @intCast(F), @intCast(HOP) });
            const lo: c_int = @intCast(j);
            const hi: c_int = @intCast(3 - j);
            var padded = part;
            if (lo > 0) padded = try g.concat(&.{ try g.zeros(&.{ lo, @intCast(HOP) }, .float32), padded }, 0);
            if (hi > 0) padded = try g.concat(&.{ padded, try g.zeros(&.{ hi, @intCast(HOP) }, .float32) }, 0);
            acc = if (acc) |s_| try g.add(s_, padded) else padded;
        }
        const total = (F + 3) * HOP;
        const out = try toF32(g.a, try g.reshape(acc.?, &.{-1}), g.s);
        defer g.a.free(out);
        // Window envelope (host, same accumulation order) and the "same" trim.
        const win = try toF32(g.a, m.window, g.s);
        defer g.a.free(win);
        const env = try g.a.alloc(f32, total);
        defer g.a.free(env);
        @memset(env, 0);
        for (0..F) |t_| {
            for (0..N_FFT) |k| env[t_ * HOP + k] += win[k] * win[k];
        }
        const pad = (N_FFT - HOP) / 2;
        const n_out = total - 2 * pad;
        const res = try g.a.alloc(f32, n_out);
        for (0..n_out) |i| res[i] = out[pad + i] / @max(env[pad + i], 1e-11);
        return res;
    }
};

fn compileEnabled() bool {
    const v = std.c.getenv("MING_COMPILE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Decoder._linear_upsample_1d (align_corners=False), in f32 like the Python.
fn linearUpsample(g: *G, x_btc: Arr, scale: usize) !Arr {
    const x = try g.astype(try g.transpose(x_btc, &.{ 0, 2, 1 }), .float32);
    const in_t: usize = @intCast(mlx.getShape(x)[2]);
    const out_t = in_t * scale;
    const idx = try g.a.alloc(f32, out_t);
    defer g.a.free(idx);
    for (0..out_t) |i| idx[i] = @floatFromInt(i);
    const ar = try g.fromF32(idx, &.{@intCast(out_t)});
    const sc = try g.scalar(@floatFromInt(scale), .float32);
    const pos = try g.sub(try g.div(try g.add(ar, try g.scalar(0.5, .float32)), sc), try g.scalar(0.5, .float32));
    try mlx.check(mlx.mlx_array_eval(pos));
    const left_f = try g.floor(pos);
    const left = try g.astype(left_f, .int32);
    const right = try g.add(left, try g.astype(try g.scalar(1, .float32), .int32));
    const w_right = try g.sub(pos, try g.astype(left, .float32));
    const lo = try g.astype(try g.scalar(0, .float32), .int32);
    const hi = try g.astype(try g.scalar(@floatFromInt(in_t - 1), .float32), .int32);
    const lv = try g.take(x, try g.clip(left, lo, hi), -1);
    const rv = try g.take(x, try g.clip(right, lo, hi), -1);
    const one = try g.scalar(1, .float32);
    const wl = try g.reshape(try g.sub(one, w_right), &.{ 1, 1, -1 });
    const wr = try g.reshape(w_right, &.{ 1, 1, -1 });
    const y = try g.add(try g.mul(lv, wl), try g.mul(rv, wr));
    return g.transpose(y, &.{ 0, 2, 1 });
}

// ─────────────────────────────────────────────────────────────────────────
// AR + flow-matching loop (core.py `sample_latents`)
// ─────────────────────────────────────────────────────────────────────────

/// Where the flow head's Gaussian draws come from: MLX's global RNG (same
/// call order and shapes as the Python engine, so a shared seed reproduces
/// it) or a replay bank [steps, 11, 64] (row 0 start noise, rows 1..10 SDE).
pub const Noise = union(enum) {
    rng,
    replay: Arr,

    fn draw(n: Noise, g: *G, step: usize, dt: Dtype) ![2]Arr {
        switch (n) {
            .rng => return .{ try g.normal(&.{ 1, 1, LATENT }, dt), try g.normal(&.{ @intCast(FLOW_STEPS), 1, 1, LATENT }, dt) },
            .replay => |bank| {
                const row = try g.sliceAxis(bank, 0, @intCast(step), @intCast(step + 1), 1);
                const n0 = try g.reshape(try g.sliceAxis(row, 1, 0, 1, 1), &.{ 1, 1, LATENT });
                const sd = try g.reshape(try g.sliceAxis(row, 1, 1, @intCast(FLOW_STEPS + 1), 1), &.{ @intCast(FLOW_STEPS), 1, 1, LATENT });
                return .{ try g.astype(n0, dt), try g.astype(sd, dt) };
            },
        }
    }
};

pub const SampleParams = struct {
    max_steps: usize = 1000,
    cfg: f32 = 2.0,
    sigma: f32 = 0.25,
    temperature: f32 = 0.0,
};

/// Optional per-step trace for parity checks (all owned by the caller).
pub const Trace = struct {
    z: std.ArrayList(Arr) = .empty,
    stop: std.ArrayList(f32) = .empty,

    pub fn deinit(t: *Trace, a: std.mem.Allocator) void {
        for (t.z.items) |x| _ = mlx.mlx_array_free(x);
        t.z.deinit(a);
        t.stop.deinit(a);
    }
};

/// embeds [1, L, 896] → latents [1, steps, 64] (owned).
pub fn sampleLatents(m: *Model, embeds_in: Arr, prompt_latent: ?Arr, p: SampleParams, noise: Noise, trace: ?*Trace) !Arr {
    const a = m.a;
    const L: usize = @intCast(mlx.getShape(embeds_in)[1]);
    if (L >= MAX_CACHE_LEN) return error.PromptTooLong;
    const budget = MAX_CACHE_LEN - L; // patch_size 1
    const steps = @min(p.max_steps, budget);
    if (std.c.getenv("MING_PROFILE") != null) std.debug.print("[ming] prefix {d} positions, step budget {d} (cap {d})\n", .{ L, budget, p.max_steps });
    var cache = try KvCache.init(a, m.lm.layers.len);
    defer cache.deinit();
    var latents: std.ArrayList(Arr) = .empty;
    defer {
        for (latents.items) |x| _ = mlx.mlx_array_free(x);
        latents.deinit(a);
    }
    var embeds = try own(embeds_in);
    defer _ = mlx.mlx_array_free(embeds);
    var history: Arr = .{ .ctx = null };
    defer if (history.ctx != null) {
        _ = mlx.mlx_array_free(history);
    };
    const io_t = std.Io.Threaded.global_single_threaded.io();
    const prof = std.c.getenv("MING_PROFILE") != null;
    var build_ns: u64 = 0;
    var eval_ns: u64 = 0;
    defer if (prof) std.debug.print("[ming] {d} steps: graph build {d:.2} ms/step, eval {d:.2} ms/step\n", .{ latents.items.len, @as(f64, @floatFromInt(build_ns)) / 1e6 / @as(f64, @floatFromInt(@max(latents.items.len, 1))), @as(f64, @floatFromInt(eval_ns)) / 1e6 / @as(f64, @floatFromInt(@max(latents.items.len, 1))) });
    for (0..steps) |step| {
        var g = G.init(a, m.s);
        defer g.deinit();
        const t_build = std.Io.Timestamp.now(io_t, .awake);
        const hidden = try m.lmForward(&g, embeds, &cache);
        const Lh = mlx.getShape(hidden)[1];
        const z = try g.sliceAxis(hidden, 1, Lh - 1, Lh, 1);
        const zdt = mlx.mlx_array_dtype(z);
        if (step == 0) {
            var h = try g.zeros(&.{ 1, 1, LATENT }, zdt);
            if (prompt_latent) |pl_in| {
                // history_patch_size 1: the prompt's last latent seeds the history.
                const pl = try g.astype(pl_in, zdt);
                const T = mlx.getShape(pl)[1];
                h = try g.sliceAxis(pl, 1, T - 1, T, 1);
            }
            history = try own(h);
        }
        const nz = try noise.draw(&g, step, zdt);
        const latent = try m.cfmSample(&g, z, history, nz[0], nz[1], p.cfg, p.sigma, p.temperature);
        const stop = try m.stopProb(&g, z);
        const t_eval = std.Io.Timestamp.now(io_t, .awake);
        build_ns += @intCast(t_build.durationTo(t_eval).nanoseconds);
        try evalAll(&.{ latent, stop });
        eval_ns += @intCast(t_eval.untilNow(io_t, .awake).nanoseconds);
        const sp = try itemF32(stop, m.s);
        if (trace) |t| {
            try t.z.append(a, try own(z));
            try t.stop.append(a, sp);
        }
        try latents.append(a, try own(latent));
        if (sp > 0.5 and step > 3) break;
        // Left lazy: they evaluate with the next step's graph, as in Python.
        const next = try m.aggregate(&g, latent);
        const nh = try g.concat(&.{ try g.sliceAxis(history, 1, 1, 1, 1), latent }, 1);
        _ = mlx.mlx_array_free(embeds);
        embeds = try own(next);
        _ = mlx.mlx_array_free(history);
        history = try own(nh);
    }
    var g = G.init(a, m.s);
    defer g.deinit();
    return own(try g.concat(latents.items, 1));
}
