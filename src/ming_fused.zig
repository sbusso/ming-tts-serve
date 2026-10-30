//! Fused Metal kernels for Ming's DiT stacks (CFM flow head + aggregator).
//!
//! The DiT runs on 6 rows (CFG batch 2 × seq 3) or 2 rows (aggregator), so
//! every linear is a bandwidth-bound GEMV. One DiT block becomes 4 dispatches
//! (MLX: ~15), each reading its weights once:
//!
//!   A  rmsnorm1 + q/k/v linear + bias + rotary      → QKV [3, R, 1024]
//!   B  attention over L positions + out linear + residual
//!   C  rmsnorm2 + ff1 linear + bias + gelu_approx    → [R, 4096]
//!   D  ff2 linear + bias + residual
//!
//! plus, per Euler step of the CFM, E (x embedder + input row assembly) and
//! F (final norm + linear on the last row + CFG combine + Euler update).
//!
//! Storage is bf16, accumulation f32, and every value MLX rounds to bf16 is
//! rounded here at the same point (linear outputs, rotary, attention output,
//! each op of the gelu chain, each op of the CFG/Euler update). What differs
//! from MLX: f32 summation order inside the dot products, the rmsnorm sum of
//! squares, and the softmax (two-pass instead of MLX's online one), and
//! gelu's x**3 is x*x*x instead of pow(x, 3). All are f32-level; outputs move
//! by at most bf16 rounding flips. MING_FUSED=0 restores the MLX graph.
//!
//! GEMV layout (A–D): a threadgroup of 4 simdgroups owns 8 output columns for
//! all R rows; the 4 simdgroups split K, lanes read 4 consecutive bf16 per
//! 128-wide block (one 8-byte load per weight row), and partial sums meet in
//! threadgroup memory. The rows a kernel multiplies (normed x, attention
//! output) sit in threadgroup memory as f32, so R is capped at 6 (24 KB).

const std = @import("std");
const mlx = @import("mlx.zig");

const Arr = mlx.mlx_array;
const S = mlx.mlx_stream;

pub const D = 1024;
pub const FF = 4096;
pub const MAX_ROWS = 6;
/// GEMV tile: output columns per threadgroup, simdgroups per threadgroup
/// (they split K), elements per lane per block (multiple of 4).
const Tile = struct { no: c_int, sg: c_int, vw: c_int };

pub fn enabled() bool {
    const v = std.c.getenv("MING_FUSED") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

const HEADER =
    \\inline float4 bf4(uint2 v) {
    \\    return float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xffff0000u),
    \\                  as_type<float>(v.y << 16), as_type<float>(v.y & 0xffff0000u));
    \\}
    \\inline float rb(float x) { return float(bfloat16_t(x)); }
    \\#define LDW(p, i) bf4(*(const device uint2*)((p) + (i)))
    \\inline float rsum_(threadgroup const float* red, int sgn, int no, int rows, int n, int r) {
    \\    float a = 0.0f;
    \\    for (int g = 0; g < sgn; g++) a += red[(g * no + n) * rows + r];
    \\    return a;
    \\}
    \\inline float rinv_(threadgroup const float* ssr, int sgn, int rows, int r) {
    \\    float a = 0.0f;
    \\    for (int g = 0; g < sgn; g++) a += ssr[g * rows + r];
    \\    return metal::precise::rsqrt(a / 1024.0f + 1e-6f);
    \\}
    \\
;

const PRELUDE =
    \\const uint t = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int ob = int(threadgroup_position_in_grid.x) * NO;
    \\threadgroup float red[SG * NO * R];
    \\threadgroup float ssr[SG * R];
    \\
;

/// Partial dot products of the threadgroup's NO output columns over all R
/// rows. SG simdgroups split K; a lane takes VW consecutive elements per
/// 32*VW-wide block. `wp` is column 0's weight row; `xv` loads float4 of row
/// r at i. With `norm`, the rows are rms-normed on the fly: the sum of squares
/// rides along in `ss`, the norm weight NW scales x here, and 1/rms is applied
/// in the epilogue (MLX instead rounds w * T(x * inv) to bf16 per element; the
/// difference is below the bf16 rounding of the output).
fn gemv(comptime k: usize, comptime wp: []const u8, comptime xv: []const u8, comptime norm: bool) []const u8 {
    return std.fmt.comptimePrint(
        \\{{
        \\const device bfloat16_t* wp = {1s};
        \\float acc[NO][R];
        \\float ss[R];
        \\for (int r = 0; r < R; r++) {{ ss[r] = 0.0f; for (int n = 0; n < NO; n++) acc[n][r] = 0.0f; }}
        \\for (int b = int(sg) * ({0d} / SG); b < (int(sg) + 1) * ({0d} / SG); b += 32 * VW)
        \\for (int u = 0; u < VW; u += 4) {{
        \\    const int i = b + int(lane) * VW + u;
        \\    float4 xv[R];
        \\    for (int r = 0; r < R; r++) xv[r] = {2s};
        \\    {3s}
        \\    for (int n = 0; n < NO; n++) {{
        \\        const float4 w = LDW(wp + n * {0d}, i);
        \\        for (int r = 0; r < R; r++) acc[n][r] += dot(w, xv[r]);
        \\    }}
        \\}}
        \\for (int r = 0; r < R; r++) {{
        \\    for (int n = 0; n < NO; n++) {{
        \\        const float v = simd_sum(acc[n][r]);
        \\        if (lane == 0) red[(int(sg) * NO + n) * R + r] = v;
        \\    }}
        \\    {4s}
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\}}
        \\
    , .{
        k,                                                                                                                          wp,                                                                                                xv,
        if (norm) "const float4 nw = LDW(NW, i); for (int r = 0; r < R; r++) { ss[r] += dot(xv[r], xv[r]); xv[r] *= nw; }" else "", if (norm) "{ const float v = simd_sum(ss[r]); if (lane == 0) ssr[int(sg) * R + r] = v; }" else "",
    });
}

/// Epilogue helpers: column n's dot for row r, and 1/rms of row r.
const SUMS =
    \\#define rsum(n, r) rsum_(red, SG, NO, R, n, r)
    \\#define rinv(r) rinv_(ssr, SG, R, r)
    \\
;

// A: rmsnorm1 + qkv + rotary. Grid: 3072 / NO threadgroups.
const SRC_QKV = PRELUDE ++
    \\const int which = ob / 1024, col0 = ob % 1024;
    \\const device bfloat16_t* W = which == 0 ? WQ : (which == 1 ? WK : WV);
    \\const device bfloat16_t* Bv = which == 0 ? BQ : (which == 1 ? BK : BV);
    \\
++ gemv(1024, "W + col0 * 1024", "LDW(X, r * 1024 + i)", true) ++ SUMS ++
    \\for (int ei = int(t); ei < NO / 2 * R; ei += 32 * SG) {
    \\    const int n = (ei / R) * 2, r = ei % R, col = col0 + n;
    \\    const float inv = rinv(r);
    \\    float e = rb(rsum(n, r) * inv + float(Bv[col]));
    \\    float od = rb(rsum(n + 1, r) * inv + float(Bv[col + 1]));
    \\    if (which < 2) {
    \\        // Interleaved-pair rotary in f32 (MLX: bf16 * f32 table), rounded once.
    \\        const int pi = (col % 64) / 2, l = r % L;
    \\        const float c = float(COS[l * 32 + pi]), s = float(SIN[l * 32 + pi]);
    \\        const float ne = e * c - od * s, no = e * s + od * c;
    \\        e = ne; od = no;
    \\    }
    \\    OUT[(which * R + r) * 1024 + col] = bfloat16_t(e);
    \\    OUT[(which * R + r) * 1024 + col + 1] = bfloat16_t(od);
    \\}
;

// B: attention + out linear + residual. Every threadgroup recomputes the
// attention (16 heads × R queries × L keys, ~40K MACs) into threadgroup
// memory as bf16 (MLX rounds the sdpa output there too): 8 lanes per
// (query, head), 8 dims each. Two-pass softmax in f32 (MLX: online softmax).
// Grid: 1024 / NO.
const SRC_ATTN = PRELUDE ++
    \\threadgroup bfloat16_t xs[R * 1024];
    \\for (int c = int(t) >> 3; c < R * 16; c += 4 * SG) {
    \\    const int r = c / 16, b = r / L, d = (c % 16) * 64 + (int(lane) & 7) * 8;
    \\    const uint4 qv = *(const device uint4*)(QKV + r * 1024 + d);
    \\    const float4 qa = 0.125f * bf4(qv.xy), qb = 0.125f * bf4(qv.zw);
    \\    float sc[L];
    \\    float m = -INFINITY;
    \\    for (int j = 0; j < L; j++) {
    \\        const uint4 kv = *(const device uint4*)(QKV + (R + b * L + j) * 1024 + d);
    \\        float v = dot(qa, bf4(kv.xy)) + dot(qb, bf4(kv.zw));
    \\        v += simd_shuffle_xor(v, 1);
    \\        v += simd_shuffle_xor(v, 2);
    \\        v += simd_shuffle_xor(v, 4);
    \\        sc[j] = v;
    \\        m = max(m, v);
    \\    }
    \\    float sum = 0.0f;
    \\    float4 oa = 0.0f, oc = 0.0f;
    \\    for (int j = 0; j < L; j++) {
    \\        const float p = metal::exp(sc[j] - m);
    \\        const uint4 vv = *(const device uint4*)(QKV + (2 * R + b * L + j) * 1024 + d);
    \\        sum += p; oa += p * bf4(vv.xy); oc += p * bf4(vv.zw);
    \\    }
    \\    for (int k = 0; k < 4; k++) {
    \\        xs[r * 1024 + d + k] = bfloat16_t(oa[k] / sum);
    \\        xs[r * 1024 + d + 4 + k] = bfloat16_t(oc[k] / sum);
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
++ gemv(1024, "WO + ob * 1024", "bf4(*(threadgroup const uint2*)(xs + r * 1024 + i))", false) ++ SUMS ++
    \\for (int e = int(t); e < NO * R; e += 32 * SG) {
    \\    const int n = e / R, r = e % R, o = ob + n;
    \\    OUT[r * 1024 + o] = bfloat16_t(float(X[r * 1024 + o]) + rb(rsum(n, r) + float(BO[o])));
    \\}
;

// C: rmsnorm2 + ff1 + mlx.nn.gelu_approx (each bf16 op rounded, constants in
// bf16 like the Python weak scalars). Grid: 4096 / NO.
const SRC_FF1 = PRELUDE ++ gemv(1024, "W1 + ob * 1024", "LDW(X, r * 1024 + i)", true) ++ SUMS ++
    \\for (int e = int(t); e < NO * R; e += 32 * SG) {
    \\    const int n = e / R, r = e % R, o = ob + n;
    \\    const float x = rb(rsum(n, r) * rinv(r) + float(B1[o]));
    \\    const float x3 = rb(x * x * x);
    \\    const float inner = rb(x + rb(rb(0.044715f) * x3));
    \\    const float th = rb(metal::precise::tanh(rb(rb(0.7978845608f) * inner)));
    \\    OUT[r * 4096 + o] = bfloat16_t(rb(0.5f * x) * rb(1.0f + th));
    \\}
;

// D: ff2 + residual. Grid: 1024 / NO.
const SRC_FF2 = PRELUDE ++ gemv(4096, "W2 + ob * 4096", "LDW(HH, r * 4096 + i)", false) ++ SUMS ++
    \\for (int e = int(t); e < NO * R; e += 32 * SG) {
    \\    const int n = e / R, r = e % R, o = ob + n;
    \\    OUT[r * 1024 + o] = bfloat16_t(float(X[r * 1024 + o]) + rb(rsum(n, r) + float(B2[o])));
    \\}
;

// E: CFM input rows [t + c, x_embed(hist), x_embed(y)] for both CFG halves.
// TE: [2, 1024] timestep embedding, CE: [2, 1024] c embedding (cond, zeros),
// XH: [1024] x_embed(hist). One thread per column.
const SRC_EMBED =
    \\const int o = int(thread_position_in_grid.x);
    \\float a = 0.0f;
    \\for (int k = 0; k < 64; k += 4)
    \\    a += dot(LDW(WX + o * 64, k), float4(float(Y[k]), float(Y[k + 1]), float(Y[k + 2]), float(Y[k + 3])));
    \\const bfloat16_t xe = bfloat16_t(a + float(BX[o]));
    \\for (int b = 0; b < 2; b++) {
    \\    OUT[(b * 3 + 0) * 1024 + o] = bfloat16_t(float(TE[b * 1024 + o]) + float(CE[b * 1024 + o]));
    \\    OUT[(b * 3 + 1) * 1024 + o] = XH[o];
    \\    OUT[(b * 3 + 2) * 1024 + o] = xe;
    \\}
;

// F: final rmsnorm + linear on each CFG half's last row, CFG combine, Euler
// step (+ SDE noise when SHIFT != 0). One simdgroup per latent channel over
// the full K (norm folded in as in A/C); grid 64 / 4 threadgroups of 128.
// S is the Euler step; CFG/SHIFT are f32 bit patterns.
const SRC_FINAL =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int o = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
    \\const int r0 = (L - 1) * 1024, r1 = (2 * L - 1) * 1024;
    \\float a0 = 0.0f, a1 = 0.0f, s0 = 0.0f, s1 = 0.0f;
    \\for (int b = 0; b < 1024; b += 128) {
    \\    const int i = b + int(lane) * 4;
    \\    const float4 w = LDW(WF + o * 1024, i) * LDW(NW, i);
    \\    const float4 x0 = LDW(X, r0 + i), x1 = LDW(X, r1 + i);
    \\    s0 += dot(x0, x0); s1 += dot(x1, x1);
    \\    a0 += dot(w, x0); a1 += dot(w, x1);
    \\}
    \\a0 = simd_sum(a0); a1 = simd_sum(a1); s0 = simd_sum(s0); s1 = simd_sum(s1);
    \\if (lane == 0) {
    \\    a0 *= metal::precise::rsqrt(s0 / 1024.0f + 1e-6f);
    \\    a1 *= metal::precise::rsqrt(s1 / 1024.0f + 1e-6f);
    \\    const float pc = rb(a0 + float(BF[o])), pu = rb(a1 + float(BF[o]));
    \\    const float f0 = rb(pc + rb(rb(pc - pu) * rb(as_type<float>(CFG))));
    \\    float y = rb(float(Y[o]) + rb(float(DTT[S]) * f0));
    \\    if (SHIFT != 0) y = rb(y + rb(rb(rb(as_type<float>(SHIFT)) * float(SQ[S])) * float(SDE[S * 64 + o])));
    \\    OUT[o] = bfloat16_t(y);
    \\}
;

const Kernel = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    src: [:0]const u8,
    tile: Tile = .{ .no = 8, .sg = 4, .vw = 4 },
    env: ?[*:0]const u8 = null, // tuning override "no,sg,vw"
    k: ?mlx.mlx_fast_metal_kernel = null,

    fn get(self: *Kernel) !mlx.mlx_fast_metal_kernel {
        if (self.k) |k| return k;
        if (self.env) |e| if (std.c.getenv(e)) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            self.tile = .{
                .no = try std.fmt.parseInt(c_int, it.next() orelse "8", 10),
                .sg = try std.fmt.parseInt(c_int, it.next() orelse "4", 10),
                .vw = try std.fmt.parseInt(c_int, it.next() orelse "4", 10),
            };
        };
        const outs = [_][*:0]const u8{"OUT"};
        const iv = mlx.mlx_vector_string_new_data(self.ins.ptr, self.ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new(self.name, iv, ov, self.src.ptr, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        self.k = k;
        return k;
    }
};

var k_qkv: Kernel = .{ .name = "ming_dit_qkv", .ins = &.{ "X", "NW", "WQ", "WK", "WV", "BQ", "BK", "BV", "COS", "SIN" }, .src = SRC_QKV, .tile = .{ .no = 2, .sg = 4, .vw = 4 }, .env = "MING_FT_A" };
var k_attn: Kernel = .{ .name = "ming_dit_attn_out", .ins = &.{ "QKV", "X", "WO", "BO" }, .src = SRC_ATTN, .tile = .{ .no = 8, .sg = 8, .vw = 4 }, .env = "MING_FT_B" };
var k_ff1: Kernel = .{ .name = "ming_dit_ff1", .ins = &.{ "X", "NW", "W1", "B1" }, .src = SRC_FF1, .tile = .{ .no = 2, .sg = 4, .vw = 4 }, .env = "MING_FT_C" };
var k_ff2: Kernel = .{ .name = "ming_dit_ff2", .ins = &.{ "HH", "X", "W2", "B2" }, .src = SRC_FF2, .tile = .{ .no = 2, .sg = 4, .vw = 4 }, .env = "MING_FT_D" };
var k_embed: Kernel = .{ .name = "ming_cfm_embed", .ins = &.{ "Y", "WX", "BX", "TE", "CE", "XH" }, .src = SRC_EMBED };
var k_final: Kernel = .{ .name = "ming_cfm_final", .ins = &.{ "X", "NW", "WF", "BF", "Y", "DTT", "SQ", "SDE" }, .src = SRC_FINAL };

const Tmpl = struct { [*:0]const u8, c_int };

/// Run one kernel → its single bf16 output (owned by the caller). With
/// `n_cols`, a GEMV kernel: n_cols / tile.no threadgroups of tile.sg
/// simdgroups; otherwise `n_cols_or_tgs` threadgroups of 128.
fn run(kr: *Kernel, inputs: []const Arr, out_shape: []const c_int, n_cols: ?c_int, tgs: c_int, tmpl: []const Tmpl, s: S) !Arr {
    const k = try kr.get();
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, .bfloat16));
    var n_tg = tgs;
    var tg: c_int = 128;
    if (n_cols) |n| {
        const tl = kr.tile;
        n_tg = @divExact(n, tl.no);
        tg = 32 * tl.sg;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NO", tl.no));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SG", tl.sg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "VW", tl.vw));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, n_tg * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    for (tmpl) |tp| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, tp[0], tp[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, v, c, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

/// Weights of one DiT block, original [out, in] layout (contiguous).
pub const Block = struct {
    norm1: Arr,
    norm2: Arr,
    wq: Arr,
    wk: Arr,
    wv: Arr,
    bq: Arr,
    bk: Arr,
    bv: Arr,
    wo: Arr,
    bo: Arr,
    w1: Arr,
    b1: Arr,
    w2: Arr,
    b2: Arr,
};

/// Whether x [B, L, 1024] fits the fused DiT kernels.
pub fn eligible(x: Arr, s: S) bool {
    if (!enabled() or !mlx.streamIsGpu(s)) return false;
    const sh = mlx.getShape(x);
    return sh.len == 3 and sh[2] == D and sh[0] * sh[1] <= MAX_ROWS and mlx.mlx_array_dtype(x) == .bfloat16;
}

pub const BlockOut = struct { qkv: Arr, x1: Arr, hh: Arr, x2: Arr };

/// One DiT block, 4 dispatches. Intermediates are returned so the caller's
/// graph scope frees them; `x2` is the block output [B, L, 1024].
pub fn block(b: *const Block, x: Arr, cos: Arr, sin: Arr, s: S) !BlockOut {
    const sh = mlx.getShape(x);
    const L = sh[1];
    const R = sh[0] * L;
    const t: []const Tmpl = &.{ .{ "R", R }, .{ "L", L } };
    const qkv = try run(&k_qkv, &.{ x, b.norm1, b.wq, b.wk, b.wv, b.bq, b.bk, b.bv, cos, sin }, &.{ 3, R, D }, 3 * D, 0, t, s);
    errdefer _ = mlx.mlx_array_free(qkv);
    const x1 = try run(&k_attn, &.{ qkv, x, b.wo, b.bo }, sh, D, 0, t, s);
    errdefer _ = mlx.mlx_array_free(x1);
    const hh = try run(&k_ff1, &.{ x1, b.norm2, b.w1, b.b1 }, &.{ R, FF }, FF, 0, t, s);
    errdefer _ = mlx.mlx_array_free(hh);
    const x2 = try run(&k_ff2, &.{ hh, x1, b.w2, b.b2 }, sh, D, 0, t, s);
    return .{ .qkv = qkv, .x1 = x1, .hh = hh, .x2 = x2 };
}

/// One kernel of `block` alone (0 = A .. 3 = D), for timing.
pub fn single(which: usize, b: *const Block, x: Arr, qkv: Arr, hh: Arr, cos: Arr, sin: Arr, s: S) !Arr {
    const sh = mlx.getShape(x);
    const R = sh[0] * sh[1];
    const t: []const Tmpl = &.{ .{ "R", R }, .{ "L", sh[1] } };
    return switch (which) {
        0 => run(&k_qkv, &.{ x, b.norm1, b.wq, b.wk, b.wv, b.bq, b.bk, b.bv, cos, sin }, &.{ 3, R, D }, 3 * D, 0, t, s),
        1 => run(&k_attn, &.{ qkv, x, b.wo, b.bo }, sh, D, 0, t, s),
        2 => run(&k_ff1, &.{ x, b.norm2, b.w1, b.b1 }, &.{ R, FF }, FF, 0, t, s),
        else => run(&k_ff2, &.{ hh, x, b.w2, b.b2 }, sh, D, 0, t, s),
    };
}

/// CFM input rows h [2, 3, 1024] from latent y [1, 1, 64].
pub fn embed(y: Arr, wx: Arr, bx: Arr, te: Arr, ce: Arr, xh: Arr, s: S) !Arr {
    return run(&k_embed, &.{ y, wx, bx, te, ce, xh }, &.{ 2, 3, D }, null, D / 128, &.{}, s);
}

/// Euler step `step` from DiT output h [2, L, 1024] → new latent [1, 1, 64].
/// dtt / sq: [FLOW_STEPS] step sizes and sqrt|dt|; sde: [FLOW_STEPS, 1, 1, 64].
pub fn final(h: Arr, nw: Arr, wf: Arr, bf: Arr, y: Arr, dtt: Arr, sq: Arr, sde: Arr, step: usize, cfg: f32, shift: f32, s: S) !Arr {
    const L = mlx.getShape(h)[1];
    return run(&k_final, &.{ h, nw, wf, bf, y, dtt, sq, sde }, &.{ 1, 1, 64 }, null, 64 / 4, &.{
        .{ "L", L },
        .{ "S", @intCast(step) },
        .{ "CFG", @bitCast(cfg) },
        .{ "SHIFT", @bitCast(shift) },
    }, s);
}
