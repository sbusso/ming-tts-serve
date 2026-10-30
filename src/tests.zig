//! `zig build test` root: pulls in every file that declares tests.
test {
    _ = @import("ming_audio.zig");
    _ = @import("ming_fr.zig");
    _ = @import("ming_text.zig");
    _ = @import("mlx.zig");
    _ = @import("tokenizer.zig");
    _ = @import("wav.zig");
}
