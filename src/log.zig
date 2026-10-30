//! Leveled logging to stderr for the shared helpers (mlx.zig, tokenizer.zig).
//! Lines at `debug` are dropped; the Ming engine itself logs through std.log.

const std = @import("std");
const builtin = @import("builtin");

pub const Level = enum { err, warn, info, debug };

const current_level: Level = .info;

/// Test builds stay quiet, so a passing test run prints nothing but the
/// test runner's own output.
const stderr_enabled = !builtin.is_test;

fn emit(comptime level: Level, comptime fmt: []const u8, args: anytype) void {
    if (stderr_enabled and @backingInt(current_level) >= @backingInt(level)) std.debug.print(fmt, args);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    emit(.info, fmt, args);
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    emit(.warn, fmt, args);
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    emit(.err, fmt, args);
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    emit(.debug, fmt, args);
}
