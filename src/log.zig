const std = @import("std");

pub const Level = enum {
    err,
    warn,
    info,
    debug,

    pub fn fromString(s: []const u8) ?Level {
        if (std.mem.eql(u8, s, "error")) return .err;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "debug")) return .debug;
        return null;
    }
};

var current_level: Level = .info;

pub fn setLevel(level: Level) void {
    current_level = level;
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.info)) {
        std.debug.print(fmt, args);
    }
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.warn)) {
        std.debug.print(fmt, args);
    }
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.err)) {
        std.debug.print(fmt, args);
    }
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.debug)) {
        std.debug.print(fmt, args);
    }
}
