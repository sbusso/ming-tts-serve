const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "ming-tts-serve requires Zig 0.17 (have {d}.{d}.{d}).",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // libmlx is built at deployment target 26.2 (scripts/build-mlx.sh, needed
    // for MLX's NAX kernels), so the binaries cannot run on older macOS anyway.
    const target = b.standardTargetOptions(.{
        .default_target = .{ .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } } },
    });
    const optimize = b.standardOptimizeOption(.{});

    // `zig build mlx` builds the pinned MLX + mlx-c submodules into lib/mlx.
    // The script is idempotent (stamp file), so the other steps run it only
    // when lib/mlx is not staged yet; rerun it by hand after a submodule bump.
    const mlx_run = b.addSystemCommand(&.{"bash"});
    mlx_run.addFileArg(b.path("scripts/build-mlx.sh"));
    const mlx_step = b.step("mlx", "Build MLX and mlx-c into lib/mlx (scripts/build-mlx.sh)");
    mlx_step.dependOn(&mlx_run.step);
    const staged = mlxStaged(b);

    // Setting a non-default target field disables Zig's native macOS SDK
    // detection, so resolve the SDK frameworks directory here.
    const frameworks: ?[]const u8 = blk: {
        var code: u8 = undefined;
        const out = b.runAllowFail(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }, &code, .inherit) catch break :blk null;
        const sdk = std.mem.trim(u8, out, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    // The embeddable module (src/ming_service.zig: the API.md routes in
    // process, plus `.engine` for the typed API). The rpath is absolute
    // because a dependent's executable lives outside this package; a relative
    // LC_RPATH would resolve against the launch directory.
    const mod = b.addModule("ming", .{
        .root_source_file = b.path("src/ming_service.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
    linkMlx(b, mod, frameworks);
    const root = b.root.root_dir.handle.realPathFileAlloc(b.graph.io, ".", b.allocator) catch @panic("cannot resolve the build root");
    mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ root, "lib/mlx/lib" }) });

    const mlx_dep: ?*std.Build.Step = if (staged) null else mlx_step;
    addExe(b, target, optimize, frameworks, mlx_dep, "ming-tts", "src/ming_main.zig", "Run the server or CLI (pass args after --)");
    addExe(b, target, optimize, frameworks, mlx_dep, "ming-parity", "src/ming_parity.zig", "Parity check against the Python MLX reference dumps");
    addExe(b, target, optimize, frameworks, mlx_dep, "ming-fused-bench", "src/ming_fused_bench.zig", "Time the fused Metal kernels against plain MLX");

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
    linkMlx(b, test_mod, frameworks);
    test_mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ root, "lib/mlx/lib" }) });
    const tests = b.addTest(.{ .root_module = test_mod });
    if (mlx_dep) |s| tests.step.dependOn(s);
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);
}

/// For dependents, with `.ming_tts_serve = .{ .path = "<this checkout>" }`
/// in their build.zig.zon:
///
///     const ming_tts = @import("ming_tts_serve");
///     const dep = b.dependency("ming_tts_serve", .{ .target = target, .optimize = .ReleaseFast });
///     ming_tts.link(dep, exe); // @import("ming") in exe's root module
///
/// Stage MLX first with `zig build mlx` in this checkout, or let a build of
/// this package do it; `link` also orders the dependent after that step when
/// lib/mlx is missing.
pub fn link(dep: *std.Build.Dependency, compile: *std.Build.Step.Compile) void {
    compile.root_module.addImport("ming", dep.module("ming"));
    if (!mlxStaged(dep.builder)) compile.step.dependOn(&dep.builder.top_level_steps.get("mlx").?.step);
}

fn addExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    frameworks: ?[]const u8,
    mlx_step: ?*std.Build.Step,
    name: []const u8,
    root: []const u8,
    desc: []const u8,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
    linkMlx(b, mod, frameworks);
    // Relative to the binary, independent of the launch directory: the
    // installed zig-out/bin/<exe> and the cached .zig-cache/o/<hash>/<exe>
    // that `zig build <name>` runs sit at different depths below the root.
    mod.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    mod.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
    const exe = b.addExecutable(.{ .name = name, .root_module = mod });
    if (mlx_step) |s| exe.step.dependOn(s);
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    b.step(name, desc).dependOn(&run.step);
}

fn linkMlx(b: *std.Build, mod: *std.Build.Module, frameworks: ?[]const u8) void {
    mod.addIncludePath(b.path("lib/mlx/include"));
    mod.addLibraryPath(b.path("lib/mlx/lib"));
    // No pkg-config: a Homebrew mlx-c must never replace the staged build.
    mod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    if (frameworks) |fw| mod.addFrameworkPath(.{ .cwd_relative = fw });
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
}

fn mlxStaged(b: *std.Build) bool {
    const root = b.root.root_dir.handle;
    for ([_][]const u8{ "lib/mlx/lib/libmlxc.dylib", "lib/mlx/lib/mlx.metallib", "lib/mlx/.version" }) |p| {
        root.access(b.graph.io, p, .{}) catch return false;
    }
    return true;
}
