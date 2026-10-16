//! Bench driver for the t-0582 harness workload (`zig build bench`).
//!
//! OpenAI chat dialect, non-streaming (streaming disables native tool calls
//! in the agent loop). One Agent per session, sequential turns. The host owns
//! every tool call: only `host_echo` (allow) and `host_delete` (deny) exist.
//! JSON snapshots under STORE after every model request (observer hook) and
//! every finished turn, so RESUME=1 finishes an interrupted turn without
//! re-running a finished tool.
const std = @import("std");
const std_compat = @import("compat");
const nullclaw = @import("nullclaw");
const json_util = nullclaw.json_util;

const Agent = nullclaw.agent.Agent;
const providers = nullclaw.providers;
const tools_mod = nullclaw.tools;
const observability = nullclaw.observability;
const Config = nullclaw.config.Config;

const HostCtx = struct {
    session: usize,
    agent: *Agent,
    store: []const u8,
    alloc: std.mem.Allocator,
    out: *std.Io.Writer,
};

fn emit(ctx: *HostCtx, line: []const u8) void {
    ctx.out.writeAll(line) catch {};
    ctx.out.writeByte('\n') catch {};
    ctx.out.flush() catch {};
}

fn emitTool(ctx: *HostCtx, name: []const u8, args_json: []const u8, allowed: bool) void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(ctx.alloc);
    var aw: std.Io.Writer.Allocating = .fromArrayList(ctx.alloc, &buf);
    const w = &aw.writer;
    w.writeAll("{\"ev\":\"tool\",\"s\":") catch return;
    w.print("{d}", .{ctx.session}) catch return;
    w.writeAll(",\"name\":") catch return;
    buf = aw.toArrayList();
    json_util.appendJsonString(&buf, ctx.alloc, name) catch return;
    var aw2: std.Io.Writer.Allocating = .fromArrayList(ctx.alloc, &buf);
    const w2 = &aw2.writer;
    w2.writeAll(",\"args\":") catch return;
    w2.writeAll(args_json) catch return;
    w2.writeAll(",\"allowed\":") catch return;
    w2.writeAll(if (allowed) "true}" else "false}") catch return;
    buf = aw2.toArrayList();
    emit(ctx, buf.items);
}

fn snapshotPath(ctx: *HostCtx, buf: []u8) ![]u8 {
    return std.fmt.bufPrint(buf, "{s}/s{d}.json", .{ ctx.store, ctx.session });
}

/// Persist full history (atomic temp + rename; kill -9 safe).
fn dumpSnapshot(ctx: *HostCtx) void {
    const alloc = ctx.alloc;
    const entries = ctx.agent.dumpHistory(alloc) catch return;
    defer Agent.freeHistoryEntries(alloc, entries);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    var aw: std.Io.Writer.Allocating = .fromArrayList(alloc, &buf);
    const w = &aw.writer;
    w.writeAll("{\"history\":[") catch return;
    for (entries, 0..) |e, i| {
        if (i > 0) w.writeByte(',') catch return;
        w.writeAll("{\"role\":\"") catch return;
        w.writeAll(e.role.toSlice()) catch return;
        w.writeAll("\",\"content\":") catch return;
        buf = aw.toArrayList();
        json_util.appendJsonString(&buf, alloc, e.content) catch return;
        aw = .fromArrayList(alloc, &buf);
        w.writeAll(",\"tool_call_id\":") catch return;
        if (e.tool_call_id) |id| {
            buf = aw.toArrayList();
            json_util.appendJsonString(&buf, alloc, id) catch return;
            aw = .fromArrayList(alloc, &buf);
        } else w.writeAll("null") catch return;
        w.writeAll(",\"tool_calls_json\":") catch return;
        if (e.tool_calls_json) |tc| {
            buf = aw.toArrayList();
            json_util.appendJsonString(&buf, alloc, tc) catch return;
            aw = .fromArrayList(alloc, &buf);
        } else w.writeAll("null") catch return;
        w.writeByte('}') catch return;
    }
    w.writeAll("]}") catch return;
    buf = aw.toArrayList();
    var path_buf: [1024]u8 = undefined;
    const path = snapshotPath(ctx, &path_buf) catch return;
    var tmp_buf: [1032]u8 = undefined;
    const tmp = std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{path}) catch return;
    const f = std_compat.fs.createFileAbsolute(tmp, .{}) catch return;
    f.writeAll(buf.items) catch {
        f.close();
        return;
    };
    f.close();
    std_compat.fs.renameAbsolute(tmp, path) catch {};
}

const EchoTool = struct {
    ctx: *HostCtx,
    pub const tool_name = "host_echo";
    pub const tool_description = "Echo text back from the host.";
    pub const tool_params =
        \\{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
    ;
    pub const vtable = tools_mod.ToolVTable(@This());
    pub fn tool(self: *@This()) tools_mod.Tool {
        return .{ .ptr = self, .vtable = &vtable };
    }
    pub fn execute(self: *@This(), allocator: std.mem.Allocator, args: tools_mod.JsonObjectMap) anyerror!tools_mod.ToolResult {
        const text = tools_mod.getString(args, "text") orelse "";
        var arg_buf: std.ArrayList(u8) = .empty;
        defer arg_buf.deinit(allocator);
        var aaw: std.Io.Writer.Allocating = .fromArrayList(allocator, &arg_buf);
        const aw = &aaw.writer;
        aw.writeAll("{\"text\":") catch {};
        arg_buf = aaw.toArrayList();
        json_util.appendJsonString(&arg_buf, allocator, text) catch {};
        arg_buf.appendSlice(allocator, "}") catch {};
        emitTool(self.ctx, tool_name, arg_buf.items, true);
        const out = try std.fmt.allocPrint(allocator, "echo:{s}", .{text});
        return .{ .success = true, .output = out };
    }
};

const DeleteTool = struct {
    ctx: *HostCtx,
    pub const tool_name = "host_delete";
    pub const tool_description = "Delete a file on the host.";
    pub const tool_params =
        \\{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    ;
    pub const vtable = tools_mod.ToolVTable(@This());
    pub fn tool(self: *@This()) tools_mod.Tool {
        return .{ .ptr = self, .vtable = &vtable };
    }
    pub fn execute(self: *@This(), allocator: std.mem.Allocator, args: tools_mod.JsonObjectMap) anyerror!tools_mod.ToolResult {
        const path = tools_mod.getString(args, "path") orelse "";
        var arg_buf: std.ArrayList(u8) = .empty;
        defer arg_buf.deinit(allocator);
        var aaw: std.Io.Writer.Allocating = .fromArrayList(allocator, &arg_buf);
        const aw = &aaw.writer;
        aw.writeAll("{\"path\":") catch {};
        arg_buf = aaw.toArrayList();
        json_util.appendJsonString(&arg_buf, allocator, path) catch {};
        arg_buf.appendSlice(allocator, "}") catch {};
        emitTool(self.ctx, tool_name, arg_buf.items, false);
        const out = try std.fmt.allocPrint(allocator, "DENIED by host policy: {s} is not allowed", .{tool_name});
        return .{ .success = false, .output = out };
    }
};

const SessionObserver = struct {
    ctx: *HostCtx,
    const vtable = observability.Observer.VTable{
        .record_event = recEvent,
        .record_metric = recMetric,
        .flush = flush,
        .name = name,
        .get_trace_id = getTraceId,
        .set_trace_id = setTraceId,
    };
    pub fn observer(self: *@This()) observability.Observer {
        return .{ .ptr = self, .vtable = &vtable };
    }
    fn recEvent(ptr: *anyopaque, ev: *const observability.ObserverEvent) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (ev.* == .llm_request) dumpSnapshot(self.ctx);
    }
    fn recMetric(_: *anyopaque, _: *const observability.ObserverMetric) void {}
    fn flush(_: *anyopaque) void {}
    fn name(_: *anyopaque) []const u8 {
        return "bench";
    }
    fn getTraceId(_: *anyopaque) ?[32]u8 {
        return null;
    }
    fn setTraceId(_: *anyopaque, _: [32]u8) void {}
};

fn recordUsage(ptr: *anyopaque, rec: Agent.UsageRecord) void {
    const st: *Session = @ptrCast(@alignCast(ptr));
    st.usage_in += rec.usage.prompt_tokens;
    st.usage_out += rec.usage.completion_tokens;
}

const Session = struct {
    id: usize,
    ctx: HostCtx,
    echo: EchoTool,
    delete: DeleteTool,
    obs: SessionObserver,
    tools: [2]tools_mod.Tool,
    agent: Agent,
    usage_in: u64,
    usage_out: u64,
};

const SnapEntry = struct {
    role: []const u8,
    content: []const u8,
    tool_call_id: ?[]const u8 = null,
    tool_calls_json: ?[]const u8 = null,
};

fn restoreFile(alloc: std.mem.Allocator, agent: *Agent, path: []const u8) !bool {
    const file = std_compat.fs.openFileAbsolute(path, .{}) catch return false;
    defer file.close();
    const data = file.readToEndAlloc(alloc, 1 << 26) catch return false;
    defer alloc.free(data);
    const parsed = std.json.parseFromSlice(struct { history: []SnapEntry }, alloc, data, .{}) catch return false;
    defer parsed.deinit();
    var entries: std.ArrayListUnmanaged(Agent.HistoryEntry) = .empty;
    defer entries.deinit(alloc);
    for (parsed.value.history) |e| {
        const role = providers.Role.fromSlice(e.role) orelse continue;
        entries.append(alloc, .{
            .role = role,
            .content = e.content,
            .tool_call_id = e.tool_call_id,
            .tool_calls_json = e.tool_calls_json,
        }) catch return false;
    }
    agent.restoreHistory(entries.items) catch return false;
    return true;
}

fn envOwned(alloc: std.mem.Allocator, key: []const u8) ![]u8 {
    return std_compat.process.getEnvVarOwned(alloc, key) catch {
        std.debug.print("bench: missing env {s}\n", .{key});
        std.process.exit(2);
    };
}

fn envInt(alloc: std.mem.Allocator, key: []const u8, dflt: usize) usize {
    const v = std_compat.process.getEnvVarOwned(alloc, key) catch return dflt;
    defer alloc.free(v);
    return std.fmt.parseInt(usize, v, 10) catch dflt;
}

fn emitTurn(ctx: *HostCtx, s: *Session, k: usize, ms: i64, text: []const u8) void {
    const alloc = ctx.alloc;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    var aw: std.Io.Writer.Allocating = .fromArrayList(alloc, &buf);
    const w = &aw.writer;
    w.print("{{\"ev\":\"turn\",\"s\":{d},\"k\":{d},\"ms\":{d},\"text\":", .{ s.id, k, ms }) catch return;
    buf = aw.toArrayList();
    json_util.appendJsonString(&buf, alloc, text) catch return;
    aw = .fromArrayList(alloc, &buf);
    aw.writer.print(",\"in\":{d},\"out\":{d}}}", .{ s.usage_in, s.usage_out }) catch return;
    buf = aw.toArrayList();
    emit(ctx, buf.items);
}

fn emitCompact(ctx: *HostCtx, s: *Session) void {
    var b: [64]u8 = undefined;
    const line = std.fmt.bufPrint(&b, "{{\"ev\":\"compact\",\"s\":{d}}}", .{s.id}) catch return;
    emit(ctx, line);
}

fn emitError(ctx: *HostCtx, s: *Session, msg: []const u8) void {
    const alloc = ctx.alloc;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    var aw: std.Io.Writer.Allocating = .fromArrayList(alloc, &buf);
    const w = &aw.writer;
    w.print("{{\"ev\":\"error\",\"s\":{d},\"msg\":", .{s.id}) catch return;
    buf = aw.toArrayList();
    json_util.appendJsonString(&buf, alloc, msg) catch return;
    var aw2: std.Io.Writer.Allocating = .fromArrayList(alloc, &buf);
    aw2.writer.writeByte('}') catch return;
    buf = aw2.toArrayList();
    emit(ctx, buf.items);
}

pub fn main(init: std.process.Init) !void {
    std_compat.initProcess(init);
    const alloc = std.heap.smp_allocator;

    var out_buffer: [64 * 1024]u8 = undefined;
    var stdout = std_compat.fs.File.stdout().writerStreaming(&out_buffer);
    const out = &stdout.interface;

    const base = try envOwned(alloc, "OPENAI_BASE");
    const api_key = try envOwned(alloc, "API_KEY");
    const model = try envOwned(alloc, "MODEL");
    const prompt = try envOwned(alloc, "PROMPT");
    const store = try envOwned(alloc, "STORE");
    const n_sessions = envInt(alloc, "SESSIONS", 1);
    const n_turns = envInt(alloc, "TURNS", 1);
    const is_resume = envInt(alloc, "RESUME", 0) == 1;
    const token_limit: u64 = envInt(alloc, "COMPACT_AT", 12000);

    std_compat.fs.makeDirAbsolute(store) catch {};
    var work_buf: [1024]u8 = undefined;
    const workdir = try std.fmt.bufPrint(&work_buf, "{s}/workspace", .{store});
    std_compat.fs.makeDirAbsolute(workdir) catch {};

    var header_lines: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (header_lines.items) |h| alloc.free(h);
        header_lines.deinit(alloc);
    }
    if (std_compat.process.getEnvVarOwned(alloc, "EXTRA_HEADERS") catch null) |eh| {
        defer alloc.free(eh);
        const pv = std.json.parseFromSlice(std.json.Value, alloc, eh, .{}) catch null;
        if (pv) |parsed| {
            defer parsed.deinit();
            if (parsed.value == .object) {
                var it = parsed.value.object.iterator();
                while (it.next()) |kv| {
                    if (header_lines.items.len >= 8) break;
                    if (kv.value_ptr.* != .string) continue;
                    const line = try std.fmt.allocPrint(alloc, "{s}: {s}", .{ kv.key_ptr.*, kv.value_ptr.string });
                    try header_lines.append(alloc, line);
                }
            }
        }
    }

    var compat_prov = providers.compatible.OpenAiCompatibleProvider.init(alloc, "bench", base, api_key, .bearer, null);
    compat_prov.extra_headers = if (header_lines.items.len > 0) header_lines.items else null;
    const provider = compat_prov.provider();

    var cfg = Config{
        .workspace_dir = workdir,
        .config_path = "bench",
        .allocator = alloc,
        .default_model = model,
        .default_temperature = 0,
        .agent = .{
            .max_history_messages = 200,
            .token_limit = token_limit,
            .token_limit_explicit = true,
            .message_timeout_secs = 120,
        },
        .memory = .{ .auto_save = false },
    };

    var sessions: std.ArrayListUnmanaged(*Session) = .empty;
    defer {
        for (sessions.items) |s| {
            s.agent.deinit();
            alloc.destroy(s);
        }
        sessions.deinit(alloc);
    }

    for (0..n_sessions) |id| {
        const s = try alloc.create(Session);
        s.* = .{
            .id = id,
            .ctx = .{ .session = id, .agent = undefined, .store = store, .alloc = alloc, .out = out },
            .echo = undefined,
            .delete = undefined,
            .obs = undefined,
            .tools = undefined,
            .agent = undefined,
            .usage_in = 0,
            .usage_out = 0,
        };
        s.echo = .{ .ctx = &s.ctx };
        s.delete = .{ .ctx = &s.ctx };
        s.obs = .{ .ctx = &s.ctx };
        s.tools = .{ s.echo.tool(), s.delete.tool() };
        s.agent = try Agent.fromConfig(alloc, &cfg, provider, &s.tools, null, s.obs.observer());
        // Keep stdout a clean event stream: NullClaw otherwise prints model
        // chatter without a newline, which corrupts host-protocol lines.
        s.agent.suppress_turn_chatter = true;
        s.ctx.agent = &s.agent;
        var pb: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&pb, "{s}/s{d}.json", .{ store, id });
        _ = restoreFile(alloc, &s.agent, path) catch false;
        try sessions.append(alloc, s);
    }

    emit(&sessions.items[0].ctx, "{\"ev\":\"ready\"}");

    if (is_resume) {
        for (sessions.items) |s| {
            const t0 = std_compat.time.milliTimestamp();
            s.usage_in = 0;
            s.usage_out = 0;
            const resp = s.agent.continueTurn() catch |err| {
                emitError(&s.ctx, s, @errorName(err));
                continue;
            };
            dumpSnapshot(&s.ctx);
            if (s.agent.last_turn_compacted) emitCompact(&s.ctx, s);
            emitTurn(&s.ctx, s, 1, std_compat.time.milliTimestamp() - t0, resp);
        }
    } else {
        for (1..n_turns + 1) |k| {
            for (sessions.items) |s| {
                const t0 = std_compat.time.milliTimestamp();
                s.usage_in = 0;
                s.usage_out = 0;
                const full = try std.fmt.allocPrint(alloc, "{s} [s{d} t{d}]", .{ prompt, s.id, k });
                defer alloc.free(full);
                const resp = s.agent.turn(full) catch |err| {
                    emitError(&s.ctx, s, @errorName(err));
                    continue;
                };
                dumpSnapshot(&s.ctx);
                if (s.agent.last_turn_compacted) emitCompact(&s.ctx, s);
                emitTurn(&s.ctx, s, k, std_compat.time.milliTimestamp() - t0, resp);
            }
        }
    }

    emit(&sessions.items[0].ctx, "{\"ev\":\"idle\"}");
    out.flush() catch {};

    var read_buffer: [1024]u8 = undefined;
    const stdin = std_compat.fs.File.stdin();
    while (true) {
        const n = stdin.read(&read_buffer) catch break;
        if (n == 0) break;
    }
}
