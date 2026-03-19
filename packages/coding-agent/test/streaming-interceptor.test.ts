/**
 * Tests for the streaming interceptor in ExtensionRunner.emitMessageUpdate().
 *
 * Covers: fast path (no interceptor), interception decisions (pass/modify/suppress/abort),
 * subtype routing, timeout safety valve, buffer lifecycle, abort retry limit,
 * and context injection.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.js";
import { ExtensionRunner } from "../src/core/extensions/runner.js";
import type {
	Extension,
	ExtensionRuntime,
	MessageUpdateEvent,
	StreamDecision,
} from "../src/core/extensions/types.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";

// ============================================================================
// Helpers
// ============================================================================

/** Create a MessageUpdateEvent with assistantMessageEvent.type === "text_delta". */
function makeTextDelta(delta: string): MessageUpdateEvent {
	return {
		type: "message_update",
		message: { role: "assistant", content: [{ type: "text", text: delta }] } as AgentMessage,
		assistantMessageEvent: {
			type: "text_delta",
			contentIndex: 0,
			delta,
			partial: { role: "assistant", content: [{ type: "text", text: delta }] },
		} as MessageUpdateEvent["assistantMessageEvent"],
	};
}

/** Create a MessageUpdateEvent with assistantMessageEvent.type === "text_end". */
function makeTextEnd(): MessageUpdateEvent {
	return {
		type: "message_update",
		message: { role: "assistant", content: [{ type: "text", text: "" }] } as AgentMessage,
		assistantMessageEvent: {
			type: "text_end",
			contentIndex: 0,
			content: "",
			partial: { role: "assistant", content: [{ type: "text", text: "" }] },
		} as MessageUpdateEvent["assistantMessageEvent"],
	};
}

/** Create a MessageUpdateEvent with assistantMessageEvent.type === "thinking_delta". */
function makeThinkingDelta(delta: string): MessageUpdateEvent {
	return {
		type: "message_update",
		message: { role: "assistant", content: [] } as AgentMessage,
		assistantMessageEvent: {
			type: "thinking_delta",
			contentIndex: 0,
			delta,
			partial: { role: "assistant", content: [] },
		} as MessageUpdateEvent["assistantMessageEvent"],
	};
}

/** Create a MessageUpdateEvent with assistantMessageEvent.type === "done". */
function makeDone(): MessageUpdateEvent {
	return {
		type: "message_update",
		message: { role: "assistant", content: [] } as AgentMessage,
		assistantMessageEvent: {
			type: "done",
			reason: "stop",
			message: { role: "assistant", content: [] },
		} as MessageUpdateEvent["assistantMessageEvent"],
	};
}

/** Build a minimal Extension with a single message_update handler. */
function createExtension(handler: (event: MessageUpdateEvent, ctx: any) => any): Extension {
	const handlers = new Map<string, ((...args: unknown[]) => Promise<unknown>)[]>();
	handlers.set("message_update", [handler as (...args: unknown[]) => Promise<unknown>]);
	return {
		path: "test-interceptor-ext",
		resolvedPath: "/tmp/test-interceptor-ext",
		handlers,
		tools: new Map(),
		messageRenderers: new Map(),
		commands: new Map(),
		flags: new Map(),
		shortcuts: new Map(),
	};
}

/** Create a minimal ExtensionRuntime with no-op action stubs. */
function createRuntime(): ExtensionRuntime {
	return {
		flagValues: new Map(),
		pendingProviderRegistrations: [],
		sendMessage: () => {},
		sendUserMessage: () => {},
		appendEntry: () => {},
		setSessionName: () => {},
		getSessionName: () => undefined,
		setLabel: () => {},
		getActiveTools: () => [],
		getAllTools: () => [],
		setActiveTools: () => {},
		getCommands: () => [],
		setModel: async () => false,
		getThinkingLevel: () => "off" as const,
		setThinkingLevel: () => {},
	};
}

/** Call runner.bindCore() with no-op defaults plus optional overrides. */
function bindCoreDefaults(runner: ExtensionRunner, overrides?: { abort?: () => void }): void {
	runner.bindCore(
		{
			sendMessage: () => {},
			sendUserMessage: () => {},
			appendEntry: () => {},
			setSessionName: () => {},
			getSessionName: () => undefined,
			setLabel: () => {},
			getActiveTools: () => [],
			getAllTools: () => [],
			setActiveTools: () => {},
			getCommands: () => [],
			setModel: async () => false,
			getThinkingLevel: () => "off" as const,
			setThinkingLevel: () => {},
		},
		{
			getModel: () => undefined,
			isIdle: () => true,
			abort: overrides?.abort ?? (() => {}),
			hasPendingMessages: () => false,
			shutdown: () => {},
			getContextUsage: () => undefined,
			compact: () => {},
			getSystemPrompt: () => "",
		},
	);
}

// ============================================================================
// Tests
// ============================================================================

describe("Streaming Interceptor", () => {
	let tempDir: string;
	let sessionManager: SessionManager;
	let modelRegistry: ModelRegistry;

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-stream-test-"));
		sessionManager = SessionManager.inMemory();
		const authStorage = AuthStorage.create(path.join(tempDir, "auth.json"));
		modelRegistry = new ModelRegistry(authStorage);
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	// ========================================================================
	// No interceptor (fast path)
	// ========================================================================

	describe("no interceptor (fast path)", () => {
		it("passes through text_delta when no handlers registered", async () => {
			const ext: Extension = {
				path: "empty-ext",
				resolvedPath: "/tmp/empty-ext",
				handlers: new Map(),
				tools: new Map(),
				messageRenderers: new Map(),
				commands: new Map(),
				flags: new Map(),
				shortcuts: new Map(),
			};
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			const event = makeTextDelta("hello");
			const result = await runner.emitMessageUpdate(event);

			expect(result.outcome).toBe("emit");
			expect((result as any).event).toBe(event);
		});

		it("passes through and calls observer returning void", async () => {
			const observed: MessageUpdateEvent[] = [];
			const ext = createExtension(async (event) => {
				observed.push(event);
				// Return void = observer only
				return undefined;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			const event = makeTextDelta("world");
			const result = await runner.emitMessageUpdate(event);

			expect(result.outcome).toBe("emit");
			expect((result as any).event).toBe(event);
			expect(observed).toHaveLength(1);
			expect(observed[0]).toBe(event);
		});
	});

	// ========================================================================
	// Interceptor active
	// ========================================================================

	describe("interceptor active", () => {
		it("holds tokens when interceptor returns void (after being marked)", async () => {
			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				// First call returns a decision to mark _hasInterceptor, subsequent calls return void
				if (callCount === 1) {
					return { action: "pass" } as StreamDecision;
				}
				return undefined;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			// First call: triggers pass (marks _hasInterceptor=true)
			const r1 = await runner.emitMessageUpdate(makeTextDelta("a"));
			expect(r1.outcome).toBe("emit");

			// Second call: interceptor is active but returns void => hold
			const r2 = await runner.emitMessageUpdate(makeTextDelta("b"));
			expect(r2.outcome).toBe("hold");
		});

		it("flushes buffer on pass decision", async () => {
			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				// First call activates interceptor via pass, second holds, third passes
				if (callCount <= 1) {
					return { action: "pass" } as StreamDecision;
				}
				if (callCount === 2) return undefined; // hold
				return { action: "pass" } as StreamDecision;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			// Activate interceptor
			await runner.emitMessageUpdate(makeTextDelta("a"));

			// Buffer a token (hold)
			await runner.emitMessageUpdate(makeTextDelta("b"));

			// Now pass - should flush the buffered "b" plus current
			const result = await runner.emitMessageUpdate(makeTextDelta("c"));
			expect(result.outcome).toBe("emit");
			// The flushed event should contain the accumulated buffer
			const emitted = result as { outcome: "emit"; event: MessageUpdateEvent };
			expect((emitted.event.assistantMessageEvent as any).delta).toBe("bc");
		});

		it("modifies buffer on modify decision (check delta equals REDACTED)", async () => {
			const ext = createExtension(async (_event) => {
				return { action: "modify", text: "REDACTED" } as StreamDecision;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			const result = await runner.emitMessageUpdate(makeTextDelta("secret data"));
			expect(result.outcome).toBe("emit_modified");
			const modified = result as { outcome: "emit_modified"; event: MessageUpdateEvent };
			expect((modified.event.assistantMessageEvent as any).delta).toBe("REDACTED");
		});

		it("suppresses buffer on suppress decision", async () => {
			const ext = createExtension(async (_event) => {
				return { action: "suppress" } as StreamDecision;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			const result = await runner.emitMessageUpdate(makeTextDelta("suppressed"));
			expect(result.outcome).toBe("suppressed");
		});

		it("aborts on abort decision (check abortFn called, consumePendingAbortReason returns reason)", async () => {
			const abortFn = vi.fn();
			const ext = createExtension(async (_event) => {
				return { action: "abort", reason: "dangerous content" } as StreamDecision;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner, { abort: abortFn });

			const result = await runner.emitMessageUpdate(makeTextDelta("bad"));
			expect(result.outcome).toBe("aborted");
			expect(abortFn).toHaveBeenCalledOnce();

			const reason = runner.consumePendingAbortReason();
			expect(reason).toBe("dangerous content");
			// Should clear after consumption
			expect(runner.consumePendingAbortReason()).toBeUndefined();
		});
	});

	// ========================================================================
	// Subtype routing
	// ========================================================================

	describe("subtype routing", () => {
		it("passes through thinking_delta without buffering", async () => {
			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				// First call activates interceptor
				if (callCount === 1) return { action: "pass" } as StreamDecision;
				return undefined;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			// Activate the interceptor
			await runner.emitMessageUpdate(makeTextDelta("a"));

			// thinking_delta should pass through directly
			const event = makeThinkingDelta("thinking...");
			const result = await runner.emitMessageUpdate(event);
			expect(result.outcome).toBe("emit");
			expect((result as any).event).toBe(event);
		});

		it("force-flushes buffer on text_end", async () => {
			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				if (callCount === 1) return { action: "pass" } as StreamDecision;
				return undefined; // hold subsequent
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			// Activate interceptor
			await runner.emitMessageUpdate(makeTextDelta("a"));

			// Buffer a token
			await runner.emitMessageUpdate(makeTextDelta("b"));

			// text_end should force flush
			const result = await runner.emitMessageUpdate(makeTextEnd());
			expect(result.outcome).toBe("emit");
			const emitted = result as { outcome: "emit"; event: MessageUpdateEvent; flushedEvent?: MessageUpdateEvent };
			// flushedEvent should contain the buffered tokens
			expect(emitted.flushedEvent).toBeDefined();
			expect((emitted.flushedEvent!.assistantMessageEvent as any).delta).toBe("b");
		});

		it("force-flushes buffer on done", async () => {
			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				if (callCount === 1) return { action: "pass" } as StreamDecision;
				return undefined;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			// Activate interceptor
			await runner.emitMessageUpdate(makeTextDelta("a"));
			// Buffer
			await runner.emitMessageUpdate(makeTextDelta("b"));

			// done should force flush
			const result = await runner.emitMessageUpdate(makeDone());
			expect(result.outcome).toBe("emit");
			const emitted = result as { outcome: "emit"; event: MessageUpdateEvent; flushedEvent?: MessageUpdateEvent };
			expect(emitted.flushedEvent).toBeDefined();
		});
	});

	// ========================================================================
	// Timeout safety valve
	// ========================================================================

	describe("timeout safety valve", () => {
		it("auto-flushes after maxHoldMs", async () => {
			vi.useFakeTimers();

			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				if (callCount === 1) return { action: "pass" } as StreamDecision;
				return undefined; // hold
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			const autoFlushed: MessageUpdateEvent[] = [];
			runner.setAutoFlushCallback((event) => {
				autoFlushed.push(event);
			});

			// Activate interceptor
			await runner.emitMessageUpdate(makeTextDelta("a"));

			// Buffer a token (hold)
			await runner.emitMessageUpdate(makeTextDelta("b"));

			// Advance past maxHoldMs (500ms default)
			vi.advanceTimersByTime(600);

			expect(autoFlushed).toHaveLength(1);
			expect((autoFlushed[0].assistantMessageEvent as any).delta).toBe("b");

			vi.useRealTimers();
		});
	});

	// ========================================================================
	// Buffer lifecycle
	// ========================================================================

	describe("buffer lifecycle", () => {
		it("flushAndClearBuffer clears state", async () => {
			let callCount = 0;
			const ext = createExtension(async (_event) => {
				callCount++;
				if (callCount === 1) return { action: "pass" } as StreamDecision;
				return undefined; // hold
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			// Activate interceptor
			await runner.emitMessageUpdate(makeTextDelta("a"));
			// Buffer a token
			await runner.emitMessageUpdate(makeTextDelta("b"));

			// Flush and clear
			const flushed = runner.flushAndClearBuffer();
			expect(flushed).toBeDefined();
			expect((flushed!.assistantMessageEvent as any).delta).toBe("b");

			// Buffer should be empty now
			const flushed2 = runner.flushAndClearBuffer();
			expect(flushed2).toBeUndefined();
		});

		it("consumePendingAbortReason returns and clears reason", async () => {
			const abortFn = vi.fn();
			const ext = createExtension(async (_event) => {
				return { action: "abort", reason: "test reason" } as StreamDecision;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner, { abort: abortFn });

			await runner.emitMessageUpdate(makeTextDelta("x"));

			// First call returns the reason
			const reason = runner.consumePendingAbortReason();
			expect(reason).toBe("test reason");

			// Second call should return undefined (cleared)
			expect(runner.consumePendingAbortReason()).toBeUndefined();
		});
	});

	// ========================================================================
	// Abort retry limit
	// ========================================================================

	describe("abort retry limit", () => {
		it("downgrades abort to suppress after 3 aborts", async () => {
			const abortFn = vi.fn();
			const ext = createExtension(async (_event) => {
				return { action: "abort", reason: "blocked" } as StreamDecision;
			});
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner, { abort: abortFn });

			const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

			// First 3 aborts should succeed
			const r1 = await runner.emitMessageUpdate(makeTextDelta("a"));
			expect(r1.outcome).toBe("aborted");

			const r2 = await runner.emitMessageUpdate(makeTextDelta("b"));
			expect(r2.outcome).toBe("aborted");

			const r3 = await runner.emitMessageUpdate(makeTextDelta("c"));
			expect(r3.outcome).toBe("aborted");

			expect(abortFn).toHaveBeenCalledTimes(3);

			// 4th abort should be downgraded to suppress
			const r4 = await runner.emitMessageUpdate(makeTextDelta("d"));
			expect(r4.outcome).toBe("suppressed");
			expect(abortFn).toHaveBeenCalledTimes(3); // no additional abort call

			expect(warnSpy).toHaveBeenCalledWith(
				expect.stringContaining("abort limit"),
			);

			warnSpy.mockRestore();
		});
	});

	// ========================================================================
	// Context injection
	// ========================================================================

	describe("context injection", () => {
		it("injectContextMessage adds text that emitContext picks up", async () => {
			const ext: Extension = {
				path: "ctx-ext",
				resolvedPath: "/tmp/ctx-ext",
				handlers: new Map(),
				tools: new Map(),
				messageRenderers: new Map(),
				commands: new Map(),
				flags: new Map(),
				shortcuts: new Map(),
			};
			const runner = new ExtensionRunner([ext], createRuntime(), tempDir, sessionManager, modelRegistry);
			bindCoreDefaults(runner);

			runner.injectContextMessage("Previous response was blocked for safety.");

			const messages: AgentMessage[] = [
				{ role: "user", content: "hello" } as AgentMessage,
			];
			const result = await runner.emitContext(messages);

			// Should have original message plus injected context
			expect(result).toHaveLength(2);
			expect(result[1]).toMatchObject({
				role: "user",
				content: [{ type: "text", text: "[System: Previous response was blocked for safety.]" }],
			});

			// Second call should not inject again (consumed)
			const result2 = await runner.emitContext(messages);
			expect(result2).toHaveLength(1);
		});
	});
});
