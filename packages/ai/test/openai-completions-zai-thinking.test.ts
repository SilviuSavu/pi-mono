import { describe, expect, it, vi } from "vitest";
import { getModel } from "../src/models.js";
import { streamSimple } from "../src/stream.js";

const mockState = vi.hoisted(() => ({ lastParams: undefined as unknown }));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: async (params: unknown) => {
					mockState.lastParams = params;
					return {
						async *[Symbol.asyncIterator]() {
							yield {
								choices: [{ delta: {}, finish_reason: "stop" }],
								usage: {
									prompt_tokens: 1,
									completion_tokens: 1,
									prompt_tokens_details: { cached_tokens: 0 },
									completion_tokens_details: { reasoning_tokens: 0 },
								},
							};
						},
					};
				},
			},
		};
	}

	return { default: FakeOpenAI };
});

describe("openai-completions Z.ai thinking payload", () => {
	it("adds clear_history when preserveThinking is enabled", async () => {
		const model = getModel("zai", "glm-5")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "hello",
						timestamp: Date.now(),
					},
				],
			},
			{
				apiKey: "test",
				reasoning: "medium",
				preserveThinking: true,
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as {
			thinking?: { type?: string; clear_history?: boolean; clear_thinking?: boolean };
		};
		expect(params.thinking).toEqual({
			type: "enabled",
			clear_history: false,
		});
	});

	it("keeps Z.ai thinking binary format without clear flags when preserveThinking is disabled", async () => {
		const model = getModel("zai", "glm-5")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "hello",
						timestamp: Date.now(),
					},
				],
			},
			{
				apiKey: "test",
				reasoning: "medium",
				preserveThinking: false,
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as {
			thinking?: { type?: string; clear_history?: boolean; clear_thinking?: boolean };
		};
		expect(params.thinking?.type).toBe("enabled");
		expect(params.thinking?.clear_history).toBeUndefined();
		expect(params.thinking?.clear_thinking).toBeUndefined();
	});
});
