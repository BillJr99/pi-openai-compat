/**
 * Regression tests for @billjr99/pi-openai-compat.
 *
 * Runs on node:test with Node's built-in TypeScript type stripping, so there
 * is no build step and no test-framework dependency. index.ts imports pi only
 * as `import type`, which is erased at runtime, and has no top-level side
 * effects, so importing it here is free.
 *
 *   node --test test/
 *
 * Each suite below is tied to a specific defect so the fix cannot silently
 * regress. Where a test encodes a rule enforced by pi rather than by this
 * extension, the pi source is cited in a comment.
 */

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import {
  TEMPLATES,
  KEYLESS_PLACEHOLDER,
  MAX_CONTEXT_WINDOW,
  MAX_OUTPUT_TOKENS,
  MAX_ERROR_BODY,
  CONFIG_FILE_MODE,
  buildProviderModels,
  compatKey,
  fetchModels,
  isLocalUrl,
  mergeModelMetadata,
  normalizeInput,
  normalizeTokenCount,
  registerProvider,
  restrictPermissions,
  tryRegisterProvider,
} from "../index.ts";

/**
 * pi's ModelRegistry.validateProviderConfig, reproduced from
 * @mariozechner/pi-coding-agent dist/core/model-registry.js. A provider that
 * defines models is rejected outright unless apiKey is a non-empty string,
 * and registerProvider throws before any /compat-* command is registered.
 */
function piWouldAccept(config: { baseUrl?: string; apiKey?: unknown; models?: unknown[] }): boolean {
  if (!config.models || config.models.length === 0) return true;
  if (!config.baseUrl) return false;
  if (!config.apiKey) return false;
  return true;
}

/** Minimal ExtensionAPI stub capturing what registerProvider hands to pi. */
function fakePi() {
  const calls: Array<{ key: string; config: any }> = [];
  const pi = {
    registerProvider(key: string, config: any) {
      // Mirror pi's own rejection so a regression surfaces as a throw here too.
      if (!piWouldAccept(config)) {
        throw new Error(`Provider ${key}: "apiKey" or "oauth" is required when defining models.`);
      }
      calls.push({ key, config });
    },
  };
  return { pi: pi as any, calls };
}

const oneModel = [{ id: "llama3" }];

// ─────────────────────────────────────────────────────────────────────────────
describe("Finding 1 — keyless providers must still register (PR #30 regression)", () => {
  test("a keyless provider is given a non-empty placeholder key", () => {
    const { pi, calls } = fakePi();
    registerProvider(pi, "ollama", {
      displayName: "Ollama (local)",
      baseUrl: "http://localhost:11434/v1",
      apiKey: null,
      cachedModels: oneModel,
    });
    assert.equal(calls.length, 1, "provider should have registered");
    assert.equal(calls[0].config.apiKey, KEYLESS_PLACEHOLDER);
    assert.equal(typeof calls[0].config.apiKey, "string");
    assert.ok(calls[0].config.apiKey.length > 0, "pi rejects a falsy apiKey");
  });

  test("a real key is passed through untouched", () => {
    const { pi, calls } = fakePi();
    registerProvider(pi, "openrouter", {
      displayName: "OpenRouter",
      baseUrl: "https://openrouter.ai/api/v1",
      apiKey: "sk-or-secret",
      cachedModels: oneModel,
    });
    assert.equal(calls[0].config.apiKey, "sk-or-secret");
  });

  test("the placeholder does not depend on the hostname being local", () => {
    // The whole point of PR #30: a .local or LAN host may still be
    // key-protected, so registration must not branch on isLocalUrl.
    const { pi, calls } = fakePi();
    for (const baseUrl of [
      "http://localhost:11434/v1",
      "http://mac-mini.local:11434/v1",
      "https://ollama.example.com/v1",
    ]) {
      registerProvider(pi, "k", {
        displayName: "x", baseUrl, apiKey: null, cachedModels: oneModel,
      });
    }
    assert.equal(calls.length, 3);
    for (const c of calls) assert.equal(c.config.apiKey, KEYLESS_PLACEHOLDER);
  });

  test("every keyless template can actually register", () => {
    const { pi, calls } = fakePi();
    for (const [key, tpl] of Object.entries(TEMPLATES)) {
      if (!tpl.keyless) continue;
      registerProvider(pi, key, {
        displayName: tpl.displayName,
        baseUrl: tpl.baseUrl || "http://localhost:1/v1",
        apiKey: null,
        cachedModels: oneModel,
      });
    }
    assert.ok(calls.every((c) => piWouldAccept(c.config)));
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("Finding 2 — one bad provider must not take down the extension", () => {
  test("tryRegisterProvider reports failure instead of throwing", () => {
    const exploding = {
      registerProvider() { throw new Error("boom"); },
    } as any;
    let ok: boolean | undefined;
    assert.doesNotThrow(() => {
      ok = tryRegisterProvider(exploding, "bad", {
        displayName: "bad", baseUrl: "https://x/v1", apiKey: "k", cachedModels: oneModel,
      });
    });
    assert.equal(ok, false);
  });

  test("a failing provider does not stop the ones after it", () => {
    const seen: string[] = [];
    const pi = {
      registerProvider(key: string) {
        if (key === "compat-bad") throw new Error("boom");
        seen.push(key);
      },
    } as any;
    const providers = {
      bad: { displayName: "bad", baseUrl: "https://x/v1", apiKey: "k", cachedModels: oneModel },
      good: { displayName: "good", baseUrl: "https://y/v1", apiKey: "k", cachedModels: oneModel },
    };
    for (const [key, p] of Object.entries(providers)) tryRegisterProvider(pi, key, p);
    assert.deepEqual(seen, [compatKey("good")]);
  });

  test("tryRegisterProvider succeeds on a valid provider", () => {
    const { pi } = fakePi();
    assert.equal(
      tryRegisterProvider(pi, "ok", {
        displayName: "ok", baseUrl: "https://x/v1", apiKey: null, cachedModels: oneModel,
      }),
      true,
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("Finding 3 — locally-hosted templates must still offer a key prompt", () => {
  test("ollama and llmproxy are not marked keyless", () => {
    // They are routinely exposed on a LAN/.local host behind a key. Marking
    // them keyless skips the wizard's key prompt with no way to supply one.
    for (const key of ["ollama", "llmproxy"]) {
      assert.equal(TEMPLATES[key].keyless, false, `${key} must prompt for an optional key`);
    }
  });

  test("a template that skips the key prompt never advertises a key source", () => {
    for (const [key, tpl] of Object.entries(TEMPLATES)) {
      if (tpl.keyless) {
        assert.equal(tpl.keyHint, undefined, `${key}: keyless template must not have a keyHint`);
      }
    }
  });

  test("every template has a usable baseUrl or prompts for one", () => {
    for (const [key, tpl] of Object.entries(TEMPLATES)) {
      assert.ok(tpl.baseUrl || tpl.promptUrl, `${key}: needs a baseUrl or promptUrl`);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("Finding 5 — config permissions (API keys are stored in cleartext)", () => {
  test("restrictPermissions tightens a world-readable file", (t) => {
    if (process.platform === "win32") return t.skip("POSIX modes only");
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "compat-perm-"));
    const file = path.join(dir, "config.json");
    fs.writeFileSync(file, "{}", { mode: 0o644 });
    fs.chmodSync(file, 0o644);
    restrictPermissions(file, CONFIG_FILE_MODE);
    assert.equal(fs.statSync(file).mode & 0o777, CONFIG_FILE_MODE);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("restrictPermissions leaves an already-strict file alone", (t) => {
    if (process.platform === "win32") return t.skip("POSIX modes only");
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "compat-perm-"));
    const file = path.join(dir, "config.json");
    fs.writeFileSync(file, "{}", { mode: 0o600 });
    fs.chmodSync(file, 0o400);
    restrictPermissions(file, CONFIG_FILE_MODE);
    // 0400 is narrower than 0600, so it must not be widened.
    assert.equal(fs.statSync(file).mode & 0o777, 0o400);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("restrictPermissions does not throw on a missing path", () => {
    assert.doesNotThrow(() => restrictPermissions("/nonexistent/compat/config.json", 0o600));
  });

  test("the config file mode is owner-only", () => {
    assert.equal(CONFIG_FILE_MODE & 0o077, 0, "config.json must not be group/world accessible");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("Finding 6 — untrusted catalog metadata is validated", () => {
  test("non-numeric token counts are rejected", () => {
    for (const bad of ["128000", null, undefined, NaN, Infinity, -1, 0, {}, []]) {
      assert.equal(normalizeTokenCount(bad, MAX_CONTEXT_WINDOW), undefined, `rejects ${String(bad)}`);
    }
  });

  test("absurd token counts are clamped, not trusted", () => {
    assert.equal(normalizeTokenCount(100_000_000, MAX_CONTEXT_WINDOW), MAX_CONTEXT_WINDOW);
    assert.equal(normalizeTokenCount(5_000_000, MAX_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS);
  });

  test("sane values pass through, fractions floored", () => {
    assert.equal(normalizeTokenCount(128_000, MAX_CONTEXT_WINDOW), 128_000);
    assert.equal(normalizeTokenCount(4096.7, MAX_OUTPUT_TOKENS), 4096);
  });

  test("a hostile cachedModels entry cannot reach pi unvalidated", () => {
    const [m] = buildProviderModels([
      { id: "evil", contextWindow: 1e12, maxTokens: "9999" as any },
    ]);
    assert.equal(m.contextWindow, MAX_CONTEXT_WINDOW);
    assert.equal(m.maxTokens, 4_096, "a string maxTokens falls back to the default");
  });

  test("defaults still apply when the catalog omits the fields", () => {
    const [m] = buildProviderModels([{ id: "plain" }]);
    assert.equal(m.contextWindow, 128_000);
    assert.equal(m.maxTokens, 4_096);
    assert.equal(m.reasoning, false);
    assert.deepEqual(m.input, ["text"]);
  });

  test("normalizeInput drops unknown modalities (PR #24/#25)", () => {
    assert.deepEqual(normalizeInput(["text", "image"]), ["text", "image"]);
    assert.deepEqual(normalizeInput(["text", "video", "audio"]), ["text"]);
    assert.equal(normalizeInput(["video"]), undefined);
    assert.equal(normalizeInput("text"), undefined);
    assert.equal(normalizeInput(null), undefined);
    assert.equal(normalizeInput([]), undefined);
  });

  test("fetchModels sanitizes a hostile /models payload end to end", async (t) => {
    const original = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: true,
      json: async () => ({
        data: [
          { id: "evil", context_window: 1e12, max_tokens: "abc", reasoning: "yes", input: ["video"] },
          { id: "fine", context_window: 200_000, max_tokens: 8_192, reasoning: true, input: ["text", "image"] },
        ],
      }),
    })) as any;
    t.after(() => { globalThis.fetch = original; });

    const models = await fetchModels("https://hostile.example/v1", null);
    const evil = models.find((m) => m.id === "evil")!;
    assert.equal(evil.contextWindow, MAX_CONTEXT_WINDOW, "clamped");
    assert.equal(evil.maxTokens, undefined, "string max_tokens dropped");
    assert.equal(evil.reasoning, undefined, "non-boolean reasoning dropped");
    assert.equal(evil.input, undefined, "unknown modality dropped");

    const fine = models.find((m) => m.id === "fine")!;
    assert.equal(fine.contextWindow, 200_000);
    assert.equal(fine.maxTokens, 8_192);
    assert.equal(fine.reasoning, true);
    assert.deepEqual(fine.input, ["text", "image"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("Finding 7 — upstream error bodies are truncated", () => {
  test("a huge error body is capped before it reaches the UI", async (t) => {
    const original = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: false,
      status: 401,
      text: async () => "S".repeat(50_000),
    })) as any;
    t.after(() => { globalThis.fetch = original; });

    await assert.rejects(
      () => fetchModels("https://x.example/v1", "sk-test"),
      (err: Error) => {
        assert.ok(err.message.length < MAX_ERROR_BODY + 200, "error message must stay bounded");
        assert.ok(err.message.includes("truncated"));
        assert.ok(err.message.includes("401"));
        return true;
      },
    );
  });

  test("a short error body is preserved verbatim for diagnosis", async (t) => {
    const original = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: false, status: 404, text: async () => "no such route",
    })) as any;
    t.after(() => { globalThis.fetch = original; });

    await assert.rejects(
      () => fetchModels("https://x.example/v1", null),
      /404.*no such route/s,
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("supporting behavior relied on by the fixes", () => {
  test("mergeModelMetadata carries hand-edited fields forward (PR #25/#28)", () => {
    const merged = mergeModelMetadata(
      [{ id: "a", contextWindow: 1_000_000, reasoning: true, input: ["text", "image"],
         thinkingLevelMap: { high: "high" }, compat: { maxTokensField: "max_tokens" } }],
      [{ id: "a" }, { id: "b" }],
    );
    const a = merged.find((m) => m.id === "a")!;
    assert.equal(a.contextWindow, 1_000_000);
    assert.equal(a.reasoning, true);
    assert.deepEqual(a.input, ["text", "image"]);
    assert.deepEqual(a.thinkingLevelMap, { high: "high" });
    assert.deepEqual(a.compat, { maxTokensField: "max_tokens" });
    assert.ok(merged.find((m) => m.id === "b"), "new ids are kept");
  });

  test("a field the provider reports wins over the cached value", () => {
    const merged = mergeModelMetadata(
      [{ id: "a", contextWindow: 1_000 }],
      [{ id: "a", contextWindow: 2_000 }],
    );
    assert.equal(merged[0].contextWindow, 2_000);
  });

  test("a model dropped upstream is dropped from the cache", () => {
    const merged = mergeModelMetadata([{ id: "old" }], [{ id: "new" }]);
    assert.deepEqual(merged.map((m) => m.id), ["new"]);
  });

  test("passthrough fields are only emitted when present", () => {
    const [bare] = buildProviderModels([{ id: "x" }]);
    assert.ok(!("thinkingLevelMap" in bare));
    assert.ok(!("compat" in bare));
    const [full] = buildProviderModels([{ id: "x", compat: { supportsDeveloperRole: false } }]);
    assert.deepEqual((full as any).compat, { supportsDeveloperRole: false });
  });

  test("isLocalUrl is still correct for the prompt hint it now only feeds", () => {
    assert.equal(isLocalUrl("http://localhost:11434/v1"), true);
    assert.equal(isLocalUrl("http://127.0.0.1:8080/v1"), true);
    assert.equal(isLocalUrl("http://mac.local:11434/v1"), true);
    assert.equal(isLocalUrl("https://openrouter.ai/api/v1"), false);
    assert.equal(isLocalUrl("not a url"), false);
  });

  test("compatKey namespaces providers", () => {
    assert.equal(compatKey("ollama"), "compat-ollama");
  });

  test("fetchModels accepts the four documented catalog shapes", async (t) => {
    const original = globalThis.fetch;
    t.after(() => { globalThis.fetch = original; });
    for (const payload of [
      { data: [{ id: "m" }] },
      { result: [{ id: "m" }] },
      { models: [{ id: "m" }] },
      [{ id: "m" }],
    ]) {
      globalThis.fetch = (async () => ({ ok: true, json: async () => payload })) as any;
      const models = await fetchModels("https://x.example/v1", null);
      assert.deepEqual(models.map((m) => m.id), ["m"]);
    }
  });

  test("fetchModels rejects an unrecognized catalog shape", async (t) => {
    const original = globalThis.fetch;
    globalThis.fetch = (async () => ({ ok: true, json: async () => ({ nope: 1 }) })) as any;
    t.after(() => { globalThis.fetch = original; });
    await assert.rejects(() => fetchModels("https://x.example/v1", null), /Unexpected model catalog payload/);
  });
});
