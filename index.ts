/**
 * @billjr99/pi-openai-compat — pi-coding-agent extension
 *
 * Registers OpenAI-compatible LLM endpoints as first-class pi providers so
 * their models appear in pi's native /model list and Ctrl+L picker.
 *
 * Design:
 *   - Every provider saved in config.json is registered automatically at
 *     startup.  The factory is async so pi waits for registration to complete
 *     before showing the model list — no session_start delay.
 *   - /compat-login  adds a provider (fetches fresh model list, registers).
 *   - /compat-logout removes a provider (unregisters, restores previous model).
 *   - No activeProviders list — presence in config.providers means registered.
 *
 * Config: ~/.config/pi-openai-compat/config.json
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface CachedModel {
  id: string;
  contextWindow?: number;
  maxTokens?: number;
}

interface ProviderConfig {
  displayName: string;
  baseUrl: string;
  apiKey: string | null;
  cachedModels: CachedModel[];
}

interface ExtensionConfig {
  /** Last non-compat model; restored when the last compat provider is removed. */
  previousModel: { provider: string; id: string } | null;
  /** Every key here is a registered provider. Presence = registered. */
  providers: Record<string, ProviderConfig>;
}

interface OpenAIModelsResponse {
  data: Array<{ id: string; context_window?: number; max_tokens?: number }>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider templates
// ─────────────────────────────────────────────────────────────────────────────

const TEMPLATES: Record<string, {
  displayName: string;
  baseUrl: string;
  keyless: boolean;
  /** If set, only models whose id appears in this list are kept after fetching. */
  modelFilter?: string[];
  /** If true, prompt the user to confirm/edit the base URL (like ollama/custom). */
  promptUrl?: boolean;
  /** Used when the provider does not support GET /v1/models (e.g. returns 405). */
  fallbackModels?: string[];
  /** Where to obtain the API key; presence implies the key is required. */
  keyHint?: string;
}> = {
  openrouter: {
    displayName: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    keyless: false,
    keyHint: "openrouter.ai/keys",
  },
  nvidia_nim: {
    displayName: "NVIDIA NIM",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    keyless: false,
    keyHint: "build.nvidia.com",
  },
  nous: {
    displayName: "Nous Research Portal",
    baseUrl: "https://inference-api.nousresearch.com/v1",
    keyless: false,
    keyHint: "nousresearch.com",
  },
  google: {
    displayName: "Google Gemini",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    keyless: false,
    keyHint: "aistudio.google.com/apikey",
  },
  cerebras: {
    displayName: "Cerebras",
    baseUrl: "https://api.cerebras.ai/v1",
    keyless: false,
    keyHint: "cloud.cerebras.ai",
  },
  github_models: {
    displayName: "GitHub Models",
    baseUrl: "https://models.inference.ai.azure.com",
    keyless: false,
    keyHint: "github.com/settings/tokens",
  },
  sambanova: {
    displayName: "SambaNova",
    baseUrl: "https://api.sambanova.ai/v1",
    keyless: false,
    keyHint: "cloud.sambanova.ai",
  },
  mistral: {
    displayName: "Mistral",
    baseUrl: "https://api.mistral.ai/v1",
    keyless: false,
    keyHint: "console.mistral.ai/api-keys",
  },
  groq: {
    displayName: "Groq",
    baseUrl: "https://api.groq.com/openai/v1",
    keyless: false,
    keyHint: "console.groq.com/keys",
  },
  cloudflare_workers: {
    displayName: "Cloudflare Workers AI",
    baseUrl: "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/ai/v1",
    keyless: false,
    promptUrl: true,
    keyHint: "dash.cloudflare.com → My Profile → API Tokens",
    // Cloudflare Workers AI returns 405 for GET /v1/models; use a curated list.
    fallbackModels: [
      "@cf/meta/llama-4-scout-17b-16e-instruct",
      "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      "@cf/meta/llama-3.1-8b-instruct",
      "@cf/qwen/qwen3-30b-a3b-fp8",
      "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    ],
  },
  cloudflare_ai_gateway: {
    displayName: "Cloudflare AI Gateway",
    // YOUR_PROVIDER is the upstream slug (e.g. "workers-ai", "openai"). /v1 is
    // appended so that fetchModels and chat completions hit the correct path.
    baseUrl: "https://gateway.ai.cloudflare.com/v1/YOUR_ACCOUNT_ID/YOUR_GATEWAY_SLUG/YOUR_PROVIDER/v1",
    keyless: false,
    promptUrl: true,
    keyHint: "dash.cloudflare.com → My Profile → API Tokens",
    fallbackModels: [
      "@cf/meta/llama-4-scout-17b-16e-instruct",
      "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      "@cf/meta/llama-3.1-8b-instruct",
      "@cf/qwen/qwen3-30b-a3b-fp8",
      "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    ],
  },
  zhipu: {
    displayName: "Zhipu (Z.ai / BigModel)",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    keyless: false,
    keyHint: "open.bigmodel.cn/usercenter/apikeys",
  },
  zai: {
    displayName: "Z.ai",
    baseUrl: "https://api.z.ai/api/paas/v4",
    keyless: false,
    keyHint: "platform.z.ai",
  },
  cohere: {
    displayName: "Cohere",
    baseUrl: "https://api.cohere.com/compatibility/v1",
    keyless: false,
    keyHint: "dashboard.cohere.com/api-keys",
  },
  deepseek: {
    displayName: "DeepSeek",
    baseUrl: "https://api.deepseek.com/v1",
    keyless: false,
    keyHint: "platform.deepseek.com/api_keys",
  },
  xai: {
    displayName: "xAI (Grok)",
    baseUrl: "https://api.x.ai/v1",
    keyless: false,
    keyHint: "console.x.ai",
  },
  huggingface: {
    displayName: "Hugging Face",
    baseUrl: "https://router.huggingface.co/v1",
    keyless: false,
    keyHint: "huggingface.co/settings/tokens",
    fallbackModels: [
      "meta-llama/Llama-3.3-70B-Instruct",
      "meta-llama/Meta-Llama-3-8B-Instruct",
      "mistralai/Mistral-7B-Instruct-v0.3",
      "Qwen/Qwen2.5-72B-Instruct",
    ],
  },
  moonshot: {
    displayName: "Moonshot (Kimi)",
    baseUrl: "https://api.moonshot.ai/v1",
    keyless: false,
    keyHint: "platform.moonshot.ai/console/api-key",
  },
  minimax: {
    displayName: "MiniMax",
    baseUrl: "https://api.minimax.io/v1",
    keyless: false,
    keyHint: "platform.minimax.io",
  },
  venice: {
    displayName: "Venice AI",
    baseUrl: "https://api.venice.ai/api/v1",
    keyless: false,
    keyHint: "venice.ai/settings/api",
  },
  fireworks: {
    displayName: "Fireworks AI",
    baseUrl: "https://api.fireworks.ai/inference/v1",
    keyless: false,
    keyHint: "fireworks.ai/account/api-keys",
  },
  together: {
    displayName: "Together AI",
    baseUrl: "https://api.together.xyz/v1",
    keyless: false,
    keyHint: "api.together.ai/settings/api-keys",
  },
  ollama: {
    displayName: "Ollama (local, keyless)",
    baseUrl: "http://localhost:11434/v1",
    keyless: true,
    promptUrl: true,
  },
  ollama_cloud: {
    displayName: "Ollama Cloud",
    baseUrl: "https://ollama.com/v1",
    keyless: false,
    keyHint: "ollama.com/settings/api-keys",
  },
  llmproxy: {
    displayName: "llmproxy (local)",
    baseUrl: "http://localhost:8080/v1",
    keyless: true,
    promptUrl: true,
  },
  vercel: {
    displayName: "Vercel AI Gateway",
    baseUrl: "https://ai-gateway.vercel.sh/v1",
    keyless: false,
    keyHint: "vercel.com/account/tokens",
  },
  opencode_zen: {
    displayName: "OpenCode Zen",
    baseUrl: "https://opencode.ai/zen/v1",
    keyless: false,
    keyHint: "opencode.ai",
    // Verify current model IDs via GET /v1/models — the free model list may change.
    modelFilter: ["big-pickle", "deepseek-v4-flash-free", "minimax-m2.5-free", "nemotron-3-super-free"],
  },
  custom: {
    displayName: "Custom Endpoint",
    baseUrl: "",
    keyless: false,
    promptUrl: true,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

const CONFIG_DIR = path.join(os.homedir(), ".config", "pi-openai-compat");
const CONFIG_PATH = path.join(CONFIG_DIR, "config.json");

function loadConfig(): ExtensionConfig {
  const empty: ExtensionConfig = { previousModel: null, providers: {} };
  try {
    if (!fs.existsSync(CONFIG_DIR)) fs.mkdirSync(CONFIG_DIR, { recursive: true });
    if (!fs.existsSync(CONFIG_PATH)) return empty;

    const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8")) as Record<string, unknown>;
    const config: ExtensionConfig = {
      previousModel: (raw.previousModel as any) ?? null,
      providers: (raw.providers as Record<string, ProviderConfig>) ?? {},
    };

    // Migrate: cachedModels missing on older records.
    for (const key of Object.keys(config.providers)) {
      if (!Array.isArray(config.providers[key].cachedModels)) {
        config.providers[key].cachedModels = [];
      }
    }

    return config;
  } catch (e) {
    console.error("[openai-compat:loadConfig]", e);
    return empty;
  }
}

function saveConfig(config: ExtensionConfig): void {
  try {
    if (!fs.existsSync(CONFIG_DIR)) fs.mkdirSync(CONFIG_DIR, { recursive: true });
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), "utf-8");
  } catch (e) {
    console.error("[openai-compat:saveConfig]", e);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Networking
// ─────────────────────────────────────────────────────────────────────────────

function isLocalUrl(url: string): boolean {
  try {
    const { hostname } = new URL(url);
    return (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "::1" ||
      hostname.endsWith(".local") ||
      hostname.endsWith(".localhost")
    );
  } catch {
    return false;
  }
}

async function fetchModels(baseUrl: string, apiKey: string | null): Promise<CachedModel[]> {
  const url = `${baseUrl.replace(/\/+$/, "")}/models`;
  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const resp = await fetch(url, { headers });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status} from ${url}: ${body}`);
  }

  const json = (await resp.json()) as OpenAIModelsResponse;
  if (!Array.isArray(json?.data)) throw new Error(`Unexpected response from ${url}`);

  return json.data
    .map((m) => ({ id: m.id, contextWindow: m.context_window, maxTokens: m.max_tokens }))
    .filter((m) => Boolean(m.id))
    .sort((a, b) => a.id.localeCompare(b.id));
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider registration helpers
// ─────────────────────────────────────────────────────────────────────────────

function buildProviderModels(models: CachedModel[]) {
  return models.map((m) => ({
    id: m.id,
    name: m.id,
    reasoning: false,
    input: ["text"] as string[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: m.contextWindow ?? 128_000,
    maxTokens: m.maxTokens ?? 4_096,
  }));
}

function compatKey(key: string): string {
  return `compat-${key}`;
}

function registerProvider(pi: ExtensionAPI, key: string, p: ProviderConfig): void {
  pi.registerProvider(compatKey(key), {
    name: `compat/${key.replace(/_/g, "-")}`,
    baseUrl: p.baseUrl,
    apiKey: p.apiKey ?? (isLocalUrl(p.baseUrl) ? "local" : ""),
    api: "openai-completions" as const,
    models: buildProviderModels(p.cachedModels),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension — async factory so registration completes before pi shows /model
// ─────────────────────────────────────────────────────────────────────────────

export default async function (pi: ExtensionAPI) {
  let config = loadConfig();

  // Register all saved providers immediately using cached model lists.
  // The factory is async, so pi waits for this to finish before startup
  // continues — providers are visible in /model from the very first render.
  for (const [key, p] of Object.entries(config.providers)) {
    if (p.cachedModels.length > 0) {
      registerProvider(pi, key, p);
    }
  }

  // ── session_start ──────────────────────────────────────────────────────────
  // Reload config in case it changed (e.g. edited by hand) and re-register
  // any providers whose cache was empty at factory time.
  pi.on("session_start", async (_event, ctx) => {
    config = loadConfig();

    const registered: string[] = [];
    const failed: string[] = [];

    for (const [key, p] of Object.entries(config.providers)) {
      if (p.cachedModels.length > 0) {
        // Use cached list — fast, no network call.
        registerProvider(pi, key, p);
        registered.push(p.displayName);
      } else {
        // Cache is empty (e.g. migrated from older config).  Try a live fetch.
        try {
          const models = await fetchModels(p.baseUrl, p.apiKey);
          if (models.length > 0) {
            p.cachedModels = models;
            saveConfig(config);
            registerProvider(pi, key, p);
            registered.push(`${p.displayName} (refreshed)`);
          } else {
            failed.push(p.displayName);
          }
        } catch {
          failed.push(p.displayName);
        }
      }
    }

    if (registered.length > 0) {
      ctx.ui.notify(`OpenAI-compat: ${registered.join(", ")} available in /model`, "info");
    }
    if (failed.length > 0) {
      ctx.ui.notify(
        `OpenAI-compat: could not reach ${failed.join(", ")} — run /compat-login to refresh.`,
        "warning"
      );
    }
  });

  // ── model_select ───────────────────────────────────────────────────────────
  // Track the last non-compat model for logout restoration.
  pi.on("model_select", async (event, _ctx) => {
    const compatProviderKeys = Object.keys(config.providers).map(compatKey);
    if (!compatProviderKeys.includes(event.model.provider)) {
      config.previousModel = { provider: event.model.provider, id: event.model.id };
      saveConfig(config);
    }
  });

  // ── /compat-login ──────────────────────────────────────────────────────────
  pi.registerCommand("compat-login", {
    description: "Fetch models from an OpenAI-compatible endpoint and add it to /model",
    handler: async (args, ctx) => {
      // Step 1 — pick provider template
      const keys = Object.keys(TEMPLATES);
      const labels = keys.map((k) => TEMPLATES[k].displayName);
      const selectedLabel = await ctx.ui.select("Select Provider", labels);
      if (!selectedLabel) { ctx.ui.notify("Login cancelled.", "info"); return; }
      const key = keys[labels.indexOf(selectedLabel)];
      const tpl = TEMPLATES[key];

      // Step 2 — base URL
      let baseUrl = tpl.baseUrl;
      if (key === "cloudflare_workers") {
        const entered = await ctx.ui.input(
          "Account ID",
          "Your Cloudflare Account ID (find it on the Cloudflare dashboard overview page):",
          ""
        );
        if (entered == null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        const accountId = entered.trim();
        if (!accountId) { ctx.ui.notify("Account ID cannot be empty.", "error"); return; }
        baseUrl = tpl.baseUrl.replace("YOUR_ACCOUNT_ID", accountId);
      } else if (key === "cloudflare_ai_gateway") {
        const accountIdInput = await ctx.ui.input(
          "Account ID",
          "Your Cloudflare Account ID (find it on the Cloudflare dashboard overview page):",
          ""
        );
        if (accountIdInput == null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        const accountId = accountIdInput.trim();
        if (!accountId) { ctx.ui.notify("Account ID cannot be empty.", "error"); return; }

        const gatewayInput = await ctx.ui.input(
          "Gateway Name",
          "Your AI Gateway name/slug (find it under AI → AI Gateway in the Cloudflare dashboard):",
          ""
        );
        if (gatewayInput == null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        const gatewaySlug = gatewayInput.trim();
        if (!gatewaySlug) { ctx.ui.notify("Gateway name cannot be empty.", "error"); return; }

        const providerInput = await ctx.ui.input(
          "Provider",
          "Upstream provider slug (e.g. openai, workers-ai, anthropic — must match your gateway config):",
          "openai"
        );
        if (providerInput == null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        const provider = providerInput.trim() || "openai";

        baseUrl = tpl.baseUrl
          .replace("YOUR_ACCOUNT_ID", accountId)
          .replace("YOUR_GATEWAY_SLUG", gatewaySlug)
          .replace("YOUR_PROVIDER", provider);
      } else if (tpl.promptUrl) {
        const defaultUrl = tpl.baseUrl;
        const prompt = isLocalUrl(defaultUrl)
          ? `Base URL — press Enter for default (${defaultUrl}):`
          : "Base URL of your endpoint (e.g. https://api.example.com/v1):";
        const entered = await ctx.ui.input("Base URL", prompt, defaultUrl);
        if (entered == null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        baseUrl = (entered.trim() || defaultUrl).replace(/\/+$/, "");
        if (!baseUrl) { ctx.ui.notify("Base URL cannot be empty.", "error"); return; }
      }

      // Step 3 — API key (skipped for keyless templates and detected local URLs)
      let apiKey: string | null = null;
      if (!tpl.keyless && !isLocalUrl(baseUrl)) {
        const keyPrompt = tpl.keyHint
          ? `Your API key (required) — get it at ${tpl.keyHint}:`
          : "Your API key — leave blank if keyless:";
        const entered = await ctx.ui.input("API Key", keyPrompt, "");
        if (entered == null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        apiKey = entered.trim() || null;
      }

      // Step 4 — fetch fresh model list
      ctx.ui.notify(`Connecting to ${baseUrl} …`, "info");
      let models: CachedModel[];
      try {
        models = await fetchModels(baseUrl, apiKey);
      } catch (err) {
        if (tpl.fallbackModels && tpl.fallbackModels.length > 0) {
          ctx.ui.notify(
            `Could not fetch model list from ${tpl.displayName} (${err}).\nUsing built-in model list instead.`,
            "warning"
          );
          models = tpl.fallbackModels.map((id) => ({ id }));
        } else {
          ctx.ui.notify(`Connection failed — not saved.\n${err}`, "error");
          return;
        }
      }
      if (!models.length) {
        ctx.ui.notify("Connected but no models returned. Check URL and key.", "error");
        return;
      }

      // Apply model filter if the template defines one (keeps only free/known models).
      if (tpl.modelFilter && tpl.modelFilter.length > 0) {
        const filterSet = new Set(tpl.modelFilter);
        const filtered = models.filter((m) => filterSet.has(m.id));
        if (filtered.length > 0) {
          models = filtered;
        } else {
          ctx.ui.notify(
            `None of the expected free models were found — registering all ${models.length} model(s) returned by the provider. The provider may have renamed their models.`,
            "warning"
          );
        }
      }

      // Step 5 — save to config and register with pi
      config.providers[key] = {
        displayName: tpl.displayName,
        baseUrl,
        apiKey,
        cachedModels: models,
      };
      saveConfig(config);
      registerProvider(pi, key, config.providers[key]);

      ctx.ui.notify(
        `${tpl.displayName} registered — ${models.length} model(s) added to /model.`,
        "success"
      );
    },
  });

  // ── /compat-logout ─────────────────────────────────────────────────────────
  pi.registerCommand("compat-logout", {
    description: "Remove an OpenAI-compatible provider from pi's model list",
    handler: async (_args, ctx) => {
      const providerKeys = Object.keys(config.providers);
      if (!providerKeys.length) {
        ctx.ui.notify("No compat providers are registered.", "info");
        return;
      }

      // If only one provider, confirm directly; otherwise ask which to remove.
      let key: string;
      if (providerKeys.length === 1) {
        key = providerKeys[0];
        const name = config.providers[key].displayName;
        const ok = await ctx.ui.confirm("Remove Provider", `Unregister "${name}"?`);
        if (!ok) { ctx.ui.notify("Cancelled.", "info"); return; }
      } else {
        const labels = providerKeys.map((k) => config.providers[k].displayName);
        const chosen = await ctx.ui.select("Remove which provider?", labels);
        if (!chosen) { ctx.ui.notify("Cancelled.", "info"); return; }
        key = providerKeys[labels.indexOf(chosen)];
      }

      const name = config.providers[key].displayName;
      pi.unregisterProvider(compatKey(key));
      delete config.providers[key];
      saveConfig(config);

      // If no compat providers remain, restore the previous model.
      const isLast = Object.keys(config.providers).length === 0;
      if (isLast && config.previousModel) {
        const prev = config.previousModel;
        try {
          await (pi as any).setModel({ provider: prev.provider, id: prev.id });
          ctx.ui.notify(`"${name}" removed. Restored ${prev.provider}/${prev.id}.`, "info");
        } catch {
          ctx.ui.notify(`"${name}" removed. Use /model to pick a model.`, "info");
        }
      } else {
        ctx.ui.notify(
          isLast
            ? `"${name}" removed. Use /model to pick a model.`
            : `"${name}" removed.`,
          "info"
        );
      }
    },
  });
}
