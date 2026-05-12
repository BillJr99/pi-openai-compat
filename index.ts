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
  keyHint?: string;
}> = {
  openrouter: {
    displayName: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    keyless: false,
    keyHint: "sk-or-... from openrouter.ai/keys",
  },
  nvidia_nim: {
    displayName: "NVIDIA NIM",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    keyless: false,
    keyHint: "nvapi-... from build.nvidia.com",
  },
  nous: {
    displayName: "Nous Research Portal",
    baseUrl: "https://inference-api.nousresearch.com/v1",
    keyless: false,
    keyHint: "Your Nous Portal API key",
  },
  ollama: {
    displayName: "Ollama (local, keyless)",
    baseUrl: "http://localhost:11434/v1",
    keyless: true,
  },
  custom: {
    displayName: "Custom Endpoint",
    baseUrl: "",
    keyless: false,
    keyHint: "Bearer token or API key (leave blank if keyless)",
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
    name: p.displayName,
    baseUrl: p.baseUrl,
    apiKey: p.apiKey ?? "ollama",
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

      // Step 2 — base URL (prompted for ollama / custom)
      let baseUrl = tpl.baseUrl;
      if (key === "ollama" || key === "custom") {
        const defaultUrl = tpl.baseUrl;
        const entered = await ctx.ui.input(
          "Base URL",
          key === "ollama"
            ? `Ollama base URL — press Enter for default (${defaultUrl}):`
            : "Base URL of your endpoint (e.g. https://api.example.com/v1):",
          defaultUrl
        );
        if (entered === null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        baseUrl = (entered.trim() || defaultUrl).replace(/\/+$/, "");
        if (!baseUrl) { ctx.ui.notify("Base URL cannot be empty.", "error"); return; }
      }

      // Step 3 — API key
      let apiKey: string | null = null;
      if (!tpl.keyless) {
        const entered = await ctx.ui.input(
          "API Key",
          `${tpl.keyHint ?? "Your API key"} — leave blank if keyless:`,
          ""
        );
        if (entered === null) { ctx.ui.notify("Login cancelled.", "info"); return; }
        apiKey = entered.trim() || null;
      }

      // Step 4 — fetch fresh model list
      ctx.ui.notify(`Connecting to ${baseUrl} …`, "info");
      let models: CachedModel[];
      try {
        models = await fetchModels(baseUrl, apiKey);
      } catch (err) {
        ctx.ui.notify(`Connection failed — not saved.\n${err}`, "error");
        return;
      }
      if (!models.length) {
        ctx.ui.notify("Connected but no models returned. Check URL and key.", "error");
        return;
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
