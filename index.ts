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
 *   - /compat-login   adds a provider (fetches fresh model list, registers).
 *   - /compat-refresh re-fetches the model list for a registered provider.
 *   - /compat-logout  removes a provider (unregisters, restores previous model).
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
  // Optional model-discovery overrides for providers whose /models endpoint
  // lives at a non-standard path / shape (e.g. GitHub Models' /catalog/models,
  // Cloudflare Workers AI's /ai/models/search). Persisted on the provider so
  // session_start re-fetches (when cachedModels is empty) use the override
  // rather than the broken default <baseUrl>/models path.
  modelsUrl?: string;
  modelsIdField?: string;
  modelsKeepTask?: string;
  // Set once we've warned the user that this provider's saved baseUrl no longer
  // matches its template and can't be auto-migrated, so the notice is shown
  // only once rather than on every session_start while the config stays stale.
  staleNotified?: boolean;
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

/** Loose shape for a single entry in any /models response. */
type RawModel = {
  id?: string;
  name?: string;
  context_window?: number;
  max_tokens?: number;
  task?: { name?: string };
};

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
  /**
   * Optional model-discovery overrides for providers whose /models endpoint
   * lives at a non-standard path / shape. May contain the same placeholders
   * as baseUrl (YOUR_ACCOUNT_ID, YOUR_GATEWAY_SLUG, YOUR_PROVIDER); they are
   * substituted alongside the baseUrl substitution in /compat-login.
   */
  modelsUrl?: string;
  /** Field on each model entry that carries the upstream id (default "id"). */
  modelsIdField?: string;
  /** Keep only models whose task.name matches this string (case-insensitive). */
  modelsKeepTask?: string;
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
    // The legacy Azure endpoint (models.inference.ai.azure.com) was retired.
    // Chat now lives under /inference, and the catalog under /catalog/models
    // on the same host — different path entirely, hence the modelsUrl override.
    baseUrl: "https://models.github.ai/inference",
    keyless: false,
    keyHint: "github.com/settings/tokens (fine-grained: Models → read)",
    modelsUrl: "https://models.github.ai/catalog/models",
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
    keyHint: "dash.cloudflare.com → My Profile → API Tokens (`Workers AI: Read`)",
    // The OpenAI-compat base /ai/v1 returns 405 for GET /models. The real
    // catalog lives at /ai/models/search, keys ids in "name" (reserving "id"
    // for an internal UUID), and mixes Text Generation with embeddings and
    // image tasks — so we filter by task.name.
    modelsUrl: "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/ai/models/search?per_page=100",
    modelsIdField: "name",
    modelsKeepTask: "Text Generation",
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
    // Common failure modes:
    //   - 401 "Unauthorized": the token lacks `AI Gateway: Run` permission,
    //     or the gateway has Authenticated Gateway enabled (requires a
    //     separate cf-aig-authorization header, not yet supported here).
    //   - 400 "Please configure AI Gateway": the gateway slug doesn't exist
    //     under this account, or the upstream provider isn't configured on it.
    baseUrl: "https://gateway.ai.cloudflare.com/v1/YOUR_ACCOUNT_ID/YOUR_GATEWAY_SLUG/YOUR_PROVIDER/v1",
    keyless: false,
    promptUrl: true,
    keyHint: "dash.cloudflare.com → My Profile → API Tokens (`AI Gateway: Run` + `Workers AI: Read`)",
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
// Discovery-field auto-heal
//
// Providers saved before model-discovery overrides existed (modelsUrl/
// modelsIdField/modelsKeepTask) have none on their config, so /compat-refresh
// and the session_start rehydrate hit the broken default <baseUrl>/models path
// (e.g. Cloudflare Workers AI's 405). When a provider's saved baseUrl still
// matches its template — so we can recover any account-id/slug placeholders —
// we backfill the discovery fields from the template. Providers whose baseUrl
// no longer matches the template (e.g. GitHub Models' retired Azure host) can't
// be healed safely and are reported so the user can re-run /compat-login.
// ─────────────────────────────────────────────────────────────────────────────

/** Placeholders that may appear in a template's baseUrl / modelsUrl. */
const URL_PLACEHOLDERS = ["YOUR_ACCOUNT_ID", "YOUR_GATEWAY_SLUG", "YOUR_PROVIDER"];

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Recover placeholder values by matching a saved URL against a template URL.
 * Returns a map (possibly empty when the template has no placeholders) when the
 * saved URL is consistent with the template, or null when it isn't — which we
 * treat as "this provider can't be healed automatically".
 */
function recoverPlaceholders(templateUrl: string, savedUrl: string): Record<string, string> | null {
  // Tolerate trailing-slash differences the same way the rest of the code does
  // (e.g. baseUrl.replace(/\/+$/, "")), so a saved URL that differs only by a
  // trailing slash still heals instead of being marked stale.
  const tpl = templateUrl.replace(/\/+$/, "");
  const saved = savedUrl.replace(/\/+$/, "");
  const order: string[] = [];
  const placeholderRe = new RegExp(URL_PLACEHOLDERS.join("|"), "g");
  let source = "^";
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = placeholderRe.exec(tpl)) !== null) {
    source += escapeRegex(tpl.slice(lastIndex, m.index)) + "([^/]+)";
    order.push(m[0]);
    lastIndex = m.index + m[0].length;
  }
  source += escapeRegex(tpl.slice(lastIndex)) + "$";

  const match = new RegExp(source).exec(saved);
  if (!match) return null;
  const out: Record<string, string> = {};
  order.forEach((name, i) => { out[name] = match[i + 1]; });
  return out;
}

function applyPlaceholders(url: string, values: Record<string, string>): string {
  let out = url;
  for (const [name, value] of Object.entries(values)) out = out.split(name).join(value);
  return out;
}

/**
 * Backfill missing discovery fields on saved providers from their template.
 * Mutates `config` in place; the caller is responsible for persisting when
 * `healed` is non-empty. `stale` lists the provider *keys* that need a manual
 * re-login (the caller resolves display names and dedupes notifications).
 */
function migrateDiscoveryFields(config: ExtensionConfig): { healed: string[]; stale: string[] } {
  const healed: string[] = [];
  const stale: string[] = [];

  for (const [key, p] of Object.entries(config.providers)) {
    const tpl = TEMPLATES[key];
    if (!tpl?.modelsUrl) continue;   // no matching template, or template needs no overrides
    if (p.modelsUrl) continue;       // already set (fresh login or manual edit) — never clobber

    const values = recoverPlaceholders(tpl.baseUrl, p.baseUrl);
    if (!values) { stale.push(key); continue; }

    p.modelsUrl = applyPlaceholders(tpl.modelsUrl, values);
    p.modelsIdField = tpl.modelsIdField;
    p.modelsKeepTask = tpl.modelsKeepTask;
    healed.push(p.displayName);
  }

  return { healed, stale };
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

/** Optional per-provider overrides controlling how /models is fetched. */
interface FetchOverrides {
  /** Full URL to fetch instead of `<baseUrl>/models`. */
  url?: string;
  /** Field on each entry that holds the upstream id (default `id`). */
  idField?: string;
  /** Keep only entries whose `task.name` matches (case-insensitive). */
  keepTask?: string;
}

async function fetchModels(
  baseUrl: string,
  apiKey: string | null,
  overrides: FetchOverrides = {},
): Promise<CachedModel[]> {
  // Honor a per-provider override (e.g. GitHub Models' /catalog/models lives
  // on a different path than its inference endpoint; Cloudflare Workers AI's
  // catalog is at /ai/models/search). Fall back to <baseUrl>/models otherwise.
  const url = overrides.url ?? `${baseUrl.replace(/\/+$/, "")}/models`;
  const idField = overrides.idField ?? "id";
  const keepTask = overrides.keepTask;

  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const resp = await fetch(url, { headers });
  if (!resp.ok) {
    // The Authorization header is never echoed here, so the error body is
    // safe to surface even though we include the upstream's full response.
    const body = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status} from ${url}: ${body}`);
  }

  // Normalize the various shapes /models can return:
  //   - OpenAI style:                  {"data": [...]}
  //   - Cloudflare / some gateways:    {"result": [...]}
  //   - Together (and a few others):   [...]   (bare JSON array)
  const json = (await resp.json()) as unknown;
  let raw: RawModel[] | undefined;
  if (Array.isArray(json)) {
    raw = json as RawModel[];
  } else if (json && typeof json === "object") {
    const obj = json as { data?: unknown; result?: unknown };
    if (Array.isArray(obj.data)) raw = obj.data as RawModel[];
    else if (Array.isArray(obj.result)) raw = obj.result as RawModel[];
  }
  if (!raw) {
    // `url` may be an override (e.g. /catalog/models, /ai/models/search), so
    // keep the wording generic rather than referring specifically to /models.
    throw new Error(
      `Unexpected model catalog payload shape from ${url} ` +
      `(expected an array or an object with a "data" or "result" array).`
    );
  }

  return raw
    .filter((m) => {
      if (keepTask === undefined) return true;
      const taskName = m.task?.name ?? "";
      return taskName.toLowerCase() === keepTask.toLowerCase();
    })
    .map((m) => {
      // Coerce the id field defensively: some upstreams expose a numeric id,
      // and storing a non-string would break the localeCompare sort below.
      const rawId = (m as Record<string, unknown>)[idField];
      const id =
        typeof rawId === "string" ? rawId :
        typeof rawId === "number" ? String(rawId) :
        "";
      return { id, contextWindow: m.context_window, maxTokens: m.max_tokens };
    })
    .filter((m) => Boolean(m.id))
    .sort((a, b) => a.id.localeCompare(b.id));
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider registration helpers
// ─────────────────────────────────────────────────────────────────────────────

function buildProviderModels(models: CachedModel[]) {
  return models.map((m) => {
    const id = m.id;
    return {
      id,
      name: id,
      reasoning: false,
      input: ["text"] as string[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: m.contextWindow ?? 128_000,
      maxTokens: m.maxTokens ?? 4_096,
    };
  });
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

  // Self-heal older configs: backfill discovery fields (modelsUrl/idField/
  // keepTask) we can derive from the template without prompting. Persist once
  // so /compat-refresh and the session_start rehydrate use the fixed values.
  if (migrateDiscoveryFields(config).healed.length > 0) saveConfig(config);

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

    // Self-heal what we can (silently), and collect providers whose saved
    // baseUrl no longer matches the template — those need a manual re-login.
    const { healed, stale } = migrateDiscoveryFields(config);
    if (healed.length > 0) saveConfig(config);

    const registered: string[] = [];
    const failed: string[] = [];

    for (const [key, p] of Object.entries(config.providers)) {
      if (p.cachedModels.length > 0) {
        // Use cached list — fast, no network call.
        registerProvider(pi, key, p);
        registered.push(p.displayName);
      } else {
        // Cache is empty (e.g. migrated from older config).  Try a live fetch,
        // honoring any per-provider discovery overrides stored on the config.
        try {
          const models = await fetchModels(p.baseUrl, p.apiKey, {
            url: p.modelsUrl,
            idField: p.modelsIdField,
            keepTask: p.modelsKeepTask,
          });
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
    // Warn about stale providers only once: re-notifying on every session_start
    // while the config stays stale would be noise. Persist the flag so the
    // notice survives restarts.
    const toNotify = stale.filter((key) => !config.providers[key]?.staleNotified);
    if (toNotify.length > 0) {
      const names = toNotify.map((key) => config.providers[key].displayName);
      ctx.ui.notify(
        `OpenAI-compat: ${names.join(", ")} ${names.length === 1 ? "has" : "have"} an out-of-date ` +
        `base URL — run /compat-login to update (its endpoint changed and can't be migrated automatically).`,
        "warning"
      );
      for (const key of toNotify) config.providers[key].staleNotified = true;
      saveConfig(config);
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
      // modelsUrl tracks the discovery URL through the same placeholder
      // substitutions baseUrl goes through, so providers that put the model
      // catalog on a different path (GitHub Models, Cloudflare Workers AI)
      // get a fully-resolved URL by the time we call fetchModels.
      let modelsUrl: string | undefined = tpl.modelsUrl;
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
        if (modelsUrl) modelsUrl = modelsUrl.replace("YOUR_ACCOUNT_ID", accountId);
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
        if (modelsUrl) {
          modelsUrl = modelsUrl
            .replace("YOUR_ACCOUNT_ID", accountId)
            .replace("YOUR_GATEWAY_SLUG", gatewaySlug)
            .replace("YOUR_PROVIDER", provider);
        }
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
        models = await fetchModels(baseUrl, apiKey, {
          url: modelsUrl,
          idField: tpl.modelsIdField,
          keepTask: tpl.modelsKeepTask,
        });
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

      // Step 5 — save to config and register with pi.
      // The discovery overrides are persisted so session_start can re-fetch
      // correctly when cachedModels is empty (without them, the rehydrate
      // path would hit <baseUrl>/models and 404 for these providers).
      config.providers[key] = {
        displayName: tpl.displayName,
        baseUrl,
        apiKey,
        cachedModels: models,
        modelsUrl,
        modelsIdField: tpl.modelsIdField,
        modelsKeepTask: tpl.modelsKeepTask,
      };
      saveConfig(config);
      registerProvider(pi, key, config.providers[key]);

      ctx.ui.notify(
        `${tpl.displayName} registered — ${models.length} model(s) added to /model.`,
        "success"
      );
    },
  });

  // ── /compat-refresh ──────────────────────────────────────────────────────────
  // Force a re-fetch of an already-registered provider's model list, reusing
  // the persisted baseUrl/apiKey and discovery overrides — no need to re-enter
  // URLs, keys, or account IDs like /compat-login. Picks up newly added (or
  // dropped) upstream models without restarting the session.
  pi.registerCommand("compat-refresh", {
    description: "Re-fetch the model list for a registered OpenAI-compatible provider",
    handler: async (_args, ctx) => {
      const providerKeys = Object.keys(config.providers);
      if (!providerKeys.length) {
        ctx.ui.notify("No compat providers are registered. Run /compat-login first.", "info");
        return;
      }

      // Choose which provider(s) to refresh. With one provider, refresh it
      // directly; otherwise offer each by name plus an "All providers" option.
      let keys: string[];
      if (providerKeys.length === 1) {
        keys = providerKeys;
      } else {
        const ALL = "All providers";
        // Embed the internal provider key in each label so duplicate
        // displayNames (or a provider literally named "All providers") can't
        // collide with each other or the special "All providers" option.
        const options = providerKeys.map((k) => ({
          key: k,
          label: `${config.providers[k].displayName} [${k}]`,
        }));
        const labels = [ALL, ...options.map((o) => o.label)];
        const chosen = await ctx.ui.select("Refresh which provider?", labels);
        if (!chosen) { ctx.ui.notify("Cancelled.", "info"); return; }
        keys = chosen === ALL ? providerKeys : [options[labels.indexOf(chosen) - 1].key];
      }

      const refreshed: string[] = [];
      const failed: string[] = [];
      for (const key of keys) {
        const p = config.providers[key];
        ctx.ui.notify(`Refreshing ${p.displayName} …`, "info");
        try {
          const models = await fetchModels(p.baseUrl, p.apiKey, {
            url: p.modelsUrl,
            idField: p.modelsIdField,
            keepTask: p.modelsKeepTask,
          });
          if (models.length > 0) {
            // Only overwrite the cache on a successful, non-empty fetch — a
            // flaky refresh must never blank out a working provider's models.
            p.cachedModels = models;
            saveConfig(config);
            registerProvider(pi, key, p);
            refreshed.push(`${p.displayName} (${models.length})`);
          } else {
            failed.push(p.displayName);
          }
        } catch {
          failed.push(p.displayName);
        }
      }

      if (refreshed.length > 0) {
        ctx.ui.notify(`Refreshed: ${refreshed.join(", ")}.`, "success");
      }
      if (failed.length > 0) {
        ctx.ui.notify(
          `Could not refresh ${failed.join(", ")} — kept the existing model list. ` +
          `Run /compat-login if the provider's URL or key changed.`,
          "warning"
        );
      }
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
