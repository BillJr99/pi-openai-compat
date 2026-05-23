# @billjr99/pi-openai-compat — pi-coding-agent extension

Registers OpenAI-compatible LLM endpoints as first-class providers inside
[pi](https://pi.dev), so their models appear directly in pi's native `/model`
list and `Ctrl+L` picker alongside built-in Anthropic, OpenAI, and Google
models.  No custom model selection UI — pi handles it natively.

Multiple providers can be registered at the same time.  All of their models
appear together in `/model` under their own provider label.

---

## Quick start

```bash
# Install from npm
pi install npm:@billjr99/pi-openai-compat

# Or from GitHub
pi install git:github.com/BillJr99/pi-openai-compat
```

Then inside pi:

```
/compat-login
```

Pick your provider, enter credentials, and pi fetches the model list
automatically.  Your models are immediately available in `/model`.

If pi is already running when you install, type `/reload` first.

---

## Supported providers

| Provider | Default base URL | Auth |
|---|---|---|
| **OpenRouter** | `https://openrouter.ai/api/v1` | `sk-or-...` from openrouter.ai/keys |
| **NVIDIA NIM** | `https://integrate.api.nvidia.com/v1` | `nvapi-...` from build.nvidia.com |
| **Nous Research Portal** | `https://inference-api.nousresearch.com/v1` | Nous Portal API key |
| **DeepSeek** | `https://api.deepseek.com/v1` | API key from platform.deepseek.com |
| **xAI (Grok)** | `https://api.x.ai/v1` | API key from console.x.ai |
| **Hugging Face** | `https://router.huggingface.co/v1` | `hf_...` token from huggingface.co/settings/tokens |
| **Moonshot (Kimi)** | `https://api.moonshot.ai/v1` | `sk-...` key from platform.moonshot.ai |
| **MiniMax** | `https://api.minimax.io/v1` | API key from platform.minimax.io |
| **Z.ai** | `https://api.z.ai/api/paas/v4` | API key from z.ai |
| **Venice AI** | `https://api.venice.ai/api/v1` | API key from venice.ai/settings/api |
| **Cloudflare Workers AI** | `https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1` | API token from dash.cloudflare.com |
| **Cloudflare AI Gateway** | `https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/openai` | API token from dash.cloudflare.com |
| **Vercel AI Gateway** | `https://ai-gateway.vercel.sh/v1` | API key from vercel.com |
| **OpenCode Zen** | `https://opencode.ai/zen/v1` | API key from opencode.ai |
| **Ollama (local)** | `http://localhost:11434/v1` | Keyless |
| **Ollama Cloud** | `https://ollama.com/v1` | Ollama Cloud API key from ollama.com |
| **Custom** | Any URL you supply | Optional bearer token |

> **Providers with built-in fallback model lists (as of May 2026)**
> Some providers do not support `GET /v1/models` or return a non-JSON response.
> For these, the extension substitutes a built-in list and shows a warning during
> `/compat-login`.
>
> | Provider | Reason |
> |---|---|
> | **Cloudflare Workers AI** | Returns HTTP 405 — method not supported |
> | **Cloudflare AI Gateway** | Returns HTTP 401 — no anonymous model enumeration |
> | **Hugging Face** | Returns HTML rather than JSON for `/v1/models` |

---

## Commands

Only two commands are needed.

### `/compat-login`

Walks you through a short wizard:

1. Select a provider from the list above (or choose Custom).
2. For Ollama and Custom, confirm or change the base URL.
3. Enter your API key (skipped for keyless providers like Ollama).
4. The extension connects, fetches the model list from `/v1/models`, and
   registers the provider with pi.

After login, the provider's models appear in pi's `/model` command and
`Ctrl+L` picker immediately.  You can run `/compat-login` again to add a
second provider — all providers are active simultaneously.

### `/compat-logout`

Unregisters a provider from pi.  If you have multiple providers registered,
you are asked which one to remove.  If you have only one, you are asked to
confirm.

When the last compat provider is removed, the extension restores whichever
built-in model you were using before you added any compat providers.

---

## Typical workflow

```
/compat-login
  → pick Ollama
  → press Enter to accept http://localhost:11434/v1
  → 3 models added to /model

/model
  → scroll to ollama/gemma4:latest
  → select it

(start chatting)

/compat-logout
  → confirm remove Ollama
  → previous model restored automatically
```

To add OpenRouter alongside Ollama:

```
/compat-login
  → pick OpenRouter
  → enter API key
  → 300+ models added to /model alongside your Ollama models
```

---

## How it works

The extension uses pi's `registerProvider` API to register each configured
endpoint as a named provider.  This is the same mechanism pi uses internally
for Anthropic, OpenAI, and Google.  Registered providers appear in `/model`
and `Ctrl+L` with their own label, and pi handles all model selection,
routing, and request formatting natively.

On unregistration, pi's built-in `unregisterProvider` restores the original
model list automatically.

Model lists are cached in `config.json` at startup so no network call is
needed to re-register providers when pi restarts.

---

## Config file

Credentials and cached model lists are stored at:

```
~/.config/pi-openai-compat/config.json
```

API keys are stored in plaintext.  Protect the file with `chmod 600` if
needed, or delete it to clear all saved credentials.

---

## Publishing to npmjs.com

### First-time setup

1. Register at https://www.npmjs.com/signup — use `billjr99` as your username
   to match the `@billjr99/` package scope.
2. Verify your email address (required before publishing).
3. Enable two-factor authentication under your npm account settings.
4. Log in from the command line:

```bash
npm login
```

### Publishing

```bash
cd /path/to/pi-openai-compat
npm publish --access public
```

`--access public` is required for scoped packages on the first publish.
Subsequent publishes do not need it.

The package is immediately available at
https://www.npmjs.com/package/@billjr99/pi-openai-compat.

### Releasing a new version

```bash
npm version patch   # 1.0.0 → 1.0.1  (bug fixes)
npm version minor   # 1.0.0 → 1.1.0  (new features)
npm version major   # 1.0.0 → 2.0.0  (breaking changes)
npm publish --access public
```

`npm version` also creates a git tag, so GitHub gets release tags automatically.

### Automated publishing via GitHub Actions

Every push to `main` automatically bumps the patch version and publishes to npm
via `.github/workflows/publish.yml`. This workflow uses **npm trusted publishing**
(OIDC) — no long-lived token or secret is needed anywhere.

#### Required one-time setup on npmjs.com

> **Do this before the first automated publish or it will fail.**

1. Go to **npmjs.com** → sign in → click your avatar → **Packages** →
   `@billjr99/pi-openai-compat`
2. Click **Settings** (in the left sidebar of the package page)
3. Scroll to **Publishing** → enable **Trusted Publishing**
4. Add a publisher with these exact values:

| Field | Value |
|---|---|
| Repository owner | `BillJr99` |
| Repository name | `pi-openai-compat` |
| Workflow filename | `publish.yml` |
| Environment | *(leave blank)* |

5. Click **Save**

No token to copy, no secret to rotate. That's the only setup step.

#### What the workflow does

```yaml
# .github/workflows/publish.yml
on:
  push:
    branches: [main]

permissions:
  contents: write    # push the version-bump commit back to main
  id-token: write    # request an OIDC token from GitHub for npm auth
```

On every push to `main` the workflow:

1. Checks out the repo with `GITHUB_TOKEN` so it can push back
2. Runs `npm version patch --no-git-tag-version` to increment the patch number
   in `package.json` (e.g. `1.1.0` → `1.1.1`)
3. Commits the updated `package.json` with `[skip ci]` in the message so the
   commit does not re-trigger the workflow
4. Pushes the bump commit to `main`
5. Runs `npm publish --access public --provenance`
   - GitHub mints a short-lived OIDC token proving the job is running from
     this exact repo and workflow
   - npm validates it against the trusted publisher config above and allows
     the publish — no `NPM_TOKEN` secret required
   - `--provenance` attaches a signed build attestation to the package,
     visible on the npmjs.com package page under **Provenance**

The published package is immediately available at
https://www.npmjs.com/package/@billjr99/pi-openai-compat.

### Installing from npm

```bash
pi install npm:@billjr99/pi-openai-compat         # latest
pi install npm:@billjr99/pi-openai-compat@1.0.0   # pinned version
pi update  npm:@billjr99/pi-openai-compat          # update to latest
pi remove  npm:@billjr99/pi-openai-compat          # uninstall
```

---

## Troubleshooting

**Connection fails during `/compat-login`**
Verify the base URL does not have a trailing slash and ends with `/v1`.
Confirm the API key is correct and has model-access permissions.
For Ollama: ensure `ollama serve` is running.

**No models appear after login**
For Ollama: pull at least one model first (`ollama pull llama3`).
For OpenRouter: some keys are restricted to free-tier models only.
For NIM: confirm your account has inference access enabled.

**Models appear in `/model` but requests fail**
Check `/compat-login` ran successfully (no error message).
Verify Ollama is still running if using a local endpoint.
