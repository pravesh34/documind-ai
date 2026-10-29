/**
 * Per-request AI provider configuration ("bring your own keys").
 *
 * Keys live ONLY in this browser's localStorage and travel with each request
 * in the `x-ai-config` JSON header; the backend never persists them.
 */

export type LLMProviderChoice =
  | 'auto'
  | 'ollama'
  | 'gemini'
  | 'groq'
  | 'openrouter'
  | 'none';

export type EmbeddingProviderChoice =
  | 'auto'
  | 'ollama'
  | 'gemini'
  | 'huggingface'
  | 'deterministic';

export interface AiProviderConfig {
  llmProvider?: LLMProviderChoice;
  embeddingProvider?: EmbeddingProviderChoice;
  geminiApiKey?: string;
  groqApiKey?: string;
  openrouterApiKey?: string;
  hfApiToken?: string;
  ollamaBaseUrl?: string;
  geminiChatModel?: string;
  groqChatModel?: string;
  openrouterChatModel?: string;
  hfEmbeddingModel?: string;
  ollamaChatModel?: string;
  ollamaEmbeddingModel?: string;
}

const AI_CONFIG_STORAGE = 'documind.aiConfig';

export function getAiConfig(): AiProviderConfig | null {
  try {
    const raw = window.localStorage.getItem(AI_CONFIG_STORAGE);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AiProviderConfig;
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

export function setAiConfig(config: AiProviderConfig | null): void {
  try {
    if (config) window.localStorage.setItem(AI_CONFIG_STORAGE, JSON.stringify(config));
    else window.localStorage.removeItem(AI_CONFIG_STORAGE);
  } catch {
    /* storage may be unavailable (private mode) */
  }
}

/** Compact copy with undefined/empty fields removed. */
export function cleanConfig(config: AiProviderConfig): AiProviderConfig {
  const cleaned: AiProviderConfig = {};
  const target = cleaned as Record<string, string>;
  for (const [key, value] of Object.entries(config)) {
    if (typeof value === 'string' && value.trim()) target[key] = value.trim();
  }
  return cleaned;
}

/** Header object to merge into API calls; {} when nothing is configured. */
export function aiConfigHeader(): Record<string, string> {
  const stored = getAiConfig();
  if (!stored) return {};
  const cleaned = cleanConfig(stored);
  if (Object.keys(cleaned).length === 0) return {};
  return { 'x-ai-config': JSON.stringify(cleaned) };
}
