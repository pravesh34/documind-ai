import React, { useState } from 'react';
import { KeyRound, Sparkles, Upload } from 'lucide-react';
import { getApiKey, setApiKey } from '../lib/api';
import {
  getAiConfig,
  setAiConfig,
  type AiProviderConfig,
} from '../lib/aiConfig';

const PROVIDER_LABELS: Record<string, string> = {
  auto: 'Auto (use any free provider)',
  none: 'None (deterministic, no key)',
  gemini: 'Google Gemini',
  groq: 'Groq',
  openrouter: 'OpenRouter',
  ollama: 'Ollama (local)',
};

const EMBEDDING_LABELS: Record<string, string> = {
  auto: 'Auto (use any free provider)',
  deterministic: 'Deterministic (no key)',
  gemini: 'Google Gemini',
  huggingface: 'Hugging Face',
  ollama: 'Ollama (local)',
};

interface FieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

const Field: React.FC<FieldProps> = ({ id, label, value, onChange, placeholder }) => (
  <label className="block" htmlFor={id}>
    <span className="block text-[10px] font-medium uppercase tracking-wide text-slate-500">{label}</span>
    <input
      id={id}
      type="text"
      autoComplete="off"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      className="mt-0.5 w-full px-2 py-1.5 text-xs rounded border border-slate-700 bg-slate-950 text-slate-100 outline-none focus:border-indigo-500"
    />
  </label>
);

export interface AiProvidersProps {
  open: boolean;
  onClose: () => void;
}

export const AiProvidersModal: React.FC<AiProvidersProps> = ({ open, onClose }) => {
  const [draft, setDraft] = useState<AiProviderConfig>(() => getAiConfig() ?? {});
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!open) return null;

  const update = (patch: Partial<AiProviderConfig>) =>
    setDraft((current) => ({ ...current, ...patch }));

  const save = () => {
    const cleaned: AiProviderConfig = {};
    for (const [key, value] of Object.entries(draft)) {
      if (typeof value === 'string' && value.trim()) {
        (cleaned as Record<string, string>)[key] = value.trim();
      }
    }
    setAiConfig(Object.keys(cleaned).length ? cleaned : null);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="w-full max-w-md max-h-[85vh] overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 p-4 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-semibold text-white">AI Providers</h2>
            <p className="text-[11px] text-slate-400">Bring your own free-tier keys. Stored in this browser only — never saved on the server.</p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-200 text-lg leading-none cursor-pointer" aria-label="Close">×</button>
        </div>

        <div className="space-y-3">
          <label className="block" htmlFor="llm-provider">
            <span className="block text-[10px] font-medium uppercase tracking-wide text-slate-500">Chat / reasoning provider</span>
            <select
              id="llm-provider"
              value={draft.llmProvider ?? 'auto'}
              onChange={(event) => update({ llmProvider: event.target.value as AiProviderConfig['llmProvider'] })}
              className="mt-0.5 w-full px-2 py-1.5 text-xs rounded border border-slate-700 bg-slate-950 text-slate-100 outline-none focus:border-indigo-500"
            >
              {Object.entries(PROVIDER_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>

          <label className="block" htmlFor="embedding-provider">
            <span className="block text-[10px] font-medium uppercase tracking-wide text-slate-500">Embedding provider</span>
            <select
              id="embedding-provider"
              value={draft.embeddingProvider ?? 'auto'}
              onChange={(event) => update({ embeddingProvider: event.target.value as AiProviderConfig['embeddingProvider'] })}
              className="mt-0.5 w-full px-2 py-1.5 text-xs rounded border border-slate-700 bg-slate-950 text-slate-100 outline-none focus:border-indigo-500"
            >
              {Object.entries(EMBEDDING_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>

          <Field
            id="gemini-key"
            label="Gemini API key (free: aistudio.google.com/apikey)"
            value={draft.geminiApiKey ?? ''}
            onChange={(value) => update({ geminiApiKey: value })}
            placeholder="AIza..."
          />
          <Field
            id="groq-key"
            label="Groq API key (free: console.groq.com/keys)"
            value={draft.groqApiKey ?? ''}
            onChange={(value) => update({ groqApiKey: value })}
            placeholder="gsk_..."
          />
          <Field
            id="openrouter-key"
            label="OpenRouter API key (free models available)"
            value={draft.openrouterApiKey ?? ''}
            onChange={(value) => update({ openrouterApiKey: value })}
            placeholder="sk-or-v1-..."
          />
          <Field
            id="hf-token"
            label="Hugging Face token (hf.co/settings/tokens)"
            value={draft.hfApiToken ?? ''}
            onChange={(value) => update({ hfApiToken: value })}
            placeholder="hf_..."
          />
          <Field
            id="ollama-url"
            label="Ollama URL (local, optional)"
            value={draft.ollamaBaseUrl ?? ''}
            onChange={(value) => update({ ollamaBaseUrl: value })}
            placeholder="http://localhost:11434"
          />

          <button
            type="button"
            onClick={() => setShowAdvanced((visible) => !visible)}
            className="text-[11px] text-indigo-300 hover:text-indigo-200 cursor-pointer"
          >
            {showAdvanced ? '▾ Hide custom models' : '▸ Custom models (optional)'}
          </button>
          {showAdvanced && (
            <div className="grid grid-cols-2 gap-2">
              <Field id="gemini-chat-model" label="Gemini chat model" value={draft.geminiChatModel ?? ''} onChange={(v) => update({ geminiChatModel: v })} placeholder="gemini-2.5-flash" />
              <Field id="groq-chat-model" label="Groq chat model" value={draft.groqChatModel ?? ''} onChange={(v) => update({ groqChatModel: v })} placeholder="llama-3.3-70b-versatile" />
              <Field id="openrouter-chat-model" label="OpenRouter model" value={draft.openrouterChatModel ?? ''} onChange={(v) => update({ openrouterChatModel: v })} placeholder="meta-llama/llama-3.3-70b-instruct:free" />
              <Field id="hf-embedding-model" label="HF embedding model" value={draft.hfEmbeddingModel ?? ''} onChange={(v) => update({ hfEmbeddingModel: v })} placeholder="BAAI/bge-small-en-v1.5" />
              <Field id="ollama-chat-model" label="Ollama chat model" value={draft.ollamaChatModel ?? ''} onChange={(v) => update({ ollamaChatModel: v })} placeholder="llama3.2:3b" />
              <Field id="ollama-embedding-model" label="Ollama embedding model" value={draft.ollamaEmbeddingModel ?? ''} onChange={(v) => update({ ollamaEmbeddingModel: v })} placeholder="nomic-embed-text" />
            </div>
          )}
        </div>

        <div className="mt-4 flex items-center justify-between gap-2">
          <span className="text-[10px] text-slate-500">
            Keys apply to chat, retrieval and benchmark runs. If a provider fails, the app falls back to deterministic mode.
          </span>
          <div className="flex items-center gap-2 shrink-0">
            <button onClick={() => { setDraft({}); setAiConfig(null); }} className="px-2 py-1 text-[11px] text-slate-400 hover:text-slate-200 cursor-pointer">Reset</button>
            <button onClick={save} className="px-3 py-1.5 text-[11px] font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded cursor-pointer">Save</button>
          </div>
        </div>
      </div>
    </div>
  );
};

const AiProvidersButton: React.FC = () => {
  const [isModalOpen, setModalOpen] = useState(false);
  const stored = getAiConfig();
  const configured = Boolean(stored && Object.keys(stored).length > 0);

  return (
    <>
      <button
        onClick={() => setModalOpen(true)}
        title="Choose AI providers and bring your own free-tier keys"
        className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors whitespace-nowrap cursor-pointer ${
          configured
            ? 'text-violet-300 border-violet-800/70 bg-violet-950/40 hover:bg-violet-950/70'
            : 'text-slate-300 border-slate-700/80 bg-slate-900 hover:bg-slate-800'
        }`}
      >
        <Sparkles className="w-3.5 h-3.5" />
        <span>AI Providers</span>
      </button>
      <AiProvidersModal open={isModalOpen} onClose={() => setModalOpen(false)} />
    </>
  );
};

interface NavbarProps {
  activeTab: 'documents' | 'chat' | 'benchmark' | 'export';
  setActiveTab: (tab: 'documents' | 'chat' | 'benchmark' | 'export') => void;
  onOpenUpload: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onOpenUpload,
}) => {
  const [isKeyOpen, setIsKeyOpen] = useState(false);
  const [keyDraft, setKeyDraft] = useState<string>(getApiKey() ?? '');
  const hasKey = Boolean(getApiKey());

  const applyKey = (value: string | null) => {
    setApiKey(value);
    setKeyDraft(value ?? '');
    setIsKeyOpen(false);
    // Re-run the initial data fetches with the new credentials.
    window.location.reload();
  };

  const saveKey = () => {
    applyKey(keyDraft.trim() || null);
  };

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-8 px-6 py-3.5">
        {/* Zone 1: Brand Wordmark (Single text element) */}
        <button
          onClick={() => setActiveTab('chat')}
          className="text-lg font-bold tracking-tight text-white whitespace-nowrap shrink-0 hover:text-indigo-400 transition-colors cursor-pointer text-left"
        >
          DocuMind AI
        </button>

        {/* Zone 2: Navigation Links (Single line, whitespace-nowrap) */}
        <nav className="hidden md:flex items-center gap-7 text-sm font-medium">
          <button
            onClick={() => setActiveTab('documents')}
            className={`whitespace-nowrap shrink-0 py-1 transition-colors cursor-pointer border-b-2 ${
              activeTab === 'documents'
                ? 'text-white border-indigo-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Documents
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`whitespace-nowrap shrink-0 py-1 transition-colors cursor-pointer border-b-2 ${
              activeTab === 'chat'
                ? 'text-white border-indigo-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Query & Citations
          </button>
          <button
            onClick={() => setActiveTab('benchmark')}
            className={`whitespace-nowrap shrink-0 py-1 transition-colors cursor-pointer border-b-2 ${
              activeTab === 'benchmark'
                ? 'text-white border-indigo-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Benchmark Lab
          </button>
          <button
            onClick={() => setActiveTab('export')}
            className={`whitespace-nowrap shrink-0 py-1 transition-colors cursor-pointer border-b-2 ${
              activeTab === 'export'
                ? 'text-white border-indigo-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Architecture & Export
          </button>
        </nav>

        {/* Zone 3: Primary Action */}
        <div className="flex items-center gap-3 shrink-0">
          <AiProvidersButton />

          <div className="relative">
            <button
              onClick={() => setIsKeyOpen((open) => !open)}
              title="Backend API access key"
              aria-expanded={isKeyOpen}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors whitespace-nowrap cursor-pointer ${
                hasKey
                  ? 'text-emerald-300 border-emerald-800/70 bg-emerald-950/40 hover:bg-emerald-950/70'
                  : 'text-slate-300 border-slate-700/80 bg-slate-900 hover:bg-slate-800'
              }`}
            >
              <KeyRound className="w-3.5 h-3.5" />
              <span>{hasKey ? 'Key set' : 'API Key'}</span>
            </button>

            {isKeyOpen && (
              <div className="absolute right-0 mt-2 w-64 rounded-md border border-slate-700 bg-slate-900 p-3 shadow-xl z-50">
                <label
                  htmlFor="api-key-input"
                  className="block text-[11px] font-medium text-slate-400 mb-1"
                >
                  Backend API key
                </label>
                <input
                  id="api-key-input"
                  type="password"
                  autoComplete="off"
                  value={keyDraft}
                  onChange={(event) => setKeyDraft(event.target.value)}
                  placeholder="x-api-key value"
                  className="w-full px-2 py-1.5 text-xs rounded border border-slate-700 bg-slate-950 text-slate-100 outline-none focus:border-indigo-500"
                />
                <p className="mt-1.5 text-[10px] leading-snug text-slate-500">
                  Stored in this browser only. Needed when the backend runs with
                  AUTH_MODE=api-key.
                </p>
                <div className="mt-2 flex items-center justify-end gap-2">
                  <button
                    onClick={() => applyKey(null)}
                    className="px-2 py-1 text-[11px] text-slate-400 hover:text-slate-200 cursor-pointer"
                  >
                    Clear
                  </button>
                  <button
                    onClick={saveKey}
                    className="px-2.5 py-1 text-[11px] font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded cursor-pointer"
                  >
                    Save
                  </button>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={onOpenUpload}
            className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 rounded-md transition-colors whitespace-nowrap shrink-0 shadow-sm shadow-indigo-950 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Document</span>
          </button>
        </div>
      </div>
    </header>
  );
};
