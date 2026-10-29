import React, { useState } from 'react';
import { KeyRound, Upload } from 'lucide-react';
import { getApiKey, setApiKey } from '../lib/api';

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
