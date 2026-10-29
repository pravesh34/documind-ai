import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Sparkles,
  FileText,
  AlertTriangle,
  Sliders,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  HelpCircle,
  Copy,
  Check
} from 'lucide-react';
import { ChatMessage, Citation, DocumentDetail, DocumentItem } from '../types';
import { apiFetch, describeError } from '../lib/api';

interface DocumentDetailResponse {
  document: DocumentDetail;
}

interface ChatWorkspaceProps {
  documents: DocumentItem[];
  selectedDocumentId?: string;
  onSelectDocument: (id: string | undefined) => void;
}

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  documents,
  selectedDocumentId,
  onSelectDocument,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-msg',
      role: 'assistant',
      content:
        'Welcome to **DocuMind AI**. I am your context-grounded document assistant. Ask any question regarding your ingested financial reports, SOC 2 policies, or technical specifications. Every answer includes verifiable page citations.',
      timestamp: new Date().toISOString(),
      groundingScore: 100,
    },
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [activeDocDetail, setActiveDocDetail] = useState<DocumentDetail | null>(null);
  const [activePageNumber, setActivePageNumber] = useState<number>(1);
  const [showSettings, setShowSettings] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // RAG Model Tuning Settings
  const [topK, setTopK] = useState(4);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.25);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Load document pages when active citation changes
  useEffect(() => {
    if (activeCitation) {
      apiFetch<DocumentDetailResponse>(`/documents/${activeCitation.documentId}`)
        .then((data) => {
          setActiveDocDetail(data.document);
          setActivePageNumber(activeCitation.pageNumber);
        })
        .catch((err) => console.error('Failed to load document preview', err));
    }
  }, [activeCitation]);

  // If selectedDocumentId changes, fetch doc detail for initial reader view
  useEffect(() => {
    if (selectedDocumentId) {
      apiFetch<DocumentDetailResponse>(`/documents/${selectedDocumentId}`)
        .then((data) => {
          setActiveDocDetail(data.document);
          setActivePageNumber(1);
        })
        .catch(console.error);
    }
  }, [selectedDocumentId]);

  const handleSend = async (queryText?: string) => {
    const q = (queryText || inputQuery).trim();
    if (!q || isLoading) return;

    setInputQuery('');
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: q,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const history = messages
        .filter((message) => message.role === 'user' || message.role === 'assistant')
        .slice(-6)
        .map((message) => ({ role: message.role, content: message.content.slice(0, 2_000) }));

      const data = await apiFetch<{
        answer: string;
        citations?: Citation[];
        groundingScore?: number;
        abstained?: boolean;
        latencyMs?: number;
        retrievalLatencyMs?: number;
        provider?: string;
        model?: string;
        retrievalAttempts?: number;
        trace?: Array<Record<string, unknown>>;
      }>('/chat', {
        method: 'POST',
        body: {
          query: q,
          topK,
          similarityThreshold,
          targetDocumentId: selectedDocumentId || undefined,
          history,
        },
      });

      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: data.answer,
        timestamp: new Date().toISOString(),
        citations: data.citations || [],
        groundingScore: data.groundingScore,
        abstained: data.abstained,
        latencyMs: data.latencyMs,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Automatically select the first citation to display in the right reader
      if (data.citations && data.citations.length > 0) {
        setActiveCitation(data.citations[0]);
      }
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `**Error:** ${describeError(err, 'Unable to retrieve answer. Please try again.')}`,
        timestamp: new Date().toISOString(),
        abstained: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const sampleQueries = [
    {
      label: 'Q3 EBITDA & Revenue',
      text: 'What was the consolidated net revenue and adjusted EBITDA margin for Q3 2026?',
    },
    {
      label: 'Data at Rest Encryption',
      text: 'What encryption algorithm and key management certification is mandated for data at rest?',
    },
    {
      label: 'Severity 1 SLA',
      text: 'What is the guaranteed triage timeline for a Severity 1 security incident?',
    },
    {
      label: 'Hybrid RRF Formula',
      text: 'What formula and parameters are used for Reciprocal Rank Fusion?',
    },
    {
      label: 'Space Flight Revenue (Negative Test)',
      text: 'What were the revenues generated from the commercial lunar space flight program in Q3?',
    },
  ];

  const currentPage = activeDocDetail?.pages.find((p) => p.pageNumber === activePageNumber);

  return (
    <div className="max-w-7xl mx-auto px-6 py-6 h-[calc(100vh-65px)] flex flex-col">
      {/* Top filter and model parameters bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-400 font-medium">Corpus Scope:</label>
          <select
            value={selectedDocumentId || ''}
            onChange={(e) => onSelectDocument(e.target.value || undefined)}
            className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
          >
            <option value="">All Ingested Documents ({documents.length})</option>
            {documents.map((d) => (
              <option key={d.id} value={d.id}>
                {d.title} ({d.pageCount} pages)
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded border transition-colors cursor-pointer ${
              showSettings
                ? 'bg-indigo-950/70 border-indigo-700 text-indigo-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>RAG Tuning (k={topK})</span>
          </button>
        </div>
      </div>

      {/* Collapsible Tuning Panel */}
      {showSettings && (
        <div className="p-3 bg-slate-900/90 border border-slate-800 rounded-md mt-2 mb-2 grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Retrieval Top-K Passages:</span>
              <span className="font-mono tabular-nums font-semibold">{topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="8"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-indigo-500"
            />
          </div>
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Similarity Confidence Threshold:</span>
              <span className="font-mono tabular-nums font-semibold">{similarityThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
              className="w-full accent-indigo-500"
            />
          </div>
        </div>
      )}

      {/* Main Split-Screen Workspace */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 pt-4 overflow-hidden">
        {/* Left Column: Conversational Chat & Citations (7 cols) */}
        <div className="lg:col-span-7 flex flex-col h-full overflow-hidden border border-slate-800 rounded-lg bg-slate-900/40">
          {/* Messages list */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${
                  msg.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[92%] rounded-lg p-3.5 text-xs ${
                    msg.role === 'user'
                      ? 'bg-indigo-600 text-white font-medium'
                      : 'bg-slate-900 border border-slate-800 text-slate-200'
                  }`}
                >
                  {/* Assistant response header */}
                  {msg.role === 'assistant' && (
                    <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/80 text-[11px] text-slate-400">
                      <div className="flex items-center gap-1.5 font-medium text-indigo-400">
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>DocuMind Grounded Engine</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {typeof msg.groundingScore === 'number' && !msg.abstained && (
                          <span
                            className={`font-mono tabular-nums px-1.5 py-0.5 rounded border ${
                              msg.groundingScore >= 70
                                ? 'text-emerald-300 border-emerald-800/70 bg-emerald-950/40'
                                : msg.groundingScore >= 40
                                  ? 'text-amber-300 border-amber-800/70 bg-amber-950/40'
                                  : 'text-rose-300 border-rose-800/70 bg-rose-950/40'
                            }`}
                            title="Share of answer claims supported by retrieved passages"
                          >
                            {Math.round(msg.groundingScore)}% grounded
                          </span>
                        )}
                        {msg.latencyMs && (
                          <span className="font-mono tabular-nums text-slate-500">
                            {msg.latencyMs}ms
                          </span>
                        )}
                        <button
                          onClick={() => handleCopy(msg.content, msg.id)}
                          className="text-slate-400 hover:text-white"
                          title="Copy Answer"
                        >
                          {copiedId === msg.id ? (
                            <Check className="w-3 h-3 text-emerald-400" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Message body */}
                  <div className="leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </div>

                  {/* Abstention alert */}
                  {msg.abstained && msg.role === 'assistant' && (
                    <div className="mt-3 p-2 bg-amber-950/40 border border-amber-800/50 rounded flex items-start gap-2 text-amber-300 text-[11px]">
                      <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
                      <div>
                        <span className="font-semibold block">Epistemic Abstention Triggered</span>
                        <span>
                          The retrieval engine verified that the ingested corpus lacks evidence for this prompt. No speculative answers were generated.
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Citations List */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-slate-800/80">
                      <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
                        Verifiable Sources ({msg.citations.length}):
                      </span>
                      <div className="space-y-1.5">
                        {msg.citations.map((c, i) => (
                          <button
                            key={c.chunkId || i}
                            onClick={() => setActiveCitation(c)}
                            className={`w-full text-left p-2 rounded border text-[11px] transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                              activeCitation?.chunkId === c.chunkId
                                ? 'bg-indigo-950/80 border-indigo-600 text-indigo-200'
                                : 'bg-slate-950/60 border-slate-800 hover:bg-slate-800 text-slate-300'
                            }`}
                          >
                            <div className="flex items-center gap-2 truncate">
                              <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                              <span className="truncate font-medium">{c.documentTitle}</span>
                              <span className="text-slate-500 font-mono">p. {c.pageNumber}</span>
                            </div>
                            <span className="font-mono text-[10px] text-slate-400 shrink-0">
                              score {c.similarityScore}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex items-center gap-2 text-xs text-slate-400 p-3 bg-slate-900 border border-slate-800 rounded-lg max-w-[280px]">
                <div className="w-3.5 h-3.5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                <span>Retrieving passages & grounding answer...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Question Suggestions */}
          <div className="px-4 py-2 border-t border-slate-800 bg-slate-950/30 overflow-x-auto flex items-center gap-2">
            <span className="text-[11px] text-slate-500 shrink-0 flex items-center gap-1">
              <HelpCircle className="w-3 h-3" />
              <span>Try:</span>
            </span>
            {sampleQueries.map((sq, i) => (
              <button
                key={i}
                onClick={() => handleSend(sq.text)}
                disabled={isLoading}
                className="px-2.5 py-1 text-[11px] font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded whitespace-nowrap transition-colors cursor-pointer disabled:opacity-50"
              >
                {sq.label}
              </button>
            ))}
          </div>

          {/* Input Form */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="p-3 border-t border-slate-800 bg-slate-900/80 flex items-center gap-2"
          >
            <input
              type="text"
              placeholder="Ask anything about the ingested documents..."
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              disabled={isLoading}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-3.5 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !inputQuery.trim()}
              className="px-3.5 py-2.5 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 text-white rounded-md text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-sm shadow-indigo-950"
            >
              <Send className="w-3.5 h-3.5" />
              <span>Ask</span>
            </button>
          </form>
        </div>

        {/* Right Column: Interactive Document & Citation Reader (5 cols) */}
        <div className="lg:col-span-5 flex flex-col h-full overflow-hidden border border-slate-800 rounded-lg bg-slate-900/60">
          <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center justify-between">
            <div className="flex items-center gap-2 truncate">
              <BookOpen className="w-4 h-4 text-indigo-400 shrink-0" />
              <span className="text-xs font-semibold text-white truncate">
                {activeDocDetail ? activeDocDetail.title : 'Document & Citation Reader'}
              </span>
            </div>

            {/* Page navigation controls */}
            {activeDocDetail && (
              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  onClick={() => setActivePageNumber((p) => Math.max(1, p - 1))}
                  disabled={activePageNumber <= 1}
                  className="p-1 text-slate-400 hover:text-white disabled:opacity-40 rounded hover:bg-slate-800 transition-colors cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-xs font-mono tabular-nums text-slate-300">
                  {activePageNumber} / {activeDocDetail.pageCount}
                </span>
                <button
                  onClick={() =>
                    setActivePageNumber((p) =>
                      Math.min(activeDocDetail.pageCount, p + 1)
                    )
                  }
                  disabled={activePageNumber >= activeDocDetail.pageCount}
                  className="p-1 text-slate-400 hover:text-white disabled:opacity-40 rounded hover:bg-slate-800 transition-colors cursor-pointer"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          {/* Reader Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {activeCitation && (
              <div className="p-3 bg-indigo-950/40 border border-indigo-800/60 rounded-md">
                <div className="flex items-center justify-between text-[11px] text-indigo-300 font-medium mb-1">
                  <span>Selected Citation Excerpt:</span>
                  <span className="font-mono text-indigo-400">
                    Page {activeCitation.pageNumber} · Sim {activeCitation.similarityScore}
                  </span>
                </div>
                <blockquote className="text-xs text-indigo-100 font-sans italic border-l-2 border-indigo-500 pl-2.5 py-0.5">
                  "{activeCitation.excerpt}"
                </blockquote>
              </div>
            )}

            {currentPage ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[11px] text-slate-500 pb-1 border-b border-slate-800/80">
                  <span>Page {currentPage.pageNumber} Raw Content</span>
                  <span className="font-mono">{currentPage.charCount} chars</span>
                </div>
                <div className="p-4 bg-slate-950 rounded border border-slate-800/80 font-mono text-[11px] text-slate-300 leading-relaxed whitespace-pre-wrap select-text">
                  {currentPage.text}
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500 text-xs">
                <FileText className="w-8 h-8 text-slate-600 mb-2" />
                <p className="font-medium text-slate-400">No Document Page Loaded</p>
                <p className="mt-1">
                  Ask a question or click any citation above to preview the original source page with grounded excerpt highlights.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
