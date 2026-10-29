import React, { useState, useEffect } from 'react';
import {
  Download,
  Copy,
  Check,
  FileCode,
  FolderGit2,
  Terminal,
  Cpu,
  Layers,
  Database,
  Container,
  CheckCircle2
} from 'lucide-react';
import { RepoFile } from '../types';
import { apiDownload, apiFetch } from '../lib/api';

interface FilesResponse {
  files: RepoFile[];
}

export const ArchitectureExport: React.FC = () => {
  const [files, setFiles] = useState<RepoFile[]>([]);
  const [activeFileIndex, setActiveFileIndex] = useState(0);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [copied, setCopied] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    apiFetch<FilesResponse>('/export/files')
      .then((data) => setFiles(data.files ?? []))
      .catch(console.error);
  }, []);

  const handleDownloadZip = () => {
    setIsDownloading(true);
    apiDownload('/export/download', 'documind-ai-backend.zip')
      .catch((err) => console.error('Failed to download source export', err))
      .finally(() => setTimeout(() => setIsDownloading(false), 800));
  };

  const handleCopyCode = () => {
    const activeFile = filteredFiles[activeFileIndex] || files[0];
    if (activeFile) {
      navigator.clipboard.writeText(activeFile.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const filteredFiles = files.filter(
    (f) => categoryFilter === 'all' || f.category === categoryFilter
  );

  const activeFile = filteredFiles[activeFileIndex] || files[0];

  const categories = [
    { id: 'all', label: 'All Files', icon: Layers },
    { id: 'fastapi', label: 'FastAPI Backend', icon: Cpu },
    { id: 'rag', label: 'Hybrid RAG Engine', icon: Terminal },
    { id: 'celery', label: 'Celery Workers', icon: Cpu },
    { id: 'database', label: 'Database & Models', icon: Database },
    { id: 'docker', label: 'Docker & Compose', icon: Container },
    { id: 'tests', label: 'Test Suite', icon: CheckCircle2 },
    { id: 'docs', label: 'Architecture Docs', icon: FolderGit2 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header section with download CTA */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            Production Architecture & Python Codebase Export
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Runnable Python 3.13 / FastAPI / LangChain + LangGraph / Celery service on PostgreSQL + pgvector.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleDownloadZip}
            disabled={isDownloading}
            className="flex items-center gap-2 px-4 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-md transition-colors cursor-pointer shadow-sm shadow-indigo-950"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Full Repository (.zip)</span>
          </button>
        </div>
      </div>

      {/* Interview & Architecture Defense Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
        <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-lg">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs mb-1.5">
            <Cpu className="w-4 h-4" />
            <span>Hybrid Dense + Sparse RRF</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Combines 768-dim dense semantic embeddings with BM25 keyword matching via Reciprocal Rank Fusion (k=60). Resolves domain acronyms and precise clauses without loss of semantic recall.
          </p>
        </div>

        <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-lg">
          <div className="flex items-center gap-2 text-sky-400 font-semibold text-xs mb-1.5">
            <FolderGit2 className="w-4 h-4" />
            <span>Graph RAG Entity Expansion</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Every chunk stores salient entities. IDF-weighted co-occurrence edges link chunks that share rare terms such as SOC 2 or pgvector, so multi-hop questions reach evidence spread across pages.
          </p>
        </div>

        <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-lg">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-xs mb-1.5">
            <CheckCircle2 className="w-4 h-4" />
            <span>Epistemic Grounding & Abstention</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            A LangGraph verifier scores claim-level lexical support, triggers one bounded revision pass, and abstains with an auditable message when retrieved evidence is insufficient.
          </p>
        </div>

        <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-lg">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs mb-1.5">
            <Database className="w-4 h-4" />
            <span>Multi-Tenant Isolation</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Row-level tenant scoping is enforced on every relational and vector query, preventing cross-tenant document or embedding leakage during nearest-neighbour search.
          </p>
        </div>
      </div>

      {/* Category Filter Tabs */}
      <div className="mt-8 flex items-center gap-1 overflow-x-auto pb-1">
        {categories.map((cat) => {
          const Icon = cat.icon;
          return (
            <button
              key={cat.id}
              onClick={() => {
                setCategoryFilter(cat.id);
                setActiveFileIndex(0);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-colors cursor-pointer ${
                categoryFilter === cat.id
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{cat.label}</span>
            </button>
          );
        })}
      </div>

      {/* Code Viewer Layout */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-4 border border-slate-800 rounded-lg bg-slate-900/40 overflow-hidden min-h-[550px]">
        {/* Left Column: File Explorer Tree (4 cols) */}
        <div className="lg:col-span-4 border-r border-slate-800/80 p-3 bg-slate-950/60 flex flex-col">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-2 py-1 block">
            Repository Structure ({filteredFiles.length} files)
          </span>
          <div className="space-y-1 mt-2 overflow-y-auto max-h-[500px]">
            {filteredFiles.map((file, idx) => (
              <button
                key={file.path}
                onClick={() => setActiveFileIndex(idx)}
                className={`w-full text-left px-2.5 py-2 rounded text-xs transition-colors cursor-pointer flex items-center gap-2 ${
                  activeFileIndex === idx
                    ? 'bg-indigo-950/70 text-indigo-300 font-medium border border-indigo-800/50'
                    : 'text-slate-300 hover:bg-slate-900 hover:text-white'
                }`}
              >
                <FileCode className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                <span className="truncate font-mono text-[11px]">{file.path}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Right Column: Code Viewer (8 cols) */}
        <div className="lg:col-span-8 flex flex-col bg-slate-950">
          {activeFile ? (
            <>
              {/* File header bar */}
              <div className="p-3 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
                <div>
                  <span className="font-mono text-xs font-semibold text-white block">
                    {activeFile.path}
                  </span>
                  <span className="text-[11px] text-slate-400">
                    {activeFile.description}
                  </span>
                </div>
                <button
                  onClick={handleCopyCode}
                  className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-300 bg-slate-850 hover:bg-slate-800 border border-slate-700/80 rounded transition-colors cursor-pointer"
                >
                  {copied ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy Code</span>
                    </>
                  )}
                </button>
              </div>

              {/* Code text content */}
              <div className="flex-1 p-4 overflow-y-auto max-h-[500px]">
                <pre className="font-mono text-[11.5px] text-slate-300 leading-relaxed whitespace-pre select-text">
                  {activeFile.content}
                </pre>
              </div>
            </>
          ) : (
            <div className="p-8 text-center text-slate-500 text-xs">
              Select a file to inspect its code.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
