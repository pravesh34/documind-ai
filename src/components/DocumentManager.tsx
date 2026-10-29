import React, { useState } from 'react';
import {
  FileText,
  Trash2,
  Eye,
  Plus,
  RefreshCw,
  Search,
  CheckCircle2,
  X,
  Layers,
  ArrowRight
} from 'lucide-react';
import { DocumentDetail, DocumentItem } from '../types';
import { apiFetch, describeError } from '../lib/api';

interface DocumentDetailResponse {
  document: DocumentDetail;
}

interface DocumentManagerProps {
  documents: DocumentItem[];
  onRefresh: () => void;
  onSelectDocumentForChat: (docId: string) => void;
  isUploadOpen: boolean;
  setIsUploadOpen: (open: boolean) => void;
}

export const DocumentManager: React.FC<DocumentManagerProps> = ({
  documents,
  onRefresh,
  onSelectDocumentForChat,
  isUploadOpen,
  setIsUploadOpen,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDocDetail, setSelectedDocDetail] = useState<DocumentDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);

  // Upload Form State
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploadText, setUploadText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const filteredDocs = documents.filter(
    (d) =>
      d.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleViewDetails = async (id: string) => {
    setIsLoadingDetail(true);
    try {
      const data = await apiFetch<DocumentDetailResponse>(`/documents/${id}`);
      setSelectedDocDetail(data.document);
    } catch (err) {
      console.error('Failed to load document details', err);
    } finally {
      setIsLoadingDetail(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this document and all associated vector embeddings?')) {
      return;
    }
    setIsDeleting(id);
    try {
      await apiFetch<void>(`/documents/${id}`, { method: 'DELETE' });
      onRefresh();
      if (selectedDocDetail?.id === id) {
        setSelectedDocDetail(null);
      }
    } catch (err) {
      console.error('Failed to delete document', err);
    } finally {
      setIsDeleting(null);
    }
  };

  const handleResetDefaults = async () => {
    try {
      await apiFetch('/documents/reset-defaults', { method: 'POST' });
      onRefresh();
    } catch (err) {
      console.error('Failed to reload default enterprise docs', err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      if (!uploadTitle) {
        setUploadTitle(file.name.replace(/\.[^/.]+$/, ''));
      }
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadTitle.trim()) {
      setUploadError('Document title is required');
      return;
    }
    if (!selectedFile && !uploadText.trim()) {
      setUploadError('Please select a PDF file or enter text content');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      let fileBase64 = '';
      if (selectedFile) {
        const reader = new FileReader();
        fileBase64 = await new Promise<string>((resolve, reject) => {
          reader.onload = () => {
            const result = reader.result as string;
            // Strip data:application/pdf;base64,
            const base64 = result.split(',')[1] || result;
            resolve(base64);
          };
          reader.onerror = reject;
          reader.readAsDataURL(selectedFile);
        });
      }

      const res = await apiFetch<{ document: DocumentDetail }>('/documents/upload', {
        method: 'POST',
        body: {
          title: uploadTitle.trim(),
          filename: selectedFile ? selectedFile.name : `${uploadTitle.trim()}.txt`,
          fileBase64: fileBase64 || undefined,
          textContent: uploadText.trim() || undefined,
          fileSize: selectedFile ? selectedFile.size : uploadText.length,
        },
      });

      if (!res.document) {
        throw new Error('Ingestion failed');
      }

      // Reset and close
      setUploadTitle('');
      setUploadText('');
      setSelectedFile(null);
      setIsUploadOpen(false);
      onRefresh();
    } catch (err) {
      setUploadError(describeError(err, 'Failed to upload and ingest document'));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header section with actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">Enterprise Document Library</h1>
          <p className="text-xs text-slate-400 mt-1">
            Ingested corpora with page segmentation, HNSW vector embeddings, and BM25 index.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleResetDefaults}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-md transition-colors cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reload Enterprise Docs</span>
          </button>
          <button
            onClick={() => setIsUploadOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-md transition-colors cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Ingest New Document</span>
          </button>
        </div>
      </div>

      {/* Search and Summary Bar */}
      <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search document title or filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-md pl-9 pr-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
          />
        </div>

        {/* Unboxed metadata statistics (Zero-pill discipline) */}
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>{documents.length} Ingested Documents</span>
          <span aria-hidden="true">·</span>
          <span>
            {documents.reduce((acc, d) => acc + d.pageCount, 0)} Total Pages
          </span>
          <span aria-hidden="true">·</span>
          <span className="font-mono tabular-nums">
            {documents.reduce((acc, d) => acc + d.chunkCount, 0)} Chunks Indexed
          </span>
        </div>
      </div>

      {/* Documents Table */}
      <div className="mt-5 border border-slate-800 rounded-lg overflow-hidden bg-slate-900/60">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900 text-slate-400 font-medium border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Document Title</th>
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4 text-right">Pages</th>
                <th className="py-3 px-4 text-right">Chunks</th>
                <th className="py-3 px-4 text-right">Size</th>
                <th className="py-3 px-4 text-center">Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {filteredDocs.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-slate-500">
                    No documents found matching your filter criteria.
                  </td>
                </tr>
              ) : (
                filteredDocs.map((doc) => (
                  <tr key={doc.id} className="hover:bg-slate-850/60 transition-colors">
                    <td className="py-3 px-4 font-medium text-white max-w-xs truncate">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
                        <span className="truncate">{doc.title}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-slate-400 font-mono text-[11px] truncate max-w-[180px]">
                      {doc.filename}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums">
                      {doc.pageCount}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-300">
                      {doc.chunkCount}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-400">
                      {(doc.fileSize / 1024).toFixed(1)} KB
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Ready</span>
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => handleViewDetails(doc.id)}
                          title="Inspect Chunks & Text"
                          className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors cursor-pointer"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => onSelectDocumentForChat(doc.id)}
                          title="Ask Questions with this Document"
                          className="flex items-center gap-1 px-2 py-1 text-[11px] text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/60 rounded border border-indigo-800/50 transition-colors cursor-pointer"
                        >
                          <span>Query</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => handleDelete(doc.id)}
                          disabled={isDeleting === doc.id}
                          title="Delete Document"
                          className="p-1 text-slate-500 hover:text-red-400 rounded hover:bg-slate-800 transition-colors cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Document Inspector Modal */}
      {selectedDocDetail && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-semibold text-white">{selectedDocDetail.title}</h3>
                <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                  <span>{selectedDocDetail.filename}</span>
                  <span aria-hidden="true">·</span>
                  <span>{selectedDocDetail.pageCount} Pages</span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono tabular-nums">{selectedDocDetail.chunks.length} Chunks</span>
                </div>
              </div>
              <button
                onClick={() => setSelectedDocDetail(null)}
                className="p-1 text-slate-400 hover:text-white rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-indigo-400" />
                <span>Extracted Pages & Semantic Chunk Envelopes</span>
              </div>

              <div className="space-y-3">
                {selectedDocDetail.chunks.map((chunk, index) => (
                  <div
                    key={chunk.chunkId}
                    className="p-3 bg-slate-950 border border-slate-800/80 rounded-md text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span className="font-mono font-medium text-indigo-400">
                        Chunk #{index + 1} · Page {chunk.pageNumber}
                      </span>
                      <span className="font-mono text-slate-500">
                        {chunk.tokenCount} tokens · chars {chunk.startChar}-{chunk.endChar}
                      </span>
                    </div>
                    <p className="text-slate-300 font-sans leading-relaxed whitespace-pre-wrap">
                      {chunk.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-3 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setSelectedDocDetail(null)}
                className="px-3 py-1.5 text-xs text-slate-300 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {isUploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg w-full max-w-xl shadow-2xl flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <h3 className="text-sm font-semibold text-white">Ingest Document to RAG Store</h3>
              <button
                onClick={() => setIsUploadOpen(false)}
                className="p-1 text-slate-400 hover:text-white rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleUploadSubmit} className="p-5 space-y-4 text-xs">
              {uploadError && (
                <div className="p-2.5 bg-red-950/50 border border-red-800/50 rounded text-red-300 text-xs">
                  {uploadError}
                </div>
              )}

              <div>
                <label className="block text-slate-300 font-medium mb-1">
                  Document Title <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  placeholder="e.g. Q4 Compliance & Audit Policy"
                  value={uploadTitle}
                  onChange={(e) => setUploadTitle(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">
                  Upload PDF File
                </label>
                <div className="border-2 border-dashed border-slate-800 hover:border-slate-700 rounded-lg p-5 text-center bg-slate-950/60">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="hidden"
                    id="pdf-file-input"
                  />
                  <label htmlFor="pdf-file-input" className="cursor-pointer block">
                    <FileText className="w-7 h-7 text-indigo-400 mx-auto mb-2" />
                    <span className="text-indigo-400 font-semibold hover:underline">
                      Click to choose PDF
                    </span>
                    <span className="text-slate-400 block mt-1">or drag and drop up to 25MB</span>
                  </label>
                  {selectedFile && (
                    <div className="mt-3 text-emerald-400 font-mono text-[11px] truncate">
                      Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                    </div>
                  )}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-slate-300 font-medium">
                    Or Paste Text / Markdown Document
                  </label>
                  <span className="text-slate-500 text-[11px]">Optional alternative</span>
                </div>
                <textarea
                  rows={4}
                  placeholder="Paste multi-page text or markdown content here..."
                  value={uploadText}
                  onChange={(e) => setUploadText(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-[11px]"
                />
              </div>

              <div className="pt-2 flex justify-end gap-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsUploadOpen(false)}
                  className="px-3.5 py-1.5 rounded text-slate-300 bg-slate-800 hover:bg-slate-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading}
                  className="px-4 py-1.5 rounded font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-colors flex items-center gap-1.5 disabled:opacity-60"
                >
                  {uploading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Ingesting & Chunking...</span>
                    </>
                  ) : (
                    <span>Ingest Document</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
