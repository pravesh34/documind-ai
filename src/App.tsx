import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { DocumentManager } from './components/DocumentManager';
import { ChatWorkspace } from './components/ChatWorkspace';
import { BenchmarkLab } from './components/BenchmarkLab';
import { ArchitectureExport } from './components/ArchitectureExport';
import { DocumentItem } from './types';
import { apiFetch } from './lib/api';

interface DocumentListResponse {
  documents: DocumentItem[];
  total: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'documents' | 'chat' | 'benchmark' | 'export'>('chat');
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | undefined>(undefined);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const fetchDocuments = async () => {
    try {
      const data = await apiFetch<DocumentListResponse>('/documents?limit=200');
      setDocuments(data.documents ?? []);
    } catch (err) {
      console.error('Failed to fetch documents', err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleSelectDocForChat = (docId: string) => {
    setSelectedDocId(docId);
    setActiveTab('chat');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenUpload={() => setIsUploadOpen(true)}
      />

      <main className="flex-1">
        {activeTab === 'chat' && (
          <ChatWorkspace
            documents={documents}
            selectedDocumentId={selectedDocId}
            onSelectDocument={setSelectedDocId}
          />
        )}

        {activeTab === 'documents' && (
          <DocumentManager
            documents={documents}
            onRefresh={fetchDocuments}
            onSelectDocumentForChat={handleSelectDocForChat}
            isUploadOpen={isUploadOpen}
            setIsUploadOpen={setIsUploadOpen}
          />
        )}

        {activeTab === 'benchmark' && (
          <BenchmarkLab documents={documents} />
        )}

        {activeTab === 'export' && (
          <ArchitectureExport />
        )}
      </main>
    </div>
  );
}
