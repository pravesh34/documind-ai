export interface DocumentItem {
  id: string;
  title: string;
  filename: string;
  fileSize: number;
  pageCount: number;
  chunkCount: number;
  status: 'pending' | 'processing' | 'ready' | 'error';
  uploadedAt: string;
  summary?: string;
}

export interface DocumentPage {
  pageNumber: number;
  text: string;
  charCount: number;
}

export interface DocumentChunk {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  pageNumber: number;
  chunkIndex: number;
  text: string;
  tokenCount: number;
  startChar: number;
  endChar: number;
}

export interface DocumentDetail extends DocumentItem {
  pages: DocumentPage[];
  chunks: DocumentChunk[];
}

export interface Citation {
  documentId: string;
  documentTitle: string;
  pageNumber: number;
  chunkId: string;
  excerpt: string;
  similarityScore: number;
  lexicalScore?: number;
  combinedScore: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: Citation[];
  groundingScore?: number;
  abstained?: boolean;
  latencyMs?: number;
}

export interface BenchmarkTestCase {
  id: string;
  documentTitle: string;
  question: string;
  expectedPassage: string;
  expectedPage: number;
  isNegativeTest: boolean;
  description: string;
}

export interface BenchmarkResult {
  id: string;
  testCaseId: string;
  question: string;
  documentTitle: string;
  isNegativeTest: boolean;
  retrievedChunks: Citation[];
  generatedAnswer: string;
  recallHit: boolean;
  faithfulnessScore: number;
  hallucinationDetected: boolean;
  abstentionCorrect: boolean;
  latencyMs: number;
  passed: boolean;
}

export interface BenchmarkRunReport {
  runId: string;
  timestamp: string;
  totalTests: number;
  passedCount: number;
  overallRecallAtK: number;
  averageFaithfulness: number;
  hallucinationRate: number;
  abstentionAccuracy: number;
  averageLatencyMs: number;
  results: BenchmarkResult[];
}

export interface RepoFile {
  path: string;
  category: 'fastapi' | 'rag' | 'celery' | 'database' | 'docker' | 'tests' | 'docs';
  description: string;
  content: string;
}
