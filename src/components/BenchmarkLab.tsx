import React, { useState, useEffect } from 'react';
import {
  Play,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Plus,
  Eye,
  X,
  Target,
  Sparkles,
  ShieldCheck,
  Zap,
  Filter
} from 'lucide-react';
import { BenchmarkResult, BenchmarkRunReport, BenchmarkTestCase, DocumentItem } from '../types';
import { apiFetch } from '../lib/api';

interface DatasetResponse {
  testCases: BenchmarkTestCase[];
}

interface ReportResponse {
  report: BenchmarkRunReport | null;
}

interface BenchmarkLabProps {
  documents: DocumentItem[];
}

export const BenchmarkLab: React.FC<BenchmarkLabProps> = ({ documents }) => {
  const [report, setReport] = useState<BenchmarkRunReport | null>(null);
  const [testCases, setTestCases] = useState<BenchmarkTestCase[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [filterType, setFilterType] = useState<'all' | 'positive' | 'negative'>('all');
  const [selectedResult, setSelectedResult] = useState<BenchmarkResult | null>(null);
  const [showAddCustomModal, setShowAddCustomModal] = useState(false);

  // Custom Test Case Form
  const [customQuestion, setCustomQuestion] = useState('');
  const [customDocTitle, setCustomDocTitle] = useState('');
  const [customExpectedPassage, setCustomExpectedPassage] = useState('');
  const [customPage, setCustomPage] = useState(1);
  const [customIsNegative, setCustomIsNegative] = useState(false);

  useEffect(() => {
    // Load dataset and latest benchmark report
    apiFetch<DatasetResponse>('/eval/dataset')
      .then((data) => setTestCases(data.testCases ?? []))
      .catch(console.error);

    apiFetch<ReportResponse>('/eval/latest')
      .then((data) => {
        if (data.report) setReport(data.report);
      })
      .catch(console.error);
  }, []);

  const handleRunBenchmark = async () => {
    setIsRunning(true);
    try {
      const data = await apiFetch<ReportResponse>('/eval/run', {
        method: 'POST',
        body: { customTests: testCases },
      });
      if (data.report) {
        setReport(data.report);
      }
    } catch (err) {
      console.error('Failed to run benchmark suite', err);
    } finally {
      setIsRunning(false);
    }
  };

  const handleAddCustomTest = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customQuestion.trim()) return;

    const newTest: BenchmarkTestCase = {
      id: `tc-custom-${Date.now()}`,
      documentTitle: customDocTitle || (documents[0]?.title || 'Custom Document'),
      question: customQuestion.trim(),
      expectedPassage: customIsNegative ? 'NONE' : customExpectedPassage.trim(),
      expectedPage: customIsNegative ? 0 : Number(customPage),
      isNegativeTest: customIsNegative,
      description: customIsNegative
        ? 'User custom negative abstention check'
        : 'User custom fact extraction evaluation',
    };

    setTestCases((prev) => [...prev, newTest]);
    setCustomQuestion('');
    setCustomExpectedPassage('');
    setShowAddCustomModal(false);
  };

  const filteredResults = report?.results.filter((r) => {
    if (filterType === 'positive') return !r.isNegativeTest;
    if (filterType === 'negative') return r.isNegativeTest;
    return true;
  }) || [];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            RAG Benchmark Lab & Faithfulness Suite
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical evaluation of Retrieval Recall@k, claim-level Faithfulness, and Epistemic Abstention accuracy.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowAddCustomModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-md transition-colors cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add Test Case</span>
          </button>
          <button
            onClick={handleRunBenchmark}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 rounded-md transition-colors cursor-pointer shadow-sm shadow-indigo-950"
          >
            {isRunning ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Evaluating Benchmark...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run Evaluation Suite ({testCases.length} Tests)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* KPI Cards: Clean 1-elevation, tabular typography */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
        {/* KPI 1: Recall@k */}
        <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Retrieval Recall@4</span>
            <Target className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono tabular-nums">
            {report ? `${report.overallRecallAtK}%` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Ground-truth hit in top 4</p>
        </div>

        {/* KPI 2: Faithfulness */}
        <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Faithfulness Score</span>
            <Sparkles className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 font-mono tabular-nums">
            {report ? `${report.averageFaithfulness}%` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Claims supported by context</p>
        </div>

        {/* KPI 3: Hallucination Rate */}
        <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Hallucination Rate</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-slate-200 font-mono tabular-nums">
            {report ? `${report.hallucinationRate}%` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Unsupported assertions</p>
        </div>

        {/* KPI 4: Negative Abstention */}
        <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Abstention Accuracy</span>
            <ShieldCheck className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-indigo-400 font-mono tabular-nums">
            {report ? `${report.abstentionAccuracy}%` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Zero-evidence refusal rate</p>
        </div>

        {/* KPI 5: Latency */}
        <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Mean Latency</span>
            <Zap className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-slate-200 font-mono tabular-nums">
            {report ? `${report.averageLatencyMs}ms` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Retrieval + generation</p>
        </div>
      </div>

      {/* Filter and Table Bar */}
      <div className="mt-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Interactive Segmented Filter Control */}
        <div className="flex items-center gap-1 p-1 bg-slate-900 border border-slate-800 rounded-lg">
          <button
            onClick={() => setFilterType('all')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filterType === 'all'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            All Test Cases ({testCases.length})
          </button>
          <button
            onClick={() => setFilterType('positive')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filterType === 'positive'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Fact Extraction
          </button>
          <button
            onClick={() => setFilterType('negative')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filterType === 'negative'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Negative Abstention Checks
          </button>
        </div>

        {/* Unboxed Metadata Summary */}
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>
            {report ? `${report.passedCount} of ${report.totalTests} Passed` : 'Ready to Run'}
          </span>
          <span aria-hidden="true">·</span>
          <span>
            {report ? `Run ID: ${report.runId.slice(0, 14)}` : 'Click Run to execute'}
          </span>
        </div>
      </div>

      {/* Benchmark Results Table */}
      <div className="mt-4 border border-slate-800 rounded-lg overflow-hidden bg-slate-900/60">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900 text-slate-400 font-medium border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Evaluation Question</th>
                <th className="py-3 px-4">Test Type</th>
                <th className="py-3 px-4 text-center">Recall Hit</th>
                <th className="py-3 px-4 text-right">Faithfulness</th>
                <th className="py-3 px-4 text-center">Hallucination</th>
                <th className="py-3 px-4 text-right">Latency</th>
                <th className="py-3 px-4 text-center">Status</th>
                <th className="py-3 px-4 text-right">Drilldown</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {filteredResults.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500">
                    {report
                      ? 'No test cases in this filter category.'
                      : 'No benchmark report generated yet. Click "Run Evaluation Suite" above to execute all regression tests.'}
                  </td>
                </tr>
              ) : (
                filteredResults.map((res) => (
                  <tr key={res.id} className="hover:bg-slate-850/60 transition-colors">
                    <td className="py-3 px-4 font-medium text-white max-w-sm truncate">
                      <span>{res.question}</span>
                      <span className="block text-[11px] text-slate-500 font-sans truncate">
                        {res.documentTitle}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400 font-mono text-[11px]">
                      {res.isNegativeTest ? (
                        <span className="text-amber-400">Negative Abstention</span>
                      ) : (
                        <span className="text-slate-300">Fact Extraction</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center">
                      {res.recallHit ? (
                        <span className="text-emerald-400 font-medium">Yes</span>
                      ) : (
                        <span className="text-red-400 font-medium">Miss</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums text-emerald-400 font-semibold">
                      {res.faithfulnessScore}%
                    </td>
                    <td className="py-3 px-4 text-center">
                      {res.hallucinationDetected ? (
                        <span className="text-red-400 font-medium">Detected</span>
                      ) : (
                        <span className="text-slate-500 font-medium">None</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-400">
                      {res.latencyMs}ms
                    </td>
                    <td className="py-3 px-4 text-center">
                      {res.passed ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>PASS</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] text-red-400 font-medium">
                          <XCircle className="w-3.5 h-3.5" />
                          <span>FAIL</span>
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => setSelectedResult(res)}
                        className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors cursor-pointer"
                        title="Inspect Evidence & Generation"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Drill-down Modal */}
      {selectedResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-semibold text-white">
                  Evaluation Audit: {selectedResult.question}
                </h3>
                <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                  <span>{selectedResult.documentTitle}</span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono tabular-nums font-semibold text-emerald-400">
                    Faithfulness: {selectedResult.faithfulnessScore}%
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono tabular-nums">{selectedResult.latencyMs}ms</span>
                </div>
              </div>
              <button
                onClick={() => setSelectedResult(null)}
                className="p-1 text-slate-400 hover:text-white rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
              {/* Generated Answer */}
              <div>
                <span className="font-semibold text-slate-300 block mb-1">
                  Model Generated Answer:
                </span>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded font-sans leading-relaxed text-slate-200">
                  {selectedResult.generatedAnswer}
                </div>
              </div>

              {/* Retrieved Context Chunks */}
              <div>
                <span className="font-semibold text-slate-300 block mb-1">
                  Retrieved Grounding Evidence ({selectedResult.retrievedChunks.length} passages):
                </span>
                {selectedResult.retrievedChunks.length === 0 ? (
                  <div className="p-3 bg-slate-950 border border-slate-800 rounded text-slate-500 italic">
                    Zero chunks retrieved (Expected for negative out-of-domain tests).
                  </div>
                ) : (
                  <div className="space-y-2">
                    {selectedResult.retrievedChunks.map((chunk, i) => (
                      <div
                        key={i}
                        className="p-3 bg-slate-950 border border-slate-800/80 rounded text-[11px] space-y-1"
                      >
                        <div className="flex justify-between text-indigo-400 font-mono">
                          <span>Page {chunk.pageNumber}</span>
                          <span>Cosine Sim {chunk.similarityScore}</span>
                        </div>
                        <p className="text-slate-300">{chunk.excerpt}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="p-3 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setSelectedResult(null)}
                className="px-3 py-1.5 text-xs text-slate-300 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
              >
                Close Audit
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Custom Test Case Modal */}
      {showAddCustomModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg w-full max-w-lg shadow-2xl flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <h3 className="text-sm font-semibold text-white">Add Custom RAG Regression Test</h3>
              <button
                onClick={() => setShowAddCustomModal(false)}
                className="p-1 text-slate-400 hover:text-white rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleAddCustomTest} className="p-5 space-y-4 text-xs">
              <div>
                <label className="block text-slate-300 font-medium mb-1">
                  Test Query / Question <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  placeholder="e.g. What is the disaster recovery RTO?"
                  value={customQuestion}
                  onChange={(e) => setCustomQuestion(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">
                  Target Document
                </label>
                <select
                  value={customDocTitle}
                  onChange={(e) => setCustomDocTitle(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  {documents.map((d) => (
                    <option key={d.id} value={d.title}>
                      {d.title}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2 pt-1">
                <input
                  type="checkbox"
                  id="custom-neg-check"
                  checked={customIsNegative}
                  onChange={(e) => setCustomIsNegative(e.target.checked)}
                  className="accent-indigo-500 rounded"
                />
                <label htmlFor="custom-neg-check" className="text-slate-300 font-medium cursor-pointer">
                  This is a Negative Test (Verify that the system cleanly abstains)
                </label>
              </div>

              {!customIsNegative && (
                <>
                  <div>
                    <label className="block text-slate-300 font-medium mb-1">
                      Expected Ground-Truth Passage Excerpt
                    </label>
                    <textarea
                      rows={3}
                      placeholder="Paste key sentence expected in the document..."
                      value={customExpectedPassage}
                      onChange={(e) => setCustomExpectedPassage(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-[11px]"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-slate-300 font-medium mb-1">
                      Expected Page Number
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={customPage}
                      onChange={(e) => setCustomPage(Number(e.target.value))}
                      className="w-24 bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
                    />
                  </div>
                </>
              )}

              <div className="pt-2 flex justify-end gap-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddCustomModal(false)}
                  className="px-3.5 py-1.5 rounded text-slate-300 bg-slate-800 hover:bg-slate-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 rounded font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-colors"
                >
                  Add Test Case
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
