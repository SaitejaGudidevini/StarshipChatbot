import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, ChevronDown, ChevronUp, Activity } from 'lucide-react';
import { useChat } from '../context/ChatContext';

export function Chat() {
  // Chat state from context (persists across page navigation)
  const { messages, loading, sendMessage } = useChat();

  // Local UI state
  const [input, setInput] = useState('');
  const [expandedPipeline, setExpandedPipeline] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const text = input;
    setInput(''); // Clear input immediately
    await sendMessage(text);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-slate-900">Chat</h2>
        <p className="text-slate-600 mt-1">Ask questions about your Q&A database</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="h-[500px] overflow-y-auto p-6 space-y-4 bg-slate-50">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.type === 'user'
                  ? 'bg-gradient-to-br from-blue-500 to-cyan-500'
                  : 'bg-gradient-to-br from-slate-600 to-slate-700'
                  }`}
              >
                {msg.type === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>
              <div className={`max-w-2xl ${msg.type === 'user' ? '' : 'w-full'}`}>
                <div
                  className={`px-4 py-3 rounded-2xl ${msg.type === 'user'
                    ? 'bg-gradient-to-br from-blue-500 to-cyan-500 text-white'
                    : 'bg-white border border-slate-200 text-slate-900'
                    }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                  {msg.confidence !== undefined && (
                    <div className="mt-2 pt-2 border-t border-slate-200 text-xs text-slate-600">
                      <span>Confidence: {(msg.confidence * 100).toFixed(1)}%</span>
                      {msg.topic && <span className="ml-2">• Topic: {msg.topic}</span>}
                      {msg.matched_by && <span className="ml-2">• Stage: {msg.matched_by}</span>}
                      {msg.url && (
                        <a
                          href={msg.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-2 text-blue-500 hover:text-blue-700 hover:underline"
                        >
                          • Source
                        </a>
                      )}
                    </div>
                  )}
                </div>

                {/* Pipeline Info Dropdown */}
                {msg.type === 'bot' && msg.pipeline_info && (
                  <div className="mt-2">
                    <button
                      onClick={() => setExpandedPipeline(expandedPipeline === msg.id ? null : msg.id)}
                      className="flex items-center gap-2 text-xs text-slate-600 hover:text-slate-900 transition-colors px-2 py-1 rounded hover:bg-slate-100"
                    >
                      <Activity className="w-3 h-3" />
                      <span>View Search Pipeline</span>
                      {expandedPipeline === msg.id ? (
                        <ChevronUp className="w-3 h-3" />
                      ) : (
                        <ChevronDown className="w-3 h-3" />
                      )}
                    </button>

                    {expandedPipeline === msg.id && (
                      <div className="mt-2 p-4 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-3">
                        <div className="font-bold text-slate-800 text-sm mb-3 pb-2 border-b border-slate-300">
                          🔍 Search Pipeline Details - {msg.pipeline_info.architecture === 'v2_parallel_fused' ? 'V2 Advanced Search' : 'V1 Sequential Search'}
                        </div>

                        {/* V2 Architecture Display */}
                        {msg.pipeline_info.architecture === 'v2_parallel_fused' && msg.pipeline_info.retrieval_details ? (
                          <div className="space-y-4">
                            {/* Step 1: Query Analysis */}
                            <div className="p-3 bg-white rounded-lg border border-slate-200">
                              <div className="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                                <span className="text-blue-600">📊</span> Step 1: Query Analysis
                              </div>
                              <div className="space-y-1 text-slate-600 ml-6">
                                <div><span className="font-medium">Intent:</span> {msg.pipeline_info.query_analysis?.intent || 'N/A'}</div>
                                <div><span className="font-medium">Entities:</span> {Object.keys(msg.pipeline_info.query_analysis?.entities || {}).length > 0 ? JSON.stringify(msg.pipeline_info.query_analysis.entities) : 'None detected'}</div>
                                {msg.pipeline_info.query_analysis?.semantic_query && (
                                  <div><span className="font-medium">Optimized:</span> "{msg.pipeline_info.query_analysis.semantic_query}"</div>
                                )}
                              </div>
                            </div>

                            {/* Step 2: Parallel Search */}
                            <div className="p-3 bg-white rounded-lg border border-slate-200">
                              <div className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                                <span className="text-green-600">🔍</span> Step 2: Parallel Search (4 Methods)
                              </div>

                              {Object.entries(msg.pipeline_info.retrieval_details.retrievers).map(([key, retriever]: [string, any]) => (
                                <div key={key} className="mb-3 last:mb-0 ml-6">
                                  <div className="font-medium text-slate-800 mb-1">
                                    {retriever.name}
                                  </div>
                                  <div className="text-xs text-slate-500 mb-2">
                                    └─ {retriever.description}
                                  </div>
                                  {retriever.results?.slice(0, 3).map((result: any, idx: number) => (
                                    <div key={idx} className="ml-3 mb-1 text-slate-600">
                                      <span className="font-medium">#{result.rank}</span> {result.score_display} {result.question.substring(0, 60)}...
                                      <div className="ml-6 text-slate-500 text-[10px]">Topic: {result.topic}</div>
                                    </div>
                                  ))}
                                </div>
                              ))}
                            </div>

                            {/* Step 3: Combining Results */}
                            <div className="p-3 bg-white rounded-lg border border-slate-200">
                              <div className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                                <span className="text-purple-600">🏆</span> Step 3: Combining Results
                              </div>

                              {msg.pipeline_info.retrieval_details.rrf_fusion?.map((item: any, idx: number) => (
                                <div key={idx} className="mb-3 last:mb-0 ml-6 p-2 bg-slate-50 rounded border border-slate-200">
                                  <div className="font-medium text-slate-800">
                                    {idx === 0 ? '🥇 Winner' : `Runner-up #${idx + 1}`}: {item.question.substring(0, 60)}...
                                  </div>
                                  <div className="mt-1 space-y-0.5 text-slate-600">
                                    <div>├─ Combined Score: <span className="font-semibold text-blue-600">{item.score_display}</span> ({item.total_score.toFixed(4)})</div>
                                    <div>├─ Agreement: {item.calculation_simple}</div>
                                    <div>├─ Consensus: <span className="font-semibold">{item.consensus}</span></div>
                                    <div className="text-[10px] text-slate-500">└─ Calculation: {item.calculation}</div>
                                  </div>
                                </div>
                              ))}
                            </div>

                            {/* Step 4: Verification */}
                            <div className="p-3 bg-white rounded-lg border border-slate-200">
                              <div className="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                                <span className="text-green-600">✅</span> Step 4: AI Verification
                              </div>
                              <div className="space-y-1 text-slate-600 ml-6">
                                <div>Candidates Evaluated: {msg.pipeline_info.candidates_evaluated}</div>
                                <div>Candidates Verified: {msg.pipeline_info.candidates_verified}</div>
                                <div className="text-green-600 font-medium">Status: ✓ Answer verified as factually correct</div>
                                {msg.pipeline_info.duration && (
                                  <div className="text-slate-500 text-[10px] mt-2">Total Duration: {msg.pipeline_info.duration.toFixed(2)}s</div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          /* V1 Architecture Display - Keep existing code */
                          msg.pipeline_info.stages?.map((stage: any, idx: number) => (
                            <div key={idx} className="p-2 bg-white rounded border border-slate-200">
                              <div className="font-medium text-slate-900 mb-1">
                                {stage.stage_name || `Stage ${stage.stage_number || idx + 1}`}
                              </div>

                              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-slate-600">
                                {stage.semantic_score !== undefined ? (
                                  <>
                                    <div>Semantic: <span className="font-mono text-blue-600">{stage.semantic_score.toFixed(4)}</span></div>
                                    <div>Reranker: <span className="font-mono text-purple-600">{stage.score.toFixed(4)}</span></div>
                                  </>
                                ) : (
                                  <div>Score: <span className="font-mono">{stage.score.toFixed(4)}</span></div>
                                )}
                                <div>Duration: <span className="font-mono">{stage.duration?.toFixed(3)}s</span></div>

                                {stage.best_match_question && (
                                  <div className="col-span-2 mt-1">
                                    Matched: <span className="italic text-slate-700">"{stage.best_match_question.substring(0, 60)}..."</span>
                                  </div>
                                )}

                                {stage.meets_ideal !== undefined && (
                                  <div className="col-span-2 flex gap-2 mt-1">
                                    <span className={stage.meets_ideal ? 'text-green-600' : 'text-amber-600'}>
                                      {stage.meets_ideal ? '✓ Ideal' : '✗ Below Ideal'}
                                    </span>
                                    <span className={stage.meets_minimal ? 'text-green-600' : 'text-red-600'}>
                                      {stage.meets_minimal ? '✓ Minimal' : '✗ Below Minimal'}
                                    </span>
                                  </div>
                                )}

                                {stage.top_k_candidates && (
                                  <div>Candidates: {stage.top_k_candidates}</div>
                                )}

                                {stage.reranker_used && (
                                  <div className="text-blue-600">🔄 Reranked</div>
                                )}

                                {stage.entity_validation && (
                                  <div className="text-purple-600">👤 Entity Check</div>
                                )}

                                {stage.rephrased_text && (
                                  <div className="col-span-2 mt-1">
                                    Rephrased: <span className="italic text-slate-700">"{stage.rephrased_text}"</span>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-white border-t border-slate-200">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Type your question..."
              className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg font-medium hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
