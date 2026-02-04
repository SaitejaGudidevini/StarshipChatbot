
import { useEffect, useState, useRef } from 'react';

import { apiClient } from '../api/client';

import { Topic, TopicDetail } from '../types';

import { Folder, ChevronRight, Sparkles, Trash2, X, Edit2, Save, XCircle, TreePalm } from 'lucide-react';

import { SemanticTree } from '../components/SemanticTree';

import '../components/SemanticTree.css';



export function Editor() {

  const [topics, setTopics] = useState<Topic[]>([]);

  const [selectedTopic, setSelectedTopic] = useState<TopicDetail | null>(null);

  const [selectedQA, setSelectedQA] = useState<Set<number>>(new Set());

  const [loading, setLoading] = useState(true);

  const [error, setError] = useState<string | null>(null);

  const [editingQA, setEditingQA] = useState<number | null>(null);

  const [editQuestion, setEditQuestion] = useState('');

  const [editAnswer, setEditAnswer] = useState('');

  const qaContentRef = useRef<HTMLDivElement>(null);

  const [treeData, setTreeData] = useState<any>(null);

  const [showTree, setShowTree] = useState(false);





  useEffect(() => {

    loadTopics();

    apiClient.get<any>('/api/tree/data')

      .then(data => setTreeData(data))

      .catch(err => console.error("Failed to load tree data", err));

  }, []);



  const loadTopics = async () => {

    try {

      setLoading(true);

      const data = await apiClient.get<Topic[]>('/api/editor/topics');

      setTopics(data);

    } catch (err) {

      setError(err instanceof Error ? err.message : 'Failed to load topics');

    } finally {

      setLoading(false);

    }

  };



  const loadTopicDetail = async (index: number) => {

    try {

      const data = await apiClient.get<TopicDetail>(`/api/editor/topics/${index}`);

      setSelectedTopic(data);

      setSelectedQA(new Set());

      setShowTree(false);

      // Scroll Q&A content to top when new topic is selected

      setTimeout(() => {

        qaContentRef.current?.scrollTo({ top: 0, behavior: 'instant' });

      }, 0);

    } catch (err) {

      setError(err instanceof Error ? err.message : 'Failed to load topic details');

    }

  };



  const handleSimplify = async () => {

    if (!selectedTopic || !window.confirm('Simplify this topic using AI?')) return;



    try {

      const topicIndex = topics.findIndex(t => t.name === selectedTopic.topic);

      await apiClient.post('/api/editor/simplify', { topic_index: topicIndex });

      alert('Topic simplified successfully!');

      await loadTopicDetail(topicIndex);

      await loadTopics();

    } catch (err) {

      alert('Failed to simplify topic: ' + (err instanceof Error ? err.message : 'Unknown error'));

    }

  };



  const handleDelete = async () => {

    if (!selectedTopic || selectedQA.size === 0) return;

    if (!window.confirm(`Delete ${selectedQA.size} Q&A pairs?`)) return;



    try {

      const topicIndex = topics.findIndex(t => t.name === selectedTopic.topic);

      await apiClient.post('/api/editor/delete', {

        topic_index: topicIndex,

        qa_indices: Array.from(selectedQA),

      });

      alert(`Deleted ${selectedQA.size} Q&A pairs`);

      await loadTopicDetail(topicIndex);

      await loadTopics();

    } catch (err) {

      alert('Failed to delete: ' + (err instanceof Error ? err.message : 'Unknown error'));

    }

  };



  const toggleQA = (index: number) => {

    const newSet = new Set(selectedQA);

    if (newSet.has(index)) {

      newSet.delete(index);

    } else {

      newSet.add(index);

    }

    setSelectedQA(newSet);

  };



  const startEdit = (index: number, question: string, answer: string) => {

    setEditingQA(index);

    setEditQuestion(question);

    setEditAnswer(answer);

  };



  const cancelEdit = () => {

    setEditingQA(null);

    setEditQuestion('');

    setEditAnswer('');

  };



  const saveEdit = async () => {

    if (!selectedTopic || editingQA === null) return;



    try {

      const topicIndex = topics.findIndex(t => t.name === selectedTopic.topic);

      await apiClient.post('/api/editor/edit', {

        topic_index: topicIndex,

        qa_index: editingQA,

        new_question: editQuestion,

        new_answer: editAnswer,

      });

      alert('Q&A pair updated successfully!');

      await loadTopicDetail(topicIndex);

      await loadTopics();

      cancelEdit();

    } catch (err) {

      alert('Failed to edit: ' + (err instanceof Error ? err.message : 'Unknown error'));

    }

  };



  if (loading) {

    return (

      <div className="flex items-center justify-center h-64">

        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>

      </div>

    );

  }



  return (

    <div className="space-y-6">

      <div>

        <h2 className="text-3xl font-bold text-slate-900">Editor</h2>

        <p className="text-slate-600 mt-1">Manage topics and Q&A pairs</p>

      </div>



      {error && (

        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">

          {error}

        </div>

      )}



      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-12rem)]">

        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col overflow-hidden">

          <h3 className="text-lg font-semibold text-slate-900 mb-4 flex-shrink-0">Topics</h3>

          <div className="space-y-2 overflow-y-auto flex-1">

            {topics.map((topic, idx) => (

              <button

                key={idx}

                onClick={() => loadTopicDetail(idx)}

                className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 transition-colors text-left"

              >

                <div className="flex items-center gap-3">

                  <Folder className="w-5 h-5 text-blue-500" />

                  <div>

                    <p className="font-medium text-slate-900">{topic.name}</p>

                    <p className="text-xs text-slate-500">{topic.qa_count} Q&A pairs</p>

                  </div>

                </div>

                <ChevronRight className="w-5 h-5 text-slate-400" />

              </button>

            ))}

          </div>

        </div>



        <div className="lg:col-span-2 overflow-hidden">

          {selectedTopic ? (

            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col overflow-hidden">

              <div className="p-6 border-b border-slate-200 flex-shrink-0">

                <div className="flex items-center justify-between">

                  <div>

                    <h3 className="text-xl font-semibold text-slate-900">{selectedTopic.topic}</h3>

                    <p className="text-sm text-slate-500 mt-1">

                      {selectedTopic.qa_count} Q&A pairs

                      {selectedQA.size > 0 && ` • ${selectedQA.size} selected`}

                    </p>

                    {selectedTopic.url && (
                      <a
                        href={selectedTopic.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-500 hover:text-blue-700 hover:underline mt-1 block truncate max-w-md"
                        title={selectedTopic.url}
                      >
                        {selectedTopic.url}
                      </a>
                    )}

                  </div>

                  <button

                    onClick={() => setSelectedTopic(null)}

                    className="p-2 hover:bg-slate-100 rounded-lg transition-colors"

                  >

                    <X className="w-5 h-5 text-slate-500" />

                  </button>

                </div>

                <div className="flex gap-2 mt-4">

                  <button

                    onClick={handleSimplify}

                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2 text-sm font-medium"

                  >

                    <Sparkles className="w-4 h-4" />

                    Simplify

                  </button>

                  <button

                    onClick={handleDelete}

                    disabled={selectedQA.size === 0}

                    className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 text-sm font-medium"

                  >

                    <Trash2 className="w-4 h-4" />

                    Delete ({selectedQA.size})

                  </button>

                  <button

                    onClick={() => setShowTree(!showTree)}

                    className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-2 text-sm font-medium"

                  >

                    <TreePalm className="w-4 h-4" />

                    {showTree ? 'Hide Tree' : 'Show Tree'}

                  </button>

                </div>

              </div>

              {showTree && treeData ? (

                <SemanticTree treeData={treeData.tree} selectedTopicName={selectedTopic.topic} />

              ) : (

                <div ref={qaContentRef} className="overflow-y-auto flex-1 p-6 space-y-3">

                  {selectedTopic.qa_pairs.map((qa, idx) => (

                    <div

                      key={idx}

                      className={`p-4 rounded-lg border-2 transition-all ${editingQA === idx

                        ? 'border-green-500 bg-green-50'

                        : selectedQA.has(idx)

                          ? 'border-blue-500 bg-blue-50'

                          : 'border-slate-200 bg-white hover:border-slate-300'

                        }`}

                    >

                      {editingQA === idx ? (

                        // Edit Mode

                        <div className="space-y-3">

                          <div>

                            <label className="block text-xs font-medium text-slate-700 mb-1">Question</label>

                            <input

                              type="text"

                              value={editQuestion}

                              onChange={(e) => setEditQuestion(e.target.value)}

                              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 text-sm"

                            />

                          </div>

                          <div>

                            <label className="block text-xs font-medium text-slate-700 mb-1">Answer</label>

                            <textarea

                              value={editAnswer}

                              onChange={(e) => setEditAnswer(e.target.value)}

                              rows={4}

                              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 text-sm resize-none"

                            />

                          </div>

                          <div className="flex gap-2">

                            <button

                              onClick={saveEdit}

                              className="px-3 py-1.5 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-1 text-sm font-medium"

                            >

                              <Save className="w-3 h-3" />

                              Save

                            </button>

                            <button

                              onClick={cancelEdit}

                              className="px-3 py-1.5 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition-colors flex items-center gap-1 text-sm font-medium"

                            >

                              <XCircle className="w-3 h-3" />

                              Cancel

                            </button>

                          </div>

                        </div>

                      ) : (

                        // View Mode

                        <div className="flex items-start gap-3">

                          <input

                            type="checkbox"

                            checked={selectedQA.has(idx)}

                            onChange={() => toggleQA(idx)}

                            className="mt-1 w-4 h-4 text-blue-500 rounded focus:ring-blue-500"

                          />

                          <div className="flex-1">

                            <p className="font-medium text-slate-900">{qa.question}</p>

                            <p className="text-sm text-slate-600 mt-2">{qa.answer}</p>

                            {qa.is_bucketed && (

                              <span className="inline-block mt-2 px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">

                                Bucket: {qa.bucket_id}

                              </span>

                            )}

                          </div>

                          <button

                            onClick={() => startEdit(idx, qa.question, qa.answer)}

                            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"

                            title="Edit"

                          >

                            <Edit2 className="w-4 h-4 text-slate-500" />

                          </button>

                        </div>

                      )}

                    </div>

                  ))}

                </div>

              )}

            </div>

          ) : (

            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col items-center justify-center text-center">

              <Folder className="w-16 h-16 text-slate-300 mb-4" />

              <p className="text-slate-600">Select a topic to view Q&A pairs</p>

            </div>

          )}

        </div>

      </div>

    </div>

  );

}



