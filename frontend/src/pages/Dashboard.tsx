import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { DashboardStats } from '../types';
import { Activity, MessageSquare, Folder, CheckCircle } from 'lucide-react';

export function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<DashboardStats>('/api/dashboard');
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stats');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
        Error: {error}
      </div>
    );
  }

  const cards = [
    {
      title: 'Total Topics',
      value: stats?.total_topics || 0,
      icon: Folder,
      color: 'from-blue-500 to-blue-600',
    },
    {
      title: 'Total Q&A',
      value: stats?.total_qa || 0,
      icon: MessageSquare,
      color: 'from-cyan-500 to-cyan-600',
    },
    {
      title: 'Chatbot Status',
      value: stats?.chatbot_status || 'unknown',
      icon: Activity,
      color: 'from-green-500 to-green-600',
    },
    {
      title: 'System Status',
      value: 'Operational',
      icon: CheckCircle,
      color: 'from-emerald-500 to-emerald-600',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-slate-900">Dashboard</h2>
        <p className="text-slate-600 mt-1">System overview and statistics</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {cards.map((card, idx) => (
          <div
            key={idx}
            className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-600">{card.title}</p>
                <p className="text-2xl font-bold text-slate-900 mt-2">
                  {typeof card.value === 'number' ? card.value.toLocaleString() : card.value}
                </p>
              </div>
              <div className={`w-12 h-12 bg-gradient-to-br ${card.color} rounded-lg flex items-center justify-center`}>
                <card.icon className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">System Components</h3>
        <div className="space-y-3">
          <StatusRow label="Chatbot Engine" status={stats?.chatbot_status} />
          <StatusRow label="Editor Service" status={stats?.editor_status} />
          <StatusRow label="Generator Service" status={stats?.generator_status} />
        </div>
      </div>
    </div>
  );
}

function StatusRow({ label, status }: { label: string; status?: string }) {
  const isReady = status === 'ready';
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
      <span className="text-slate-700">{label}</span>
      <span
        className={`px-3 py-1 rounded-full text-xs font-medium ${
          isReady ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'
        }`}
      >
        {status || 'unknown'}
      </span>
    </div>
  );
}
