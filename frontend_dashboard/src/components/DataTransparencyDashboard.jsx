import { useState, useEffect } from 'react';
import { getUserInfo, getTransparencyStats, getTrainingLog } from '../api';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export default function DataTransparencyDashboard() {
  const [userId, setUserId] = useState(0);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [trainingLog, setTrainingLog] = useState([]);

  const fetchInfo = async () => {
    setLoading(true);
    setInfo(null);
    try {
      const { data } = await getUserInfo(userId);
      setInfo(data);
    } catch (err) {
      setInfo({ error: true, message: err.response?.data?.detail || err.message });
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    setStatsLoading(true);
    try {
      const { data } = await getTransparencyStats();
      setStats(data);
    } catch {
      setStats(null);
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchTrainingLog = async () => {
    try {
      const { data } = await getTrainingLog();
      setTrainingLog(data || []);
    } catch {
      setTrainingLog([]);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchTrainingLog();
  }, []);

  // Chart data from contribution_history
  const chartData = (info?.contribution_history || []).map((w, idx) => ({
    round: idx + 1,
    weight: +(w * 100).toFixed(2),
  }));

  // Training progress chart from training log
  const progressData = trainingLog.map((entry) => ({
    round: entry.round_number,
    avg_loss: entry.avg_loss,
    avg_accuracy: +(entry.avg_accuracy * 100).toFixed(2),
    participants: (entry.participating_clients || []).length,
    excluded: (entry.excluded_clients || []).length,
  }));

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Data Transparency Portal</h2>
        <p className="mt-2 text-gray-500 text-sm leading-relaxed">
          View real training logs, per-client participation, and contribution data from
          <strong> PneumoniaMNIST</strong> federated training. All data comes from actual training runs.
        </p>
      </div>

      {/* Global stats banner */}
      {stats && (
        <div className="grid sm:grid-cols-4 gap-3">
          <MiniStat label="Training Rounds" value={stats.total_rounds} />
          <MiniStat label="Participations" value={stats.total_participations} />
          <MiniStat label="Exclusions" value={stats.total_exclusions} />
          <MiniStat label="Unique Clients" value={stats.unique_clients} />
        </div>
      )}

      {/* Training progress chart */}
      {progressData.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Training Progress (Real Data)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={progressData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="round" tick={{ fontSize: 12 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 12 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 12 }} label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} label={{ value: 'Acc %', angle: 90, position: 'insideRight', fontSize: 12 }} />
              <Tooltip contentStyle={{ borderRadius: '0.75rem', fontSize: '0.8rem' }} />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="avg_loss" stroke="#ef4444" strokeWidth={2} name="Avg Loss" dot={{ r: 4 }} />
              <Line yAxisId="right" type="monotone" dataKey="avg_accuracy" stroke="#10b981" strokeWidth={2} name="Avg Accuracy %" dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training log table */}
      {trainingLog.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-700">Training Log (Per Round)</h3>
            <button onClick={fetchTrainingLog}
              className="text-xs rounded-lg bg-gray-100 hover:bg-gray-200 px-3 py-1.5 font-medium text-gray-600 transition-colors">
              🔄 Refresh
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-100 text-left font-semibold text-gray-500 uppercase tracking-wide">
                  <th className="py-2 pr-3">Round</th>
                  <th className="py-2 pr-3">Participants</th>
                  <th className="py-2 pr-3">Excluded</th>
                  <th className="py-2 pr-3">Avg Loss</th>
                  <th className="py-2 pr-3">Avg Acc</th>
                  <th className="py-2 pr-3">Contributions</th>
                  <th className="py-2">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {trainingLog.map((entry) => (
                  <tr key={entry.round_number} className="border-b border-gray-50">
                    <td className="py-2 pr-3 font-medium">{entry.round_number}</td>
                    <td className="py-2 pr-3">{(entry.participating_clients || []).join(', ') || '—'}</td>
                    <td className="py-2 pr-3">{(entry.excluded_clients || []).length > 0
                      ? entry.excluded_clients.join(', ')
                      : <span className="text-gray-300">none</span>}</td>
                    <td className="py-2 pr-3">{entry.avg_loss}</td>
                    <td className="py-2 pr-3 font-semibold text-emerald-600">{(entry.avg_accuracy * 100).toFixed(1)}%</td>
                    <td className="py-2 pr-3">
                      {Object.entries(entry.contribution_weights || {}).map(([cid, w]) => (
                        <span key={cid} className="inline-block mr-2">
                          <span className="font-medium">C{cid}:</span> {(w * 100).toFixed(1)}%
                        </span>
                      ))}
                    </td>
                    <td className="py-2 text-gray-400">{entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {trainingLog.length === 0 && (
        <div className="rounded-2xl border bg-gray-50 border-gray-200 text-gray-500 p-5 text-sm text-center">
          No training data yet. Start a federated training session from the Training Control page.
        </div>
      )}

      {/* Client lookup */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 flex items-end gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-1">Client ID</label>
          <input
            type="number"
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
            className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"
          />
        </div>
        <button onClick={fetchInfo} disabled={loading}
          className="rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-medium px-6 py-2.5 text-sm transition-colors disabled:opacity-50">
          {loading ? 'Loading…' : 'Fetch Client Info'}
        </button>
        <button onClick={() => { fetchStats(); fetchTrainingLog(); }} disabled={statsLoading}
          className="rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium px-4 py-2.5 text-sm transition-colors disabled:opacity-50">
          🔄
        </button>
      </div>

      {/* Error */}
      {info?.error && (
        <div className="rounded-2xl border bg-red-50 border-red-200 text-red-700 p-5 text-sm">❌ {info.message}</div>
      )}

      {/* No records */}
      {info && !info.error && info.message && !info.training_rounds && (
        <div className="rounded-2xl border bg-gray-50 border-gray-200 text-gray-600 p-5 text-sm">ℹ️ {info.message}</div>
      )}

      {/* Client data cards */}
      {info && !info.error && info.training_rounds !== undefined && (
        <>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <InfoCard label="Client ID" value={info.user_id} icon="👤" />
            <InfoCard label="Training Purpose" value={info.purpose || '—'} icon="🎯" />
            <InfoCard label="Training Rounds" value={info.training_rounds} icon="🔄" />
            <InfoCard label="Avg. Contribution" value={`${(info.contribution_weight * 100).toFixed(2)}%`} icon="📈" />
            <InfoCard label="Last Training" value={info.last_training ?? '—'} icon="🕒" />
            <InfoCard label="Data Used" value={(info.data_used || []).join(', ') || '—'} icon="📦" />
            {info.dataset && <InfoCard label="Dataset" value={info.dataset} icon="💾" />}
            {info.algorithm && <InfoCard label="Algorithm" value={info.algorithm} icon="⚡" />}
            {info.exclusions > 0 && <InfoCard label="Times Excluded" value={info.exclusions} icon="🚫" />}
          </div>

          {/* Contribution chart */}
          {chartData.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">
                Client {info.user_id} – Contribution Weight per Round (%)
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="round" tick={{ fontSize: 12 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} label={{ value: '%', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: '0.75rem', fontSize: '0.8rem' }} formatter={(v) => [`${v}%`, 'Weight']} />
                  <Bar dataKey="weight" fill="#338fff" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function InfoCard({ label, value, icon }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-5 flex items-start gap-3">
      <span className="text-2xl">{icon}</span>
      <div className="min-w-0">
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">{label}</p>
        <p className="mt-1 text-sm font-semibold text-gray-800 truncate">{String(value)}</p>
      </div>
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
      <p className="text-2xl font-bold text-brand-600">{value ?? '—'}</p>
      <p className="text-xs text-gray-500 mt-1">{label}</p>
    </div>
  );
}
