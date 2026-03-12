import { useState, useEffect, useRef } from 'react';
import { startTraining, getTrainingStatus, stopTraining, getTransparencyStats } from '../api';

export default function TrainingControl() {
  // ── Config state (PneumoniaMNIST only) ──────────────────────
  const [config, setConfig] = useState({
    global_rounds: 5,
    local_epochs: 2,
    batch_size: 32,
    learning_rate: 0.001,
    num_clients: 2,
  });

  // ── Job state ───────────────────────────────────────────────
  const [job, setJob] = useState(null);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState(null);
  const [liveStats, setLiveStats] = useState(null);
  const pollRef = useRef(null);
  const logEndRef = useRef(null);

  const updateField = (field, value) =>
    setConfig((prev) => ({ ...prev, [field]: value }));

  // ── Polling ─────────────────────────────────────────────────
  const startPolling = () => {
    stopPolling();
    const tick = async () => {
      try {
        const { data } = await getTrainingStatus();
        setJob(data);
        try {
          const { data: stats } = await getTransparencyStats();
          setLiveStats(stats);
        } catch { /* ignore */ }
        if (!data.running && data.status !== 'waiting' && data.status !== 'aggregating') {
          stopPolling();
        }
      } catch { /* ignore */ }
    };
    tick();
    pollRef.current = setInterval(tick, 2000);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const { data } = await getTrainingStatus();
        setJob(data);
        if (data.running || data.status === 'waiting' || data.status === 'aggregating') {
          startPolling();
        }
      } catch { /* server not up */ }
    })();
    return () => stopPolling();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [job?.output_lines?.length]);

  // ── Actions ─────────────────────────────────────────────────
  const handleStart = async () => {
    setLaunching(true);
    setError(null);
    try {
      const { data } = await startTraining(config);
      setJob({ ...data, running: true, output_lines: [], status: 'waiting' });
      startPolling();
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      setError(msg);
    } finally {
      setLaunching(false);
    }
  };

  const handleStop = async () => {
    try {
      await stopTraining();
      setTimeout(async () => {
        try {
          const { data } = await getTrainingStatus();
          setJob(data);
        } catch { /* ignore */ }
      }, 500);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const isRunning = job?.running || job?.status === 'waiting' || job?.status === 'aggregating';

  const statusColor = {
    idle: 'bg-gray-100 text-gray-600',
    waiting: 'bg-blue-100 text-blue-700 animate-pulse',
    aggregating: 'bg-yellow-100 text-yellow-700 animate-pulse',
    completed: 'bg-emerald-100 text-emerald-700',
    stopped: 'bg-red-100 text-red-700',
  };

  // Build round history chart data
  const roundHistory = job?.round_history || [];

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Federated Training Control</h2>
        <p className="mt-2 text-gray-500 text-sm leading-relaxed">
          Launch a <strong>real federated learning</strong> session on <strong>PneumoniaMNIST</strong>.
          Clients train locally and upload weights. The server aggregates via <strong>FedAvg</strong>.
          All statistics shown are from real training runs — no mock data.
        </p>
      </div>

      {/* Fixed config info */}
      <div className="bg-blue-50 rounded-2xl border border-blue-200 p-4 text-sm text-blue-800">
        <p className="font-semibold mb-1">📋 Fixed Configuration</p>
        <p><strong>Dataset:</strong> PneumoniaMNIST &nbsp;|&nbsp; <strong>Model:</strong> PneumoniaCNN &nbsp;|&nbsp;
           <strong>Algorithm:</strong> FedAvg &nbsp;|&nbsp; <strong>Purpose:</strong> Image Classification</p>
        <p className="text-xs text-blue-600 mt-1">
          Ensure clients are running: <code>python run_client.py --client-id 0</code> and
          <code> python run_client.py --client-id 1</code>
        </p>
      </div>

      {/* Config card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-5">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
          Training Parameters
        </h3>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Field label="Global Rounds">
            <input type="number" min={1} max={50} value={config.global_rounds}
              onChange={(e) => updateField('global_rounds', Number(e.target.value))}
              className="input" />
          </Field>

          <Field label="Expected Clients">
            <input type="number" min={1} max={10} value={config.num_clients}
              onChange={(e) => updateField('num_clients', Number(e.target.value))}
              className="input" />
          </Field>

          <Field label="Local Epochs">
            <input type="number" min={1} max={20} value={config.local_epochs}
              onChange={(e) => updateField('local_epochs', Number(e.target.value))}
              className="input" />
          </Field>

          <Field label="Batch Size">
            <input type="number" min={1} max={256} value={config.batch_size}
              onChange={(e) => updateField('batch_size', Number(e.target.value))}
              className="input" />
          </Field>

          <Field label="Learning Rate">
            <input type="number" step={0.0001} min={0.0001} value={config.learning_rate}
              onChange={(e) => updateField('learning_rate', Number(e.target.value))}
              className="input" />
          </Field>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleStart}
            disabled={launching || isRunning}
            className="flex-1 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-medium py-3 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {launching ? '⏳ Launching…' : isRunning ? '🔄 Training in progress…' : '🚀 Start Federated Training'}
          </button>

          {isRunning && (
            <button
              onClick={handleStop}
              className="rounded-xl bg-red-500 hover:bg-red-600 text-white font-medium px-6 py-3 text-sm transition-colors"
            >
              ⛔ Stop
            </button>
          )}
        </div>

        {error && (
          <div className="rounded-xl bg-red-50 border border-red-200 text-red-700 p-4 text-sm">
            ❌ {error}
          </div>
        )}
      </div>

      {/* Job status */}
      {job && job.status !== 'idle' && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Training Status</h3>
            <span className={`rounded-full px-3 py-1 text-xs font-semibold ${statusColor[job.status] || statusColor.idle}`}>
              {job.status?.toUpperCase()}
            </span>
          </div>

          {/* Meta info */}
          <div className="grid sm:grid-cols-3 gap-3 text-xs text-gray-500">
            {job.started_at && <div><span className="font-semibold text-gray-600">Started:</span> {new Date(job.started_at).toLocaleString()}</div>}
            {job.finished_at && <div><span className="font-semibold text-gray-600">Finished:</span> {new Date(job.finished_at).toLocaleString()}</div>}
            {job.config && <div><span className="font-semibold text-gray-600">Rounds:</span> {job.config.global_rounds}</div>}
            {job.config && <div><span className="font-semibold text-gray-600">Clients:</span> {job.config.num_clients}</div>}
            {job.config && <div><span className="font-semibold text-gray-600">Dataset:</span> {job.config.dataset}</div>}
            {job.config && <div><span className="font-semibold text-gray-600">Model:</span> {job.config.model}</div>}
          </div>

          {/* Live stats */}
          {liveStats && (
            <div className="grid sm:grid-cols-4 gap-3">
              <MiniStat label="Rounds Complete" value={liveStats.total_rounds} />
              <MiniStat label="Participations" value={liveStats.total_participations} />
              <MiniStat label="Exclusions" value={liveStats.total_exclusions} />
              <MiniStat label="Active Clients" value={liveStats.unique_clients} />
            </div>
          )}

          {/* Round history table */}
          {roundHistory.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">Round History</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-100 text-left font-semibold text-gray-500 uppercase tracking-wide">
                      <th className="py-2 pr-3">Round</th>
                      <th className="py-2 pr-3">Participants</th>
                      <th className="py-2 pr-3">Avg Loss</th>
                      <th className="py-2 pr-3">Avg Acc</th>
                      <th className="py-2">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {roundHistory.map((r) => (
                      <tr key={r.round} className="border-b border-gray-50">
                        <td className="py-2 pr-3 font-medium">{r.round}</td>
                        <td className="py-2 pr-3">{(r.participants || []).join(', ')}</td>
                        <td className="py-2 pr-3">{r.avg_loss}</td>
                        <td className="py-2 pr-3 font-semibold text-emerald-600">{(r.avg_accuracy * 100).toFixed(1)}%</td>
                        <td className="py-2 text-gray-400">{r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Log output */}
          {job.output_lines && job.output_lines.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 mb-2">
                Server Log ({job.total_lines ?? job.output_lines.length} lines)
              </p>
              <div className="bg-gray-900 rounded-xl p-4 max-h-72 overflow-y-auto font-mono text-xs text-green-400 leading-relaxed">
                {job.output_lines.map((line, i) => (
                  <div key={i} className={
                    line.includes('[Consent]') || line.includes('[ConsentManager]')
                      ? 'text-yellow-300'
                      : line.includes('[PurposeValidator]')
                      ? 'text-red-400'
                      : line.includes('[TransparencyLogger]')
                      ? 'text-cyan-300'
                      : line.includes('✅')
                      ? 'text-emerald-300'
                      : line.includes('❌') || line.includes('BLOCKED')
                      ? 'text-red-400 font-semibold'
                      : line.includes('===')
                      ? 'text-white font-bold'
                      : ''
                  }>
                    {line}
                  </div>
                ))}
                <div ref={logEndRef} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


function Field({ label, children }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-500 mb-1">{label}</label>
      {children}
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div className="bg-gray-50 rounded-xl border border-gray-100 p-3 text-center">
      <p className="text-xl font-bold text-brand-600">{value ?? '—'}</p>
      <p className="text-xs text-gray-500 mt-0.5">{label}</p>
    </div>
  );
}
