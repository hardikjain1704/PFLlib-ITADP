import { useState, useEffect } from 'react';
import { postConsent, getAllConsent } from '../api';

const PURPOSES = [
  { value: 'image_classification', label: 'Image Classification' },
  { value: 'text_classification', label: 'Text Classification' },
];

export default function ConsentManager() {
  const [clientId, setClientId] = useState(0);
  const [checked, setChecked] = useState(false);
  const [purpose, setPurpose] = useState('image_classification');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [allConsent, setAllConsent] = useState({});

  const fetchAllConsent = async () => {
    try {
      const { data } = await getAllConsent();
      setAllConsent(data || {});
    } catch {
      setAllConsent({});
    }
  };

  useEffect(() => {
    fetchAllConsent();
  }, []);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const { data } = await postConsent(clientId, checked, purpose);
      setResult(data);
      fetchAllConsent();
    } catch (err) {
      setResult({ error: true, message: err.response?.data?.detail || err.message });
    } finally {
      setLoading(false);
    }
  };

  const consentEntries = Object.values(allConsent);

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">
          Federated Learning Participation Consent
        </h2>
        <p className="mt-2 text-gray-500 text-sm leading-relaxed">
          Our federated learning system trains a <strong>PneumoniaMNIST</strong> classification model
          <strong> locally on your device</strong>. Only aggregated model weights — not raw data — are
          shared with the central server. Your participation requires consent and a valid training purpose.
        </p>
      </div>

      {/* Card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-6">
        {/* Client ID */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Client ID
          </label>
          <input
            type="number"
            min={0}
            max={1}
            value={clientId}
            onChange={(e) => setClientId(Number(e.target.value))}
            className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"
          />
          <p className="text-xs text-gray-400 mt-1">0 = Laptop A, 1 = Laptop B</p>
        </div>

        {/* Training Purpose */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Training Purpose
          </label>
          <select
            value={purpose}
            onChange={(e) => setPurpose(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"
          >
            {PURPOSES.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
          <p className="text-xs text-gray-400 mt-1">
            Only <strong>Image Classification</strong> is valid for PneumoniaMNIST.
            Selecting another purpose will block training.
          </p>
        </div>

        {/* Consent checkbox */}
        <label className="flex items-start gap-3 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            className="mt-1 h-5 w-5 rounded border-gray-300 text-brand-600 focus:ring-brand-500"
          />
          <span className="text-sm text-gray-700 leading-relaxed">
            I consent to use my local PneumoniaMNIST data shard for federated model training
            for the purpose of <strong>{PURPOSES.find(p => p.value === purpose)?.label}</strong>.
            I understand that only model weight updates will be shared and my raw data stays on my device.
          </span>
        </label>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-medium py-3 text-sm transition-colors disabled:opacity-50"
        >
          {loading ? 'Submitting…' : 'Submit Consent'}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div
          className={`rounded-2xl border p-5 text-sm ${
            result.error
              ? 'bg-red-50 border-red-200 text-red-700'
              : result.consent && result.purpose_valid
              ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
              : 'bg-amber-50 border-amber-200 text-amber-700'
          }`}
        >
          <p className="font-semibold mb-1">
            {result.error
              ? '❌ Error'
              : result.consent && result.purpose_valid
              ? '✅ Consent Granted – Training Allowed'
              : result.consent && !result.purpose_valid
              ? '⚠️ Consent Granted – But Purpose Invalid'
              : '⚠️ Consent Denied'}
          </p>
          <p>{result.message}</p>
          {result.purpose && (
            <p className="mt-1 text-xs opacity-70">Purpose: {result.purpose}</p>
          )}
          {result.timestamp && (
            <p className="mt-1 text-xs opacity-70">Recorded at: {result.timestamp}</p>
          )}
        </div>
      )}

      {/* All consent records table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700">All Consent Records</h3>
          <button
            onClick={fetchAllConsent}
            className="text-xs rounded-lg bg-gray-100 hover:bg-gray-200 px-3 py-1.5 font-medium text-gray-600 transition-colors"
          >
            🔄 Refresh
          </button>
        </div>
        {consentEntries.length === 0 ? (
          <p className="text-sm text-gray-400">No consent records yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  <th className="py-2 pr-4">Client ID</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 pr-4">Purpose</th>
                  <th className="py-2">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {consentEntries.map((rec) => (
                  <tr key={rec.client_id} className="border-b border-gray-50">
                    <td className="py-2 pr-4 font-medium">{rec.client_id}</td>
                    <td className="py-2 pr-4">
                      <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        rec.consent
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {rec.consent ? '✅ Granted' : '❌ Denied'}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-xs">
                      {rec.purpose || '—'}
                      {rec.purpose_valid === false && (
                        <span className="ml-1 text-red-500">⚠️ invalid</span>
                      )}
                    </td>
                    <td className="py-2 text-xs text-gray-400">{rec.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
