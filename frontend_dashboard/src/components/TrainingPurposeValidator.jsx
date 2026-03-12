import { useState } from 'react';
import { validatePurpose, getPurposeViolations } from '../api';

const PURPOSES = [
  { value: 'image_classification', label: 'Image Classification', features: ['image', 'label'] },
  { value: 'text_classification', label: 'Text Classification', features: ['text', 'label'] },
];

export default function TrainingPurposeValidator() {
  const [purpose, setPurpose] = useState(PURPOSES[0].value);
  const [featureInput, setFeatureInput] = useState('image, label');
  const [result, setResult] = useState(null);
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleValidate = async () => {
    setLoading(true);
    setResult(null);
    try {
      const features = featureInput.split(',').map((f) => f.trim()).filter(Boolean);
      const { data } = await validatePurpose(purpose, features);
      setResult(data);
    } catch (err) {
      setResult({ error: true, message: err.response?.data?.detail || err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleFetchViolations = async () => {
    try {
      const { data } = await getPurposeViolations();
      setViolations(data || []);
    } catch {
      setViolations([]);
    }
  };

  const parsedFeatures = featureInput.split(',').map((f) => f.trim()).filter(Boolean);

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Purpose Validation</h2>
        <p className="mt-2 text-gray-500 text-sm leading-relaxed">
          Validate that dataset features comply with the declared training purpose before
          federated training begins. <strong>PneumoniaMNIST</strong> only supports
          <strong> image_classification</strong> with features <code>image, label</code>.
        </p>
      </div>

      {/* Card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-6">
        {/* Purpose */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Training Purpose
          </label>
          <select
            value={purpose}
            onChange={(e) => {
              setPurpose(e.target.value);
              const match = PURPOSES.find((p) => p.value === e.target.value);
              if (match) setFeatureInput(match.features.join(', '));
            }}
            className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"
          >
            {PURPOSES.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>

        {/* Features */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Dataset Features (comma-separated)
          </label>
          <input
            type="text"
            value={featureInput}
            onChange={(e) => setFeatureInput(e.target.value)}
            placeholder="image, label"
            className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"
          />
          <div className="flex flex-wrap gap-2 mt-2">
            {parsedFeatures.map((f) => (
              <span key={f} className="inline-flex items-center rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-700">
                {f}
              </span>
            ))}
          </div>
        </div>

        {/* Validate */}
        <button
          onClick={handleValidate}
          disabled={loading}
          className="w-full rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-medium py-3 text-sm transition-colors disabled:opacity-50"
        >
          {loading ? 'Validating…' : 'Validate Purpose'}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div
          className={`rounded-2xl border p-5 text-sm space-y-3 ${
            result.error
              ? 'bg-red-50 border-red-200 text-red-700'
              : result.valid
              ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
              : 'bg-amber-50 border-amber-200 text-amber-700'
          }`}
        >
          <p className="font-semibold">
            {result.error ? '❌ Error' : result.valid ? '✅ Validation Passed' : '⚠️ Validation Failed'}
          </p>
          <p>{result.message}</p>

          {!result.error && (
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1">Allowed Features</p>
                <ul className="space-y-1">
                  {(result.allowed_features || []).map((f) => (
                    <li key={f} className="flex items-center gap-1 text-emerald-700">✔ {f}</li>
                  ))}
                </ul>
              </div>
              {result.invalid_features?.length > 0 && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide mb-1">Disallowed Features</p>
                  <ul className="space-y-1">
                    {result.invalid_features.map((f) => (
                      <li key={f} className="flex items-center gap-1 text-red-600">❌ {f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Violations */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700">Purpose Violation History</h3>
          <button onClick={handleFetchViolations}
            className="text-xs rounded-lg bg-gray-100 hover:bg-gray-200 px-3 py-1.5 font-medium text-gray-600 transition-colors">
            🔄 Refresh
          </button>
        </div>
        {violations.length === 0 ? (
          <p className="text-sm text-gray-400">No violations logged.</p>
        ) : (
          <ul className="space-y-2 max-h-48 overflow-y-auto">
            {violations.map((v, idx) => (
              <li key={idx} className="text-xs bg-red-50 border border-red-100 rounded-lg p-3">
                <span className="font-semibold text-red-700">{v.purpose}</span>
                <span className="text-red-500 ml-2">Invalid: {(v.invalid_features || []).join(', ')}</span>
                {v.timestamp && <span className="block text-red-300 mt-1">{v.timestamp}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
