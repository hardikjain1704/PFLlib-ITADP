import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Consent (now includes purpose) ───────────────────────────
export const postConsent = (clientId, consent, purpose = 'image_classification') =>
  api.post('/consent', { client_id: clientId, consent, purpose });

export const getConsent = (clientId) =>
  api.get(`/consent/${clientId}`);

export const getAllConsent = () =>
  api.get('/consent');

// ── Purpose validation ───────────────────────────────────────
export const validatePurpose = (purpose, datasetFeatures) =>
  api.post('/validate-purpose', { purpose, dataset_features: datasetFeatures });

export const getPurposeViolations = () =>
  api.get('/purpose-violations');

// ── Transparency ─────────────────────────────────────────────
export const getUserInfo = (userId) =>
  api.get(`/user-info/${userId}`);

export const getTransparencyLog = () =>
  api.get('/transparency-log');

export const getTransparencyStats = () =>
  api.get('/transparency-stats');

// ── Training control (federated) ─────────────────────────────
export const startTraining = (config) =>
  api.post('/start-training', config);

export const getTrainingStatus = () =>
  api.get('/training-status');

export const stopTraining = () =>
  api.post('/stop-training');

// ── Training log (per-round details) ─────────────────────────
export const getTrainingLog = () =>
  api.get('/training-log');

// ── Client info ──────────────────────────────────────────────
export const getClientInfo = (clientId) =>
  api.get(`/client/${clientId}`);

export default api;
