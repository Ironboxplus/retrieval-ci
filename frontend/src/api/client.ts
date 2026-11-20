import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    // A 404 means "this specific resource doesn't exist" — an expected,
    // handled outcome (e.g. a task referencing a since-deleted document;
    // callers like useDocumentDetail/usePaperDetail fall back gracefully),
    // not an application error. Every other status still logs.
    if (err.response?.status !== 404) {
      const msg = err.response?.data?.detail || err.message;
      console.error('[API]', err.config?.url, msg);
    }
    return Promise.reject(err);
  },
);

export default api;
