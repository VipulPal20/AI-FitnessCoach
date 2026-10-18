// Thin wrappers over the FastAPI backend (see app.py).

async function request(url, options) {
  const response = await fetch(url, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON error page; fall through with body = null.
  }
  if (!response.ok) {
    const detail = body?.detail || body?.message || `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body;
}

const postJSON = (url, payload) =>
  request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });

export const api = {
  health: () => request('/api/v1/health'),
  stats: () => request('/api/v1/stats'),

  async processFrame(blob, timestampSeconds) {
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');
    if (timestampSeconds !== undefined && timestampSeconds !== null) {
      form.append('timestamp_seconds', Number(timestampSeconds).toFixed(3));
    }
    const body = await request('/api/v1/process_frame', { method: 'POST', body: form });
    return body.data;
  },

  reset: () => postJSON('/api/v1/reset_counters'),
  setTarget: (action) => postJSON('/api/v1/target', { action: action || null }),
  settings: (values) => postJSON('/api/v1/settings', values),

  exercises(params) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== '' && value !== null && value !== undefined && value !== false) query.set(key, value);
    });
    return request(`/api/v1/exercises?${query.toString()}`);
  },
  exercise: (id) => request(`/api/v1/exercises/${encodeURIComponent(id)}`),
  categories: () => request('/api/v1/exercises/categories/summary'),
};

export const mediaUrl = (path) => (path ? `/exercise-media/${path}` : '');
