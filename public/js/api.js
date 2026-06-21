const API_BASE = window.location.origin;

const Api = {
  async _fetch(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(API_BASE + path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
    return data;
  },
  get: (path) => Api._fetch('GET', path),
  post: (path, body = {}) => Api._fetch('POST', path, body),
};