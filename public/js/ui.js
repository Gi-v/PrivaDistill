const UI = {
  log(msg, type = 'system') {
    const body = document.getElementById('console-output');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    entry.textContent = `[${ts}] ${msg}`;
    body.appendChild(entry);
    body.scrollTop = body.scrollHeight;
  },

  setStatus(online) {
    const dot = document.getElementById('status-pulse');
    const text = document.getElementById('status-text');
    dot.className = `pulse-dot ${online ? 'online' : 'offline'}`;
    text.textContent = online ? 'System Online' : 'System Offline';
  },

  setConfig(cfg) {
    document.getElementById('val-teacher').textContent = cfg.model_name ?? '—';
    document.getElementById('val-epsilon').textContent = cfg.epsilon ?? '—';
    document.getElementById('val-delta').textContent = cfg.delta ?? '—';
    document.getElementById('val-noise').textContent = cfg.noise_multiplier ?? '—';
  },

  setButtonLoading(btn, loading) {
    btn.disabled = loading;
    if (loading) {
      btn.dataset.originalText = btn.innerHTML;
      btn.innerHTML = '<span class="spinner"></span> Running…';
    } else {
      btn.innerHTML = btn.dataset.originalText || btn.innerHTML;
    }
  },

  showBadge(id, text, duration = 2500) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.remove('hidden');
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.add('hidden'), duration);
  },

  showOutput(text, ms) {
    const out = document.getElementById('inference-result');
    if (!out) return;
    out.classList.remove('hidden');
    document.getElementById('result-text').textContent = text;
    document.getElementById('result-ms').textContent = `${ms} ms`;
  },

  updateProgress(pct, label) {
    const bar = document.getElementById('progress-fill');
    const lbl = document.getElementById('progress-label');
    if (bar) bar.style.width = `${pct}%`;
    if (lbl) lbl.textContent = label;
  },
};