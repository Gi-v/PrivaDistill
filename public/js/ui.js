/* ============================================================
   ui.js — DOM helpers & display logic
   ============================================================ */

const UI = {

  /* ── Terminal log ── */
  _logTarget: 'log-pipeline',

  setLogTarget(id) { this._logTarget = id; },

  log(msg, type = 'info') {
    const el = document.getElementById(this._logTarget);
    if (!el) return;
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const line = document.createElement('span');
    line.className = `log ${type}`;
    line.textContent = `[${ts}] ${msg}\n`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  },

  logTo(targetId, msg, type = 'info') {
    const prev = this._logTarget;
    this.setLogTarget(targetId);
    this.log(msg, type);
    this._logTarget = prev;
  },

  clearLog(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  },

  /* ── Status bar ── */
  setStatus(state) {
    const pulse = document.getElementById('status-pulse');
    const text  = document.getElementById('status-text');
    if (!pulse || !text) return;

    pulse.className = 'pulse';
    if (state === 'online') {
      pulse.classList.add('online');
      text.textContent = 'System Online';
    } else if (state === 'error') {
      pulse.classList.add('error');
      text.textContent = 'Offline';
    } else {
      text.textContent = 'Connecting…';
    }
  },

  /* ── Config panel ── */
  setConfig(cfg) {
    const set = (id, val, fallback = '—') => {
      const el = document.getElementById(id);
      if (el) el.textContent = val ?? fallback;
    };

    set('cfg-teacher',  cfg.model_name);
    set('cfg-epsilon',  cfg.epsilon);
    set('cfg-delta',    cfg.delta);
    set('cfg-noise',    cfg.noise_multiplier);

    set('topo-teacher', cfg.model_name);
    set('topo-epsilon', cfg.epsilon);
    set('topo-delta',   cfg.delta);
    set('topo-noise',   cfg.noise_multiplier);
    set('topo-clip',    cfg.max_grad_norm ?? '1.0');
    set('topo-batch',   cfg.batch_size    ?? '16');

    set('arch-teacher-name', `${cfg.model_name ?? 'GPT-2'} — frozen, generates soft labels`);
    set('arch-noise-desc',   `Gaussian noise σ=${cfg.noise_multiplier ?? '—'} injected per-gradient`);

    // Privacy meter
    const eps = parseFloat(cfg.epsilon);
    const fill   = document.getElementById('meter-fill');
    const rating = document.getElementById('meter-rating');
    if (fill && rating) {
      if (!isNaN(eps)) {
        if (eps < 1)  { fill.className = 'meter-fill strong';   rating.textContent = 'Strong'; }
        else if (eps < 10) { fill.className = 'meter-fill moderate'; rating.textContent = 'Moderate'; }
        else          { fill.className = 'meter-fill weak';     rating.textContent = 'Weak'; }
      }
    }
  },

  /* ── Button loading state ── */
  setLoading(btnId, loading, label) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    if (loading) {
      btn._origHTML = btn.innerHTML;
      btn.innerHTML = `<span class="spinner"></span> ${label || 'Running…'}`;
      btn.disabled = true;
    } else {
      if (btn._origHTML) btn.innerHTML = btn._origHTML;
      btn.disabled = false;
    }
  },

  /* ── Progress bar ── */
  setProgress(pct, msg, done = false) {
    const wrap  = document.getElementById('train-progress-wrap');
    const fill  = document.getElementById('progress-fill-el');
    const bar   = document.getElementById('progress-bar-el');
    const lbl   = document.getElementById('progress-label-text');
    const pctEl = document.getElementById('progress-pct-text');
    const msgEl = document.getElementById('progress-msg-el');

    if (!wrap) return;
    wrap.classList.remove('hidden');

    const p = Math.max(0, Math.min(100, pct));
    if (fill)  { fill.style.width = `${p}%`; if (done) fill.classList.add('success'); }
    if (bar)   bar.setAttribute('aria-valuenow', p);
    if (pctEl) pctEl.textContent = `${p}%`;
    if (lbl)   lbl.textContent   = done ? 'Complete' : 'Training…';
    if (msgEl) msgEl.textContent = msg ?? '';
  },

  hideProgress() {
    const wrap = document.getElementById('train-progress-wrap');
    if (wrap) wrap.classList.add('hidden');
  },

  /* ── Step state ── */
  markStepDone(n) {
    const el = document.getElementById(`step-${n}`);
    if (!el) return;
    el.classList.remove('active');
    el.classList.add('done');
    const num = el.querySelector('.step-num');
    if (num) num.textContent = '✓';
    const badge = el.querySelector('.step-badge');
    if (badge) { badge.className = 'step-badge done-tag'; badge.textContent = 'Done'; }
  },

  /* ── Inference result ── */
  showResult(text, ms) {
    const card   = document.getElementById('result-card');
    const output = document.getElementById('result-output');
    const pill   = document.getElementById('result-ms-pill');
    if (card)   card.classList.remove('hidden');
    if (output) output.textContent = text;
    if (pill)   pill.textContent = `${ms} ms`;
  },

  /* ── Session stats ── */
  _runs: 0, _totalMs: 0, _totalChars: 0,

  recordRun(ms, chars) {
    this._runs++;
    this._totalMs   += ms;
    this._totalChars += chars;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('stat-runs',    this._runs);
    set('stat-avg-ms',  Math.round(this._totalMs / this._runs));
    set('stat-tokens',  this._totalChars);
    set('stat-last-ms', ms);
  },

  /* ── Prompt history ── */
  _history: [],

  addHistory(prompt) {
    this._history.unshift(prompt);
    if (this._history.length > 15) this._history.pop();
    this._renderHistory();
  },

  _renderHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;
    if (!this._history.length) {
      list.innerHTML = '<div class="history-empty">No prompts yet. Run your first inference above.</div>';
      return;
    }
    list.innerHTML = this._history.map((p, i) =>
      `<div class="history-item" onclick="setPrompt(UI._history[${i}])" title="${p.replace(/"/g,'&quot;')}">${p.slice(0, 60)}${p.length > 60 ? '…' : ''}</div>`
    ).join('');
  },

  /* ── Prereq banner ── */
  hidePrereqBanner() {
    const el = document.getElementById('inference-prereq-banner');
    if (el) el.classList.add('hidden');
  },

  /* ── Char counter ── */
  updateCharCount(n, max) {
    const el = document.getElementById('char-count');
    if (!el) return;
    el.textContent = `${n} / ${max}`;
    el.className = 'char-count';
    if (n > max * 0.95) el.classList.add('danger');
    else if (n > max * 0.8) el.classList.add('warn');
  },
};