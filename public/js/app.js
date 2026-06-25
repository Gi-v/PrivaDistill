/* ============================================================
   app.js — SPA router, pipeline, inference, topology
   ============================================================ */

/* ── Navigation ── */
function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  const el = document.getElementById(`page-${page}`);
  if (el) el.classList.add('active');
  const tab = document.querySelector(`[data-page="${page}"]`);
  if (tab) tab.classList.add('active');
  UI.setLogTarget(page === 'inference' ? 'log-inference' : 'log-pipeline');
}

/* ── Step selector ── */
let currentStep = 1;
function selectStep(n) {
  currentStep = n;
  document.querySelectorAll('.step-item').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(`step-${n}`);
  if (el && !el.classList.contains('done')) el.classList.add('active');
  [1,2,3].forEach(i => {
    const c = document.getElementById(`step-content-${i}`);
    if (c) c.classList.toggle('hidden', i !== n);
  });
}

/* ── Run a pipeline step ── */
async function runStep(action, btnId) {
  UI.setLoading(btnId, true);
  UI.log(`POST ${action}…`);
  const msgEl = {
    '/train':    'train-status-msg',
    '/export':   'export-status-msg',
    '/quantize': 'quantize-status-msg',
  }[action];

  try {
    const data = await Api.post(action);
    const msg  = data.status ?? data.message ?? 'ok';

    if (action === '/quantize' && data.status === 'skipped') {
      UI.log(`/quantize → skipped (no CUDA GPU detected — this is normal)`, 'warn');
      setStatusMsg(msgEl, 'Skipped — no GPU', 'warn');
    } else {
      UI.log(`${action} → ${msg}`, 'success');
      setStatusMsg(msgEl, msg, 'success');
      const stepMap = { '/train': 1, '/export': 2, '/quantize': 3 };
      UI.markStepDone(stepMap[action]);
      if (action === '/train')  pollTraining();
      if (action === '/export') UI.hidePrereqBanner();
    }
  } catch (e) {
    UI.log(`${action} failed: ${e.message}`, 'error');
    setStatusMsg(msgEl, `Failed: ${e.message}`, 'error');
  } finally {
    UI.setLoading(btnId, false);
  }
}

function setStatusMsg(id, text, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.style.color = type === 'success' ? 'var(--green)' : type === 'error' ? 'var(--red)' : 'var(--amber)';
}

/* ── Training poller ── */
function pollTraining() {
  UI.setProgress(0, 'Starting…');
  const iv = setInterval(async () => {
    try {
      const s = await Api.get('/status');
      const pct = s.progress ?? 0;
      UI.setProgress(pct, s.message);
      UI.log(`Training ${pct}% — ${s.message ?? ''}`, 'info');
      if (s.status !== 'running') {
        clearInterval(iv);
        const done = s.status === 'completed';
        UI.setProgress(done ? 100 : pct, s.message, done);
        UI.log(`Training ${s.status}: ${s.message}`, done ? 'success' : 'error');
      }
    } catch { clearInterval(iv); }
  }, 2000);
}

/* ── Inference ── */
async function runInference() {
  const ta   = document.getElementById('prompt-input');
  const text = ta?.value.trim();
  if (!text) return;

  UI.setLoading('btn-analyze', true, 'Running…');
  UI.logTo('log-inference', `Inference request (${text.length} chars)…`);

  try {
    const t0   = Date.now();
    const data = await Api.post('/analyze', { text });
    const ms   = data.inference_ms ?? (Date.now() - t0);
    UI.showResult(data.generated_text, ms);
    UI.recordRun(ms, text.length);
    UI.addHistory(text);
    UI.logTo('log-inference', `Done in ${ms} ms`, 'success');
  } catch (e) {
    UI.logTo('log-inference', `Inference failed: ${e.message}`, 'error');
  } finally {
    UI.setLoading('btn-analyze', false);
  }
}

/* ── Quick prompt helper ── */
function setPrompt(text) {
  const ta = document.getElementById('prompt-input');
  if (!ta) return;
  ta.value = text;
  ta.dispatchEvent(new Event('input'));
  ta.focus();
}

/* ── Char counter ── */
function initCharCounter() {
  const ta  = document.getElementById('prompt-input');
  const btn = document.getElementById('btn-analyze');
  if (!ta) return;
  ta.addEventListener('input', () => {
    const n = ta.value.length;
    UI.updateCharCount(n, 4096);
    if (btn) btn.disabled = n === 0;
  });
}

/* ── Boot ── */
document.addEventListener('DOMContentLoaded', async () => {
  initCharCounter();
  selectStep(1);

  // Health check
  try {
    const h = await Api.get('/health');
    UI.setStatus('online');
    UI.log(`Backend online — model loaded: ${h.model_loaded}`, 'success');
    if (h.model_loaded) UI.hidePrereqBanner();
  } catch {
    UI.setStatus('error');
    UI.log('Backend unreachable. Is docker compose up -d running?', 'error');
  }

  // Config
  try {
    const cfg = await Api.get('/config');
    UI.setConfig(cfg);
    UI.log(`Config: ε=${cfg.epsilon}, δ=${cfg.delta}, σ=${cfg.noise_multiplier}`, 'info');
  } catch {
    UI.log('Could not load config.', 'warn');
  }
});