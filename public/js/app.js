document.addEventListener('DOMContentLoaded', async () => {
  UI.log('Initializing PrivaDistill sandbox…');

  // Boot: health + config
  try {
    const health = await Api.get('/health');
    UI.setStatus(true);
    UI.log(`Backend online. Model loaded: ${health.model_loaded}`, 'success');
  } catch {
    UI.setStatus(false);
    UI.log('Backend unreachable — running in demo mode.', 'error');
  }

  try {
    const cfg = await Api.get('/config');
    UI.setConfig(cfg);
    UI.log(`Config loaded — ε=${cfg.epsilon}, δ=${cfg.delta}`, 'system');
  } catch {
    UI.log('Could not fetch config.', 'error');
  }

  // Char counter
  const textarea = document.getElementById('prompt-input');
  const analyzeBtn = document.getElementById('btn-analyze');
  const charVal = document.getElementById('char-val');
  textarea.addEventListener('input', () => {
    const n = textarea.value.length;
    charVal.textContent = n;
    analyzeBtn.disabled = n === 0;
  });

  // Pipeline buttons
  document.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const action = btn.dataset.action;
      UI.setButtonLoading(btn, true);
      UI.log(`POST ${action}…`);

      try {
        const data = await Api.post(action);
        UI.log(`${action} → ${data.status ?? data.message ?? 'ok'}`, 'success');
        UI.showBadge('save-badge', action === '/train' ? 'Training started' : 'Done');

        // Poll /status while training runs
        if (action === '/train') pollTrainingStatus();
      } catch (e) {
        UI.log(`${action} failed: ${e.message}`, 'error');
      } finally {
        UI.setButtonLoading(btn, false);
      }
    });
  });

  // Inference
  analyzeBtn.addEventListener('click', async () => {
    const text = textarea.value.trim();
    if (!text) return;
    UI.setButtonLoading(analyzeBtn, true);
    UI.log(`Inference request (${text.length} chars)…`);
    try {
      const data = await Api.post('/analyze', { text });
      UI.showOutput(data.generated_text, data.inference_ms);
      UI.log(`Done in ${data.inference_ms} ms`, 'success');
      UI.showBadge('save-badge', 'Memory Synced');
    } catch (e) {
      UI.log(`Inference failed: ${e.message}`, 'error');
    } finally {
      UI.setButtonLoading(analyzeBtn, false);
    }
  });
});

// Poll /status every 2 s while training is running
function pollTrainingStatus() {
  const interval = setInterval(async () => {
    try {
      const s = await Api.get('/status');
      UI.updateProgress(s.progress, s.message);
      if (s.status !== 'running') {
        clearInterval(interval);
        const type = s.status === 'completed' ? 'success' : 'error';
        UI.log(`Training ${s.status}: ${s.message}`, type);
        if (s.status === 'completed') UI.updateProgress(100, 'Training complete');
      }
    } catch {
      clearInterval(interval);
    }
  }, 2000);
}