/* ============================================================
   MolPredict — Predict page JS
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const smilesInput   = document.getElementById('smiles-input');
  const predictBtn    = document.getElementById('predict-btn');
  const btnText       = document.getElementById('btn-text');
  const clearBtn      = document.getElementById('clear-btn');
  const resultCard    = document.getElementById('result-card');

  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultContent     = document.getElementById('result-content');
  const resultError       = document.getElementById('result-error');
  const resultLoader      = document.getElementById('result-loader');

  const resultBadge    = document.getElementById('result-badge');
  const resultLabelVal = document.getElementById('result-label-val');
  const confBar        = document.getElementById('conf-bar');
  const confPct        = document.getElementById('conf-pct');
  const resultSmilesOut= document.getElementById('result-smiles-out');
  const probActiveBar  = document.getElementById('prob-active-bar');
  const probInactiveBar= document.getElementById('prob-inactive-bar');
  const probActiveVal  = document.getElementById('prob-active-val');
  const probInactiveVal= document.getElementById('prob-inactive-val');
  const errorMsg       = document.getElementById('error-msg');

  const historyList    = document.getElementById('history-list');
  const clearHistBtn   = document.getElementById('clear-hist-btn');

  let history = [];

  /* ── Quick example buttons ─────────────────────────────── */
  document.querySelectorAll('.ex-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      smilesInput.value = btn.dataset.smi;
      smilesInput.focus();
    });
  });

  /* ── Clear input ───────────────────────────────────────── */
  clearBtn.addEventListener('click', () => {
    smilesInput.value = '';
    smilesInput.focus();
  });

  /* ── Show states ───────────────────────────────────────── */
  function showState(state) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display     = 'none';
    resultError.style.display       = 'none';
    resultLoader.style.display      = 'none';
    resultCard.classList.remove('has-result');

    if (state === 'placeholder') {
      resultPlaceholder.style.display = 'flex';
      resultCard.style.alignItems     = 'center';
      resultCard.style.justifyContent = 'center';
    } else if (state === 'loading') {
      resultLoader.style.display      = 'flex';
      resultCard.style.alignItems     = 'center';
      resultCard.style.justifyContent = 'center';
    } else if (state === 'result') {
      resultContent.style.display     = 'flex';
      resultCard.classList.add('has-result');
      resultCard.style.alignItems     = '';
      resultCard.style.justifyContent = '';
    } else if (state === 'error') {
      resultError.style.display       = 'flex';
      resultCard.style.alignItems     = 'center';
      resultCard.style.justifyContent = 'center';
    }
  }

  /* ── Populate result ───────────────────────────────────── */
  function populateResult(data) {
    const isActive = data.prediction === 'Active';
    const conf     = data.confidence;

    // badge
    resultBadge.textContent = data.prediction.toUpperCase();
    resultBadge.className   = 'result-badge ' + (isActive ? 'active' : 'inactive');
    resultLabelVal.textContent = data.prediction;
    resultLabelVal.style.color = isActive ? 'var(--green)' : 'var(--red)';

    // confidence bar
    const pct = (conf * 100).toFixed(1);
    confPct.textContent = pct + '%';
    requestAnimationFrame(() => {
      confBar.style.width = pct + '%';
    });

    // SMILES
    resultSmilesOut.textContent = data.smiles;

    // probability breakdown — try to get both class probs
    // Our API only returns max confidence; we infer the other class
    const activeProb   = isActive ? conf : 1 - conf;
    const inactiveProb = isActive ? 1 - conf : conf;

    const activePct   = (activeProb * 100).toFixed(1) + '%';
    const inactivePct = (inactiveProb * 100).toFixed(1) + '%';

    probActiveVal.textContent   = activePct;
    probInactiveVal.textContent = inactivePct;

    requestAnimationFrame(() => {
      probActiveBar.style.width   = activePct;
      probInactiveBar.style.width = inactivePct;
    });
  }

  /* ── Add to history ─────────────────────────────────────── */
  function addToHistory(data) {
    history.unshift(data);
    if (history.length > 20) history.pop();
    renderHistory();
  }

  function renderHistory() {
    if (history.length === 0) {
      historyList.innerHTML = '<p class="history-empty">No predictions yet.</p>';
      return;
    }
    historyList.innerHTML = history.map((item, i) => `
      <div class="history-item" data-idx="${i}">
        <span class="history-smiles" title="${item.smiles}">${item.smiles}</span>
        <span class="history-badge ${item.prediction === 'Active' ? 'active' : 'inactive'}">
          ${item.prediction}
        </span>
      </div>
    `).join('');

    historyList.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        const item = history[parseInt(el.dataset.idx)];
        smilesInput.value = item.smiles;
        showState('result');
        populateResult(item);
      });
    });
  }

  clearHistBtn.addEventListener('click', () => {
    history = [];
    renderHistory();
  });

  /* ── Main predict call ──────────────────────────────────── */
  async function runPrediction() {
    const smiles = smilesInput.value.trim();
    if (!smiles) {
      smilesInput.focus();
      smilesInput.style.borderColor = 'var(--red)';
      setTimeout(() => smilesInput.style.borderColor = '', 1200);
      return;
    }

    predictBtn.disabled = true;
    btnText.textContent = 'Running…';
    showState('loading');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles }),
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        errorMsg.textContent = data.error || `Server error (${res.status})`;
        showState('error');
      } else {
        showState('result');
        populateResult(data);
        addToHistory(data);
      }
    } catch (err) {
      errorMsg.textContent = 'Network error. Is the server running?';
      showState('error');
    } finally {
      predictBtn.disabled = false;
      btnText.textContent = 'Run Prediction';
    }
  }

  predictBtn.addEventListener('click', runPrediction);

  smilesInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runPrediction();
  });

});
