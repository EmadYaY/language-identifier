"""
Language Identification System - Web UI
File: app.py

Launches a local web interface for interactive language detection.
Requires the trained model (run 1_train_model_v3.py first).

Usage:
    python app.py
    Then open: http://localhost:5000

Author: Emad
Version: 1.0.0
"""

import pickle
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse
import os

# ── Load model ───────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "language_model.pkl"

def load_classifier():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"\n✗ Model not found at {MODEL_PATH}")
        print("  Run:  python 1_train_model_v3.py  first.\n")
        return None

CLASSIFIER = load_classifier()

# ── Language metadata ─────────────────────────────────────────────────────────

LANG_META = {
    "english":  {"flag": "🇬🇧", "label": "English"},
    "french":   {"flag": "🇫🇷", "label": "French"},
    "german":   {"flag": "🇩🇪", "label": "German"},
    "italian":  {"flag": "🇮🇹", "label": "Italian"},
    "persian":  {"flag": "🇮🇷", "label": "Persian"},
    "spanish":  {"flag": "🇪🇸", "label": "Spanish"},
}

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Language Identifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #0d0d0f;
    --surface: #141416;
    --border:  #242428;
    --accent:  #e8ff5a;
    --accent2: #5affd6;
    --text:    #f0f0f0;
    --muted:   #666672;
    --danger:  #ff5a5a;
    --r:       10px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 16px 80px;
    overflow-x: hidden;
  }

  /* ── Animated grid background ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(232,255,90,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(232,255,90,.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
  }

  .wrap {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 720px;
  }

  /* ── Header ── */
  header {
    margin-bottom: 40px;
    text-align: center;
  }

  .badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .12em;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 3px 10px;
    margin-bottom: 18px;
    opacity: .8;
  }

  h1 {
    font-size: clamp(2rem, 6vw, 3.4rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -.02em;
  }

  h1 span { color: var(--accent); }

  .subtitle {
    margin-top: 10px;
    color: var(--muted);
    font-size: .92rem;
    font-weight: 400;
  }

  /* ── Card ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 28px;
    margin-bottom: 20px;
  }

  /* ── Textarea ── */
  .textarea-wrap {
    position: relative;
  }

  textarea {
    width: 100%;
    min-height: 160px;
    resize: vertical;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 16px;
    font-family: 'DM Mono', monospace;
    font-size: .9rem;
    line-height: 1.6;
    outline: none;
    transition: border-color .2s;
  }

  textarea:focus { border-color: var(--accent); }
  textarea::placeholder { color: var(--muted); }

  .char-count {
    position: absolute;
    bottom: 10px;
    right: 14px;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: var(--muted);
    pointer-events: none;
  }

  /* ── Buttons row ── */
  .btn-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
  }

  button {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: .88rem;
    cursor: pointer;
    border: none;
    border-radius: var(--r);
    padding: 11px 22px;
    transition: transform .12s, opacity .12s, box-shadow .2s;
    letter-spacing: .02em;
  }

  button:active { transform: scale(.96); }

  .btn-detect {
    background: var(--accent);
    color: #0d0d0f;
    flex: 1;
  }
  .btn-detect:hover { box-shadow: 0 0 20px rgba(232,255,90,.4); }
  .btn-detect:disabled { opacity: .4; cursor: not-allowed; transform: none; }

  .btn-clear {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-clear:hover { border-color: var(--text); color: var(--text); }

  .btn-quit {
    background: transparent;
    color: var(--danger);
    border: 1px solid rgba(255,90,90,.3);
  }
  .btn-quit:hover { background: rgba(255,90,90,.08); border-color: var(--danger); }

  /* ── Result area ── */
  #result {
    display: none;
    animation: fadeUp .35s ease forwards;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-top {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
  }

  .lang-flag {
    font-size: 3rem;
    line-height: 1;
  }

  .lang-info h2 {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
  }

  .lang-info .conf-label {
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    color: var(--accent2);
    margin-top: 4px;
  }

  .low-conf-warn {
    display: none;
    background: rgba(255,90,90,.08);
    border: 1px solid rgba(255,90,90,.25);
    border-radius: var(--r);
    padding: 10px 14px;
    font-size: .83rem;
    color: var(--danger);
    margin-bottom: 20px;
    font-family: 'DM Mono', monospace;
  }

  /* ── Probability bars ── */
  .bars-title {
    font-size: .75rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    font-family: 'DM Mono', monospace;
  }

  .bar-row {
    display: grid;
    grid-template-columns: 100px 1fr 54px;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }

  .bar-lang {
    font-size: .82rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    text-align: right;
  }
  .bar-lang.top { color: var(--text); font-weight: 600; }

  .bar-track {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 99px;
    background: var(--accent);
    width: 0;
    transition: width .6s cubic-bezier(.16,1,.3,1);
  }
  .bar-fill.other { background: var(--muted); }

  .bar-pct {
    font-family: 'DM Mono', monospace;
    font-size: .78rem;
    color: var(--muted);
    text-align: right;
  }
  .bar-pct.top { color: var(--accent); }

  /* ── History ── */
  #history-section { display: none; }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
  }

  .section-label {
    font-size: .75rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
  }

  .btn-clear-hist {
    background: none;
    border: none;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: var(--muted);
    cursor: pointer;
    padding: 0;
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .btn-clear-hist:hover { color: var(--text); }

  .hist-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 260px;
    overflow-y: auto;
  }

  .hist-item {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    cursor: pointer;
    transition: border-color .15s;
  }
  .hist-item:hover { border-color: var(--muted); }

  .hist-flag { font-size: 1.3rem; flex-shrink: 0; }

  .hist-text {
    font-family: 'DM Mono', monospace;
    font-size: .78rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  }

  .hist-lang {
    font-size: .78rem;
    font-weight: 600;
    color: var(--text);
    flex-shrink: 0;
  }

  /* ── Loading spinner ── */
  .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 2px solid rgba(13,13,15,.4);
    border-top-color: #0d0d0f;
    border-radius: 50%;
    animation: spin .6s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Quit overlay ── */
  #quit-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(13,13,15,.85);
    backdrop-filter: blur(6px);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  #quit-overlay.show { display: flex; }

  .quit-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 32px;
    max-width: 360px;
    width: 90%;
    text-align: center;
    animation: fadeUp .25s ease;
  }

  .quit-box h3 { font-size: 1.3rem; margin-bottom: 8px; }
  .quit-box p  { color: var(--muted); font-size: .88rem; margin-bottom: 24px; }

  .quit-btns { display: flex; gap: 12px; }
  .quit-btns button { flex: 1; padding: 12px; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
</head>
<body>

<div class="wrap">

  <!-- Header -->
  <header>
    <div class="badge">v1.0.0 · Naive Bayes · Bigram + Trigram</div>
    <h1>Language<br><span>Identifier</span></h1>
    <p class="subtitle">Paste or type any text — the model detects the language instantly.</p>
  </header>

  <!-- Input card -->
  <div class="card">
    <div class="textarea-wrap">
      <textarea id="inputText" placeholder="Type or paste text here…&#10;&#10;Examples:&#10;  Hello, how are you?&#10;  Bonjour le monde&#10;  سلام دنیا"></textarea>
      <span class="char-count" id="charCount">0</span>
    </div>

    <div class="btn-row">
      <button class="btn-detect" id="detectBtn" onclick="detect()">
        <span id="btnLabel">Detect Language</span>
        <span class="spinner" id="spinner"></span>
      </button>
      <button class="btn-clear" onclick="clearInput()" title="Clear text">Clear</button>
      <button class="btn-quit"  onclick="showQuit()"  title="Exit app">Quit</button>
    </div>
  </div>

  <!-- Result card -->
  <div class="card" id="result">
    <div class="result-top">
      <div class="lang-flag" id="resFlag">🌐</div>
      <div class="lang-info">
        <h2 id="resName">—</h2>
        <div class="conf-label" id="resConf"></div>
      </div>
    </div>

    <div class="low-conf-warn" id="lowConfWarn">
      ⚠ Text is very short — result may be unreliable.
    </div>

    <div class="bars-title">Probability breakdown</div>
    <div id="barsContainer"></div>
  </div>

  <!-- History -->
  <div class="card" id="history-section">
    <div class="history-header">
      <span class="section-label">Recent detections</span>
      <button class="btn-clear-hist" onclick="clearHistory()">Clear history</button>
    </div>
    <div class="hist-list" id="histList"></div>
  </div>

</div>

<!-- Quit overlay -->
<div id="quit-overlay">
  <div class="quit-box">
    <h3>Quit the app?</h3>
    <p>This will shut down the local server.<br>Close this tab afterwards.</p>
    <div class="quit-btns">
      <button class="btn-clear" onclick="hideQuit()">Cancel</button>
      <button class="btn-quit" onclick="confirmQuit()">Quit</button>
    </div>
  </div>
</div>

<script>
const LANG_META = __LANG_META__;

let history = [];

// ── Character counter ─────────────────────────────────────────────────────
const ta = document.getElementById('inputText');
const cc = document.getElementById('charCount');
ta.addEventListener('input', () => { cc.textContent = ta.value.length; });

// Keyboard shortcut: Ctrl/Cmd + Enter → detect
ta.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') detect();
});

// ── Detect ────────────────────────────────────────────────────────────────
async function detect() {
  const text = ta.value.trim();
  if (!text) return;

  const btn   = document.getElementById('detectBtn');
  const label = document.getElementById('btnLabel');
  const spin  = document.getElementById('spinner');

  btn.disabled = true;
  label.style.display = 'none';
  spin.style.display  = 'block';

  try {
    const res  = await fetch('/detect', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ text })
    });
    const data = await res.json();

    if (data.error) { alert('Error: ' + data.error); return; }

    renderResult(data);
    addHistory(data, text);
  } catch(e) {
    alert('Could not reach the server. Is it still running?');
  } finally {
    btn.disabled = false;
    label.style.display = 'inline';
    spin.style.display  = 'none';
  }
}

// ── Render result ─────────────────────────────────────────────────────────
function renderResult(data) {
  const meta = LANG_META[data.language] || { flag: '🌐', label: data.language };

  document.getElementById('resFlag').textContent = meta.flag;
  document.getElementById('resName').textContent = meta.label;
  document.getElementById('resConf').textContent =
    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  document.getElementById('lowConfWarn').style.display =
    data.low_confidence ? 'block' : 'none';

  // Bars
  const container = document.getElementById('barsContainer');
  container.innerHTML = '';

  const sorted = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1]);

  sorted.forEach(([lang, prob], i) => {
    const m    = LANG_META[lang] || { flag: '🌐', label: lang };
    const isTop = i === 0;
    const pct  = (prob * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-lang ${isTop ? 'top' : ''}">${m.flag} ${m.label}</span>
      <div class="bar-track">
        <div class="bar-fill ${isTop ? '' : 'other'}" data-pct="${prob}"></div>
      </div>
      <span class="bar-pct ${isTop ? 'top' : ''}">${pct}%</span>
    `;
    container.appendChild(row);
  });

  // Show result with animation
  const resultEl = document.getElementById('result');
  resultEl.style.display = 'block';
  resultEl.style.animation = 'none';
  void resultEl.offsetWidth;
  resultEl.style.animation = '';

  // Animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = (parseFloat(el.dataset.pct) * 100) + '%';
    });
  });
}

// ── History ───────────────────────────────────────────────────────────────
function addHistory(data, text) {
  history.unshift({ data, text });
  if (history.length > 10) history.pop();
  renderHistory();
}

function renderHistory() {
  const section = document.getElementById('history-section');
  const list    = document.getElementById('histList');

  if (!history.length) { section.style.display = 'none'; return; }
  section.style.display = 'block';

  list.innerHTML = '';
  history.forEach((entry, idx) => {
    const m    = LANG_META[entry.data.language] || { flag: '🌐', label: entry.data.language };
    const item = document.createElement('div');
    item.className = 'hist-item';
    item.title = 'Click to restore';
    item.innerHTML = `
      <span class="hist-flag">${m.flag}</span>
      <span class="hist-text">${escHtml(entry.text.slice(0, 80))}${entry.text.length > 80 ? '…' : ''}</span>
      <span class="hist-lang">${m.label}</span>
    `;
    item.onclick = () => restoreHistory(idx);
    list.appendChild(item);
  });
}

function restoreHistory(idx) {
  const entry = history[idx];
  ta.value = entry.text;
  cc.textContent = entry.text.length;
  renderResult(entry.data);
}

function clearHistory() {
  history = [];
  renderHistory();
}

// ── Utilities ─────────────────────────────────────────────────────────────
function clearInput() {
  ta.value = '';
  cc.textContent = '0';
  document.getElementById('result').style.display = 'none';
  ta.focus();
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Quit overlay ──────────────────────────────────────────────────────────
function showQuit() { document.getElementById('quit-overlay').classList.add('show'); }
function hideQuit() { document.getElementById('quit-overlay').classList.remove('show'); }

async function confirmQuit() {
  try { await fetch('/quit', { method: 'POST' }); } catch(_) {}
  document.querySelector('.wrap').innerHTML =
    '<div style="text-align:center;padding:80px 0;color:var(--muted)">' +
    '<div style="font-size:2rem;margin-bottom:16px">👋</div>' +
    '<div style="font-family:\'DM Mono\',monospace;font-size:.9rem">Server stopped.<br>You can close this tab.</div>' +
    '</div>';
  document.getElementById('quit-overlay').classList.remove('show');
}
</script>
</body>
</html>
"""

# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log; we print our own

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            meta_json = json.dumps(LANG_META)
            page = HTML.replace("__LANG_META__", meta_json).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(page))
            self.end_headers()
            self.wfile.write(page)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        # ── /detect ──────────────────────────────────────────────────────
        if self.path == "/detect":
            if CLASSIFIER is None:
                self.send_json({"error": "Model not loaded. Run 1_train_model_v3.py first."}, 500)
                return
            try:
                payload = json.loads(body)
                text    = payload.get("text", "").strip()
                if not text:
                    self.send_json({"error": "Empty text."}, 400)
                    return

                language, probs = CLASSIFIER.predict_with_confidence(text)
                low_conf = bool(probs.pop("low_confidence", False))
                confidence = probs.get(language, 0.0)

                print(f"  → [{language.upper():10s}]  {confidence:.1%}  "
                      f"{'⚠ low-conf' if low_conf else ''}  "
                      f"\"{text[:60]}{'…' if len(text)>60 else ''}\"")

                self.send_json({
                    "language":      language,
                    "confidence":    confidence,
                    "low_confidence": low_conf,
                    "probabilities": probs,
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        # ── /quit ─────────────────────────────────────────────────────────
        elif self.path == "/quit":
            self.send_json({"status": "bye"})
            print("\n  Server shutting down …\n")
            # Shutdown in a new thread so response can be sent first
            import threading
            threading.Timer(0.3, server.shutdown).start()

        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

PORT = 5000

if __name__ == "__main__":
    if CLASSIFIER is None:
        exit(1)

    server = HTTPServer(("localhost", PORT), Handler)

    print("\n" + "="*52)
    print("  Language Identifier — Web UI  v1.0.0")
    print("="*52)
    print(f"\n  ✓ Model loaded  ({len(CLASSIFIER.languages)} languages)")
    print(f"  ✓ Server running at  http://localhost:{PORT}")
    print(f"\n  Open the URL above in your browser.")
    print(f"  Press  Ctrl+C  or click Quit in the UI to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.\n")
