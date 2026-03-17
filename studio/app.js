/**
 * Viterbox Studio — Audiobook Production Frontend
 * Manages sentence splitting, per-sentence generation, playback, and export.
 */

// ═══════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════

const API = '';  // Same origin

/** Build a session-aware audio URL.  e.g. /api/audio/session_id/001_abc.wav
 *  @param {string} filename
 *  @param {number} [version] - cache-busting version (incremented on regen)
 */
function audioUrl(filename, version) {
    let url;
    if (state.session_id) url = `${API}/api/audio/${state.session_id}/${filename}`;
    else url = `${API}/api/audio/${filename}`;
    if (version) url += `?v=${version}`;
    return url;
}
let state = {
    sentences: [],           // [{id, index, text, status, audio_file, duration, pause_after_ms}]
    session_id: null,        // Session folder for organized output
    isGenerating: false,
    currentlyPlaying: null,  // sentence id being played
    globalAudio: null,       // Audio element for global playback
    isPlayingAll: false,
    abortController: null,
};

// ═══════════════════════════════════════════════════════════
//  DOM References
// ═══════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    editor: $('#script-editor'),
    charCount: $('#char-count'),
    splitBtn: $('#split-btn'),
    smartSplitBtn: $('#smart-split-btn'),
    clearBtn: $('#clear-btn'),
    sentencesList: $('#sentences-list'),
    sentenceCount: $('#sentence-count'),
    generateAllBtn: $('#generate-all-btn'),
    voiceSelect: $('#voice-select'),
    languageSelect: $('#language-select'),
    settingsBtn: $('#settings-btn'),
    settingsModal: $('#settings-modal'),
    playAllBtn: $('#play-all-btn'),
    stopBtn: $('#stop-btn'),
    currentTime: $('#current-time'),
    totalTime: $('#total-time'),
    progressFill: $('#progress-fill'),
    progressCursor: $('#progress-cursor'),
    exportBtn: $('#export-btn'),
    globalPause: $('#global-pause'),
    toastContainer: $('#toast-container'),
    saveConfigBtn: $('#save-config-btn'),
    loadConfigBtn: $('#load-config-btn'),
    configFileInput: $('#config-file-input'),
    voicePreviewBtn: $('#voice-preview-btn'),
    exportSrtBtn: $('#export-srt-btn'),
    importVoiceBtn: $('#import-voice-btn'),
    voiceFileInput: $('#voice-file-input'),
};

// ═══════════════════════════════════════════════════════════
//  Initialization
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await loadVoices();
    setupSettingsSliders();
});

function setupEventListeners() {
    // Editor
    DOM.editor.addEventListener('input', updateCharCount);
    DOM.splitBtn.addEventListener('click', splitText);
    DOM.smartSplitBtn.addEventListener('click', smartSplitText);
    DOM.clearBtn.addEventListener('click', clearAll);

    // Settings modal
    DOM.settingsBtn.addEventListener('click', () => DOM.settingsModal.hidden = false);
    DOM.settingsModal.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', () => DOM.settingsModal.hidden = true);
    });
    DOM.settingsModal.querySelector('.modal-backdrop').addEventListener('click', () => {
        DOM.settingsModal.hidden = true;
    });

    // Generate all
    DOM.generateAllBtn.addEventListener('click', generateAll);

    // Global playback
    DOM.playAllBtn.addEventListener('click', playAll);
    DOM.stopBtn.addEventListener('click', stopPlayback);

    // Export
    DOM.exportBtn.addEventListener('click', exportAudio);
    DOM.exportSrtBtn.addEventListener('click', exportSrt);

    // Config import/export
    DOM.saveConfigBtn.addEventListener('click', exportConfig);
    DOM.loadConfigBtn.addEventListener('click', () => DOM.configFileInput.click());
    DOM.configFileInput.addEventListener('change', importConfig);

    // Voice preview
    DOM.voicePreviewBtn.addEventListener('click', previewVoice);

    // Voice import
    DOM.importVoiceBtn.addEventListener('click', () => DOM.voiceFileInput.click());
    DOM.voiceFileInput.addEventListener('change', importVoice);

    // Global pause change → update all sentences
    DOM.globalPause.addEventListener('change', applyGlobalPause);
}

// Master voice list — kept in sync so per-sentence dropdowns can be rebuilt
let voiceList = [];

async function loadVoices() {
    try {
        const res = await fetch(`${API}/api/voices`);
        const data = await res.json();
        voiceList = data.voices;
        _populateVoiceSelect(DOM.voiceSelect, voiceList, 'voice0');
    } catch (e) {
        toast('Could not load voices', 'error');
    }
}

/** Populate any <select> element with the current voice list */
function _populateVoiceSelect(selectEl, voices, defaultHint) {
    const prev = selectEl.value;  // preserve current selection
    selectEl.innerHTML = '';
    voices.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.path;
        opt.textContent = v.name;
        selectEl.appendChild(opt);
    });
    // Restore previous selection or fall back to hint
    if (prev && [...selectEl.options].some(o => o.value === prev)) {
        selectEl.value = prev;
    } else if (defaultHint) {
        const match = [...selectEl.options].find(o => o.textContent.includes(defaultHint));
        if (match) selectEl.value = match.value;
    }
}

async function importVoice(e) {
    const file = e.target.files[0];
    if (!file) return;

    DOM.importVoiceBtn.classList.add('uploading');
    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API}/api/import-voice`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Upload failed');
        }

        const newVoice = await res.json();
        // Add to master voice list
        voiceList.push(newVoice);

        // Refresh global dropdown and select the new voice
        _populateVoiceSelect(DOM.voiceSelect, voiceList);
        DOM.voiceSelect.value = newVoice.path;

        // Refresh all per-sentence voice dropdowns
        DOM.sentencesList.querySelectorAll('.sc-voice-select').forEach(sel => {
            _populateVoiceSelect(sel, voiceList);
        });

        toast(`📥 Imported voice: ${newVoice.name}`, 'success');
    } catch (err) {
        toast(`Voice import failed: ${err.message}`, 'error');
    } finally {
        DOM.importVoiceBtn.classList.remove('uploading');
        DOM.voiceFileInput.value = '';  // allow re-selecting same file
    }
}

function setupSettingsSliders() {
    $$('.setting-item input[type="range"]').forEach(slider => {
        const display = $(`.setting-value[data-for="${slider.id}"]`);
        if (display) {
            slider.addEventListener('input', () => display.textContent = slider.value);
        }
    });
}

// ═══════════════════════════════════════════════════════════
//  Text Splitting
// ═══════════════════════════════════════════════════════════

function updateCharCount() {
    const len = DOM.editor.value.length;
    DOM.charCount.textContent = `${len.toLocaleString()} chars`;
}

async function splitText() {
    const text = DOM.editor.value.trim();
    if (!text) {
        toast('Please enter some text first', 'error');
        return;
    }

    DOM.splitBtn.disabled = true;
    DOM.splitBtn.querySelector('.btn-icon').textContent = '⏳';

    try {
        const res = await fetch(`${API}/api/split`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();

        // Store session ID for this generation session
        state.session_id = data.session_id || null;

        // Add per-sentence config defaults from global settings
        const defaults = getSettings();
        state.sentences = data.sentences.map(s => ({
            ...s,
            config: {
                exaggeration: defaults.exaggeration,
                cfg_weight: defaults.cfg_weight,
                temperature: defaults.temperature,
                speed: 1.0,
                voice: DOM.voiceSelect.value,
            },
            showConfig: false,
        }));
        renderSentences();
        updateControls();
        toast(`Split into ${data.sentences.length} sentences`, 'success');
    } catch (e) {
        toast('Failed to split text', 'error');
    } finally {
        DOM.splitBtn.disabled = false;
        DOM.splitBtn.querySelector('.btn-icon').textContent = '✂️';
    }
}

async function smartSplitText() {
    const text = DOM.editor.value.trim();
    if (!text) {
        toast('Please enter some text first', 'error');
        return;
    }

    DOM.smartSplitBtn.disabled = true;
    DOM.splitBtn.disabled = true;
    const originalHtml = DOM.smartSplitBtn.innerHTML;
    DOM.smartSplitBtn.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px"></div> Analyzing...';

    try {
        const res = await fetch(`${API}/api/smart-split`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                language: DOM.languageSelect.value,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error ${res.status}`);
        }

        const data = await res.json();

        if (data.llm_used) {
            // Store session ID
            state.session_id = data.session_id || null;

            // LLM returned per-sentence configs — use them directly
            state.sentences = data.sentences.map(s => ({
                ...s,
                config: s.config || {
                    exaggeration: 0.5,
                    cfg_weight: 0.5,
                    temperature: 0.8,
                    speed: 1.0,
                    voice: DOM.voiceSelect.value,
                },
                showConfig: true, // Auto-expand to show AI suggestions
            }));
            // Add voice to each config
            state.sentences.forEach(s => {
                if (s.config && !s.config.voice) s.config.voice = DOM.voiceSelect.value;
            });
            const cacheNote = data.cached ? ' (from cache ⚡)' : '';
            toast(`✨ AI split into ${data.sentences.length} sentences with tuned configs${cacheNote}`, 'success');
        } else {
            // Fallback: no LLM configs
            state.session_id = data.session_id || null;
            const defaults = getSettings();
            state.sentences = data.sentences.map(s => ({
                ...s,
                config: {
                    exaggeration: defaults.exaggeration,
                    cfg_weight: defaults.cfg_weight,
                    temperature: defaults.temperature,
                    speed: 1.0,
                    voice: DOM.voiceSelect.value,
                },
                showConfig: false,
            }));
            toast('⚠️ AI unavailable, used regex split (no per-sentence tuning)', 'error');
        }

        renderSentences();
        updateControls();
    } catch (e) {
        toast('Smart split failed: ' + e.message, 'error');
    } finally {
        DOM.smartSplitBtn.disabled = false;
        DOM.splitBtn.disabled = false;
        DOM.smartSplitBtn.innerHTML = originalHtml;
    }
}

function clearAll() {
    stopPlayback();
    DOM.editor.value = '';
    state.sentences = [];
    state.session_id = null;
    renderSentences();
    updateControls();
    updateCharCount();
}

// ═══════════════════════════════════════════════════════════
//  Sentence Rendering
// ═══════════════════════════════════════════════════════════

function renderSentences() {
    if (state.sentences.length === 0) {
        DOM.sentencesList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📖</div>
                <p>Paste your text in the editor and click<br>"Split into Sentences" to get started</p>
            </div>`;
        return;
    }

    const cfg = (s) => s.config || { exaggeration: 0.5, cfg_weight: 0.5, temperature: 0.8, speed: 1.0, voice: '' };

    DOM.sentencesList.innerHTML = state.sentences.map((s, i) => `
        <div class="sentence-row ${s.status}" data-id="${s.id}" id="sentence-${s.id}">
            <div class="sentence-num">${i + 1}</div>
            <div class="sentence-content">
                <div class="sentence-text" contenteditable="true" data-id="${s.id}"
                     spellcheck="false">${escapeHtml(s.text)}</div>
                ${s.status === 'done' ? `
                    <div class="sentence-waveform" id="waveform-${s.id}">
                        <canvas id="canvas-${s.id}"></canvas>
                    </div>
                ` : s.status === 'generating' ? `
                    <div class="sentence-waveform" style="display:flex;align-items:center;justify-content:center;">
                        <div class="spinner"></div>
                        <span style="margin-left:8px;font-size:0.75rem;color:var(--text-muted)">Generating...</span>
                    </div>
                ` : ''}
                <div class="sentence-meta">
                    <div class="status-dot ${s.status}"></div>
                    ${s.duration > 0 ? `<span class="sentence-duration">${s.duration.toFixed(1)}s</span>` : ''}
                    <div class="sentence-pause">
                        <span style="font-size:0.7rem;color:var(--text-muted)">pause:</span>
                        <input type="number" class="pause-input" value="${s.pause_after_ms}" 
                               min="0" max="5000" step="100" data-id="${s.id}">
                        <span style="font-size:0.65rem;color:var(--text-muted)">ms</span>
                    </div>
                </div>
                <!-- Per-sentence config panel -->
                <div class="sentence-config ${s.showConfig ? 'open' : ''}" data-id="${s.id}">
                    <div class="sc-row sc-voice-row">
                        <label>Voice</label>
                        <select class="sc-voice-select" data-id="${s.id}">
                            ${voiceList.map(v => `<option value="${v.path}" ${v.path === (cfg(s).voice || DOM.voiceSelect.value) ? 'selected' : ''}>${v.name}</option>`).join('')}
                        </select>
                        <span class="sc-val sc-voice-badge">${cfg(s).voice && cfg(s).voice !== DOM.voiceSelect.value ? '✦' : ''}</span>
                    </div>
                    <div class="sc-row">
                        <label>Exagg</label>
                        <input type="range" class="sc-slider" data-id="${s.id}" data-key="exaggeration"
                               min="0" max="2" step="0.05" value="${cfg(s).exaggeration}">
                        <span class="sc-val">${cfg(s).exaggeration}</span>
                    </div>
                    <div class="sc-row">
                        <label>CFG</label>
                        <input type="range" class="sc-slider" data-id="${s.id}" data-key="cfg_weight"
                               min="0" max="1" step="0.05" value="${cfg(s).cfg_weight}">
                        <span class="sc-val">${cfg(s).cfg_weight}</span>
                    </div>
                    <div class="sc-row">
                        <label>Temp</label>
                        <input type="range" class="sc-slider" data-id="${s.id}" data-key="temperature"
                               min="0.1" max="1" step="0.05" value="${cfg(s).temperature}">
                        <span class="sc-val">${cfg(s).temperature}</span>
                    </div>
                    <div class="sc-row">
                        <label>Speed</label>
                        <input type="range" class="sc-slider" data-id="${s.id}" data-key="speed"
                               min="0.95" max="1.05" step="0.01" value="${cfg(s).speed || 1.0}">
                        <span class="sc-val">${cfg(s).speed || 1.0}</span>
                    </div>
                </div>
            </div>
            <div class="sentence-controls">
                <button class="sentence-play-btn" data-id="${s.id}" 
                        ${s.status !== 'done' ? 'disabled' : ''} title="Play">
                    <svg viewBox="0 0 24 24"><polygon points="6,3 20,12 6,21" fill="currentColor"/></svg>
                </button>
                <button class="sentence-regen-btn" data-id="${s.id}" title="Regenerate">🔄</button>
                <button class="sentence-config-btn" data-id="${s.id}" title="Settings"
                        style="opacity:${s.showConfig ? '1' : '0.5'}">⚙️</button>
            </div>
        </div>
    `).join('');

    // Attach event listeners
    DOM.sentencesList.querySelectorAll('.sentence-play-btn').forEach(btn => {
        btn.addEventListener('click', () => playSentence(btn.dataset.id));
    });

    DOM.sentencesList.querySelectorAll('.sentence-regen-btn').forEach(btn => {
        btn.addEventListener('click', () => generateSentence(btn.dataset.id));
    });

    // Per-sentence config toggle
    DOM.sentencesList.querySelectorAll('.sentence-config-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const s = state.sentences.find(s => s.id === btn.dataset.id);
            if (s) {
                s.showConfig = !s.showConfig;
                const panel = DOM.sentencesList.querySelector(`.sentence-config[data-id="${s.id}"]`);
                if (panel) panel.classList.toggle('open', s.showConfig);
                btn.style.opacity = s.showConfig ? '1' : '0.5';
            }
        });
    });

    // Per-sentence config sliders
    DOM.sentencesList.querySelectorAll('.sc-slider').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const s = state.sentences.find(s => s.id === slider.dataset.id);
            if (s && s.config) {
                s.config[slider.dataset.key] = parseFloat(slider.value);
                // Update the value display next to the slider
                const valSpan = slider.nextElementSibling;
                if (valSpan) valSpan.textContent = slider.value;
            }
        });
    });

    // Per-sentence voice select
    DOM.sentencesList.querySelectorAll('.sc-voice-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            const s = state.sentences.find(s => s.id === sel.dataset.id);
            if (s && s.config) {
                s.config.voice = sel.value;
                // Update the badge indicator
                const badge = sel.closest('.sc-voice-row')?.querySelector('.sc-voice-badge');
                if (badge) badge.textContent = sel.value !== DOM.voiceSelect.value ? '✦' : '';
            }
        });
    });

    DOM.sentencesList.querySelectorAll('.pause-input').forEach(input => {
        input.addEventListener('change', (e) => {
            const s = state.sentences.find(s => s.id === input.dataset.id);
            if (s) s.pause_after_ms = parseInt(e.target.value) || 500;
        });
    });

    DOM.sentencesList.querySelectorAll('.sentence-text[contenteditable]').forEach(el => {
        el.addEventListener('blur', (e) => {
            const s = state.sentences.find(s => s.id === el.dataset.id);
            if (s) {
                const newText = el.textContent.trim();
                if (newText !== s.text) {
                    s.text = newText;
                    s.status = 'pending';
                    s.audio_file = null;
                    s.duration = 0;
                    renderSentences();
                }
            }
        });
    });

    // Draw waveforms for completed sentences
    state.sentences.filter(s => s.status === 'done').forEach(s => {
        drawWaveform(s.id);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════
//  Audio Generation
// ═══════════════════════════════════════════════════════════

async function generateSentence(id) {
    const sentence = state.sentences.find(s => s.id === id);
    if (!sentence) return;

    sentence.status = 'generating';
    // Update only this row — don't re-render the whole list (preserves playback)
    updateSentenceRow(sentence);

    try {
        // Use per-sentence config, falling back to global settings
        const globalSettings = getSettings();
        const sc = sentence.config || {};
        const res = await fetch(`${API}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sentence_id: sentence.id,
                sentence_index: state.sentences.indexOf(sentence),
                text: sentence.text,
                language: DOM.languageSelect.value,
                voice: sc.voice || DOM.voiceSelect.value,
                exaggeration: sc.exaggeration ?? globalSettings.exaggeration,
                cfg_weight: sc.cfg_weight ?? globalSettings.cfg_weight,
                temperature: sc.temperature ?? globalSettings.temperature,
                speed: sc.speed ?? 1.0,
                session_id: state.session_id,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Generation failed');
        }

        const data = await res.json();
        sentence.status = 'done';
        sentence.audio_file = data.audio_file;
        sentence.duration = data.duration;
        sentence._audioVersion = (sentence._audioVersion || 0) + 1;
    } catch (e) {
        sentence.status = 'error';
        toast(`Error: ${e.message}`, 'error');
    }

    // Update only this row — preserves playback state of other sentences
    updateSentenceRow(sentence);
    updateControls();
}

async function generateAll() {
    if (state.isGenerating) return;
    state.isGenerating = true;
    DOM.generateAllBtn.disabled = true;

    const pending = state.sentences.filter(s => s.status !== 'done');
    if (pending.length === 0) {
        state.isGenerating = false;
        updateControls();
        return;
    }

    // Mark all pending as generating and show spinners
    pending.forEach(s => s.status = 'generating');
    renderSentences();

    const totalCount = pending.length;
    let doneCount = 0;
    const updateBtn = () => {
        DOM.generateAllBtn.innerHTML = `<div class="spinner" style="width:14px;height:14px;border-width:2px"></div> ${doneCount}/${totalCount}`;
    };
    updateBtn();

    // Concurrency-limited parallel: keep 2 requests in-flight so the GPU
    // pipeline stays full (one processing + one waiting), while leaving
    // plenty of browser connections free for audio playback & waveforms.
    const MAX_CONCURRENT = 2;
    const globalSettings = getSettings();
    let nextIdx = 0;

    const processOne = async () => {
        while (nextIdx < pending.length) {
            const idx = nextIdx++;
            const sentence = pending[idx];
            const sc = sentence.config || {};
            try {
                const res = await fetch(`${API}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sentence_id: sentence.id,
                        sentence_index: state.sentences.indexOf(sentence),
                        text: sentence.text,
                        language: DOM.languageSelect.value,
                        voice: sc.voice || DOM.voiceSelect.value,
                        exaggeration: sc.exaggeration ?? globalSettings.exaggeration,
                        cfg_weight: sc.cfg_weight ?? globalSettings.cfg_weight,
                        temperature: sc.temperature ?? globalSettings.temperature,
                        speed: sc.speed ?? 1.0,
                        session_id: state.session_id,
                    }),
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `Server error ${res.status}`);
                }

                const data = await res.json();
                sentence.status = 'done';
                sentence.audio_file = data.audio_file;
                sentence.duration = data.duration;
                sentence._audioVersion = (sentence._audioVersion || 0) + 1;
            } catch (e) {
                sentence.status = 'error';
                toast(`Error generating "${sentence.text.slice(0, 30)}...": ${e.message}`, 'error');
            }
            doneCount++;
            updateBtn();
            updateSentenceRow(sentence);
            updateControls();
        }
    };

    // Launch up to MAX_CONCURRENT workers
    const workers = Array.from(
        { length: Math.min(MAX_CONCURRENT, pending.length) },
        () => processOne()
    );
    await Promise.all(workers);

    state.isGenerating = false;
    DOM.generateAllBtn.innerHTML = '<span class="btn-icon">⚡</span> Generate All';
    updateControls();

    const successCount = pending.filter(s => s.status === 'done').length;
    if (successCount > 0) {
        toast(`Generated ${successCount} sentences`, 'success');
    }
}

/**
 * Update a single sentence row in-place without re-rendering the whole list.
 * Swaps the spinner/waveform area, updates status dot, enables play button.
 */
function updateSentenceRow(sentence) {
    const row = document.getElementById(`sentence-${sentence.id}`);
    if (!row) return;

    // Preserve playing class if this sentence is currently playing
    const isPlaying = state.currentlyPlaying === sentence.id;
    row.className = `sentence-row ${sentence.status}${isPlaying ? ' playing' : ''}`;

    // Update status dot
    const dot = row.querySelector('.status-dot');
    if (dot) dot.className = `status-dot ${sentence.status}`;

    // Update waveform / spinner area
    const content = row.querySelector('.sentence-content');
    const existingWaveform = row.querySelector('.sentence-waveform');

    if (sentence.status === 'generating') {
        // Show spinner
        if (existingWaveform) {
            existingWaveform.innerHTML = `
                <div class="spinner"></div>
                <span style="margin-left:8px;font-size:0.75rem;color:var(--text-muted)">Generating...</span>`;
            existingWaveform.style.display = 'flex';
            existingWaveform.style.alignItems = 'center';
            existingWaveform.style.justifyContent = 'center';
        } else {
            const textEl = content.querySelector('.sentence-text');
            if (textEl) {
                const wfDiv = document.createElement('div');
                wfDiv.className = 'sentence-waveform';
                wfDiv.style.display = 'flex';
                wfDiv.style.alignItems = 'center';
                wfDiv.style.justifyContent = 'center';
                wfDiv.innerHTML = `
                    <div class="spinner"></div>
                    <span style="margin-left:8px;font-size:0.75rem;color:var(--text-muted)">Generating...</span>`;
                textEl.after(wfDiv);
            }
        }

        // Disable play button while generating
        const playBtn = row.querySelector('.sentence-play-btn');
        if (playBtn) playBtn.disabled = true;

        // Remove duration display
        const durSpan = row.querySelector('.sentence-duration');
        if (durSpan) durSpan.remove();

    } else if (sentence.status === 'done') {
        // Replace spinner with waveform
        if (existingWaveform) {
            existingWaveform.innerHTML = `<canvas id="canvas-${sentence.id}"></canvas>`;
            existingWaveform.style = '';  // Clear inline flex styles from spinner
        } else {
            // Insert waveform after .sentence-text
            const textEl = content.querySelector('.sentence-text');
            if (textEl) {
                const wfDiv = document.createElement('div');
                wfDiv.className = 'sentence-waveform';
                wfDiv.id = `waveform-${sentence.id}`;
                wfDiv.innerHTML = `<canvas id="canvas-${sentence.id}"></canvas>`;
                textEl.after(wfDiv);
            }
        }

        // Draw the waveform
        drawWaveform(sentence.id);

        // Update duration display
        const meta = row.querySelector('.sentence-meta');
        if (meta) {
            let durSpan = meta.querySelector('.sentence-duration');
            if (!durSpan) {
                durSpan = document.createElement('span');
                durSpan.className = 'sentence-duration';
                const dotEl = meta.querySelector('.status-dot');
                if (dotEl) dotEl.after(durSpan);
            }
            durSpan.textContent = `${sentence.duration.toFixed(1)}s`;
        }

        // Enable play button
        const playBtn = row.querySelector('.sentence-play-btn');
        if (playBtn) playBtn.disabled = false;

    } else if (sentence.status === 'error') {
        // Remove spinner if present
        if (existingWaveform) existingWaveform.remove();

        // Disable play button
        const playBtn = row.querySelector('.sentence-play-btn');
        if (playBtn) playBtn.disabled = true;
    }
}

function getSettings() {
    return {
        exaggeration: parseFloat($('#setting-exaggeration').value),
        cfg_weight: parseFloat($('#setting-cfg').value),
        temperature: parseFloat($('#setting-temperature').value),
        crossfade_ms: parseInt($('#setting-crossfade').value),
    };
}

// ═══════════════════════════════════════════════════════════
//  Waveform Drawing
// ═══════════════════════════════════════════════════════════

async function drawWaveform(id) {
    const sentence = state.sentences.find(s => s.id === id);
    if (!sentence || !sentence.audio_file) return;

    const canvas = document.getElementById(`canvas-${id}`);
    if (!canvas) return;

    try {
        const res = await fetch(audioUrl(sentence.audio_file, sentence._audioVersion));
        const buffer = await res.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioCtx.decodeAudioData(buffer);
        const rawData = decoded.getChannelData(0);

        // Downsample for drawing
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * 2;  // Retina
        canvas.height = rect.height * 2;
        const ctx = canvas.getContext('2d');

        const samples = canvas.width;
        const blockSize = Math.floor(rawData.length / samples);
        const filteredData = [];
        for (let i = 0; i < samples; i++) {
            let sum = 0;
            for (let j = 0; j < blockSize; j++) {
                sum += Math.abs(rawData[i * blockSize + j] || 0);
            }
            filteredData.push(sum / blockSize);
        }

        const multiplier = canvas.height / 2 / Math.max(...filteredData, 0.01);

        // Draw waveform
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(1, '#a78bfa');

        ctx.fillStyle = gradient;
        for (let i = 0; i < filteredData.length; i++) {
            const barHeight = filteredData[i] * multiplier;
            const y = (canvas.height - barHeight) / 2;
            ctx.fillRect(i, y, 1, barHeight);
        }

        audioCtx.close();
    } catch (e) {
        console.warn('Waveform draw failed:', e);
    }
}

// ═══════════════════════════════════════════════════════════
//  Voice Preview
// ═══════════════════════════════════════════════════════════

let voicePreviewAudio = null;

function previewVoice() {
    // If already playing, stop it
    if (voicePreviewAudio) {
        voicePreviewAudio.pause();
        voicePreviewAudio = null;
        DOM.voicePreviewBtn.classList.remove('previewing');
        DOM.voicePreviewBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg>';
        return;
    }

    const voicePath = DOM.voiceSelect.value;
    if (!voicePath) {
        toast('No voice selected', 'error');
        return;
    }

    // Extract just the filename from the path (e.g. "wavs/voice0.mp3" → "voice0.mp3")
    const filename = voicePath.split('/').pop().split('\\').pop();
    voicePreviewAudio = new Audio(`${API}/api/voice-audio/${filename}`);
    DOM.voicePreviewBtn.classList.add('previewing');
    DOM.voicePreviewBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><rect x="5" y="4" width="5" height="16" rx="1" fill="currentColor"/><rect x="14" y="4" width="5" height="16" rx="1" fill="currentColor"/></svg>';

    voicePreviewAudio.addEventListener('ended', () => {
        voicePreviewAudio = null;
        DOM.voicePreviewBtn.classList.remove('previewing');
        DOM.voicePreviewBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg>';
    });

    voicePreviewAudio.addEventListener('error', () => {
        toast('Could not play voice preview', 'error');
        voicePreviewAudio = null;
        DOM.voicePreviewBtn.classList.remove('previewing');
        DOM.voicePreviewBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg>';
    });

    voicePreviewAudio.play().catch(e => {
        toast('Could not play voice preview', 'error');
        voicePreviewAudio = null;
        DOM.voicePreviewBtn.classList.remove('previewing');
        DOM.voicePreviewBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg>';
    });
}

// ═══════════════════════════════════════════════════════════
//  Playback
// ═══════════════════════════════════════════════════════════

let currentAudio = null;

function playSentence(id) {
    const sentence = state.sentences.find(s => s.id === id);
    if (!sentence || !sentence.audio_file) return;

    // Stop only current audio, don't reset playAll state
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    // Remove previous playing highlights
    $$('.sentence-row.playing').forEach(el => el.classList.remove('playing'));
    state.currentlyPlaying = null;

    currentAudio = new Audio(audioUrl(sentence.audio_file, sentence._audioVersion));
    state.currentlyPlaying = id;

    // Highlight
    const row = document.getElementById(`sentence-${id}`);
    if (row) row.classList.add('playing');

    currentAudio.addEventListener('ended', () => {
        if (row) row.classList.remove('playing');
        state.currentlyPlaying = null;
        currentAudio = null;
    });

    currentAudio.play();
}

async function playAll() {
    if (state.isPlayingAll) {
        stopPlayback();
        return;
    }

    // Need at least one done sentence or an active generation to start
    const hasDone = state.sentences.some(s => s.status === 'done');
    if (!hasDone && !state.isGenerating) {
        toast('No audio to play. Generate sentences first.', 'error');
        return;
    }

    state.isPlayingAll = true;
    DOM.playAllBtn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20"><rect x="5" y="4" width="5" height="16" rx="1" fill="currentColor"/><rect x="14" y="4" width="5" height="16" rx="1" fill="currentColor"/></svg>';
    DOM.stopBtn.disabled = false;

    // Walk through ALL sentences in order, waiting for each to become
    // available if generation is still in progress.
    for (let i = 0; i < state.sentences.length; i++) {
        if (!state.isPlayingAll) break;

        const s = state.sentences[i];

        // If this sentence isn't done yet, wait for it (generation in progress)
        if (s.status !== 'done') {
            // Skip sentences that have permanently failed
            if (s.status === 'error') continue;

            // Wait for the sentence to finish generating (poll every 300ms)
            const ready = await new Promise((resolve) => {
                if (s.status === 'done') { resolve(true); return; }
                if (s.status === 'error') { resolve(false); return; }

                const check = setInterval(() => {
                    if (!state.isPlayingAll) {
                        clearInterval(check);
                        resolve(false);
                    } else if (s.status === 'done') {
                        clearInterval(check);
                        resolve(true);
                    } else if (s.status === 'error') {
                        clearInterval(check);
                        resolve(false);
                    } else if (!state.isGenerating && s.status !== 'done') {
                        // Generation stopped and this sentence never finished
                        clearInterval(check);
                        resolve(false);
                    }
                }, 300);
            });

            if (!ready) continue; // skip this sentence
        }

        if (!state.isPlayingAll) break;

        // Play this sentence
        await new Promise((resolve) => {
            currentAudio = new Audio(audioUrl(s.audio_file, s._audioVersion));
            state.currentlyPlaying = s.id;

            const row = document.getElementById(`sentence-${s.id}`);
            if (row) {
                row.classList.add('playing');
                row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            currentAudio.addEventListener('ended', () => {
                if (row) row.classList.remove('playing');
                state.currentlyPlaying = null;

                // Add pause before next sentence
                const hasMore = state.sentences.slice(i + 1).some(
                    ns => ns.status === 'done' || ns.status === 'generating'
                );
                if (hasMore && state.isPlayingAll) {
                    setTimeout(resolve, s.pause_after_ms);
                } else {
                    resolve();
                }
            });

            currentAudio.addEventListener('error', resolve);
            currentAudio.play().catch(resolve);
        });
    }

    stopPlayback();
}

function stopPlayback() {
    state.isPlayingAll = false;
    state.currentlyPlaying = null;

    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }

    // Remove playing highlights
    $$('.sentence-row.playing').forEach(el => el.classList.remove('playing'));

    // Reset play all button
    DOM.playAllBtn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg>';
    DOM.stopBtn.disabled = true;
}

// ═══════════════════════════════════════════════════════════
//  Export
// ═══════════════════════════════════════════════════════════

async function exportAudio() {
    const doneSentences = state.sentences.filter(s => s.status === 'done');
    if (doneSentences.length === 0) {
        toast('No audio to export. Generate sentences first.', 'error');
        return;
    }

    DOM.exportBtn.disabled = true;
    DOM.exportBtn.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px"></div> Exporting...';

    try {
        const settings = getSettings();
        const res = await fetch(`${API}/api/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sentences: doneSentences.map(s => ({
                    id: s.id,
                    audio_file: s.audio_file,
                    pause_after_ms: s.pause_after_ms,
                })),
                crossfade_ms: settings.crossfade_ms,
                session_id: state.session_id,
            }),
        });

        if (!res.ok) throw new Error('Export failed');

        const data = await res.json();

        const audioPath = state.session_id
            ? `${API}/api/audio/${state.session_id}/${data.audio_file}`
            : `${API}/api/audio/${data.audio_file}`;
        const link = document.createElement('a');
        link.href = audioPath;
        link.download = `viterbox_export_${Date.now()}.wav`;
        link.click();

        toast(`Exported ${data.duration.toFixed(1)}s audio`, 'success');
    } catch (e) {
        toast(`Export failed: ${e.message}`, 'error');
    } finally {
        DOM.exportBtn.disabled = false;
        DOM.exportBtn.innerHTML = '<span class="btn-icon">💾</span> Export WAV';
    }
}

// ═══════════════════════════════════════════════════════════
//  UI Helpers
// ═══════════════════════════════════════════════════════════

function updateControls() {
    const hasSentences = state.sentences.length > 0;
    const hasDone = state.sentences.some(s => s.status === 'done');
    const hasPending = state.sentences.some(s => s.status !== 'done');

    DOM.sentenceCount.textContent = `${state.sentences.length} sentences`;
    DOM.generateAllBtn.disabled = !hasPending || state.isGenerating;
    DOM.playAllBtn.disabled = !hasDone;
    DOM.exportBtn.disabled = !hasDone;
    DOM.exportSrtBtn.disabled = !hasDone;
    DOM.saveConfigBtn.disabled = !hasSentences;

    // Update total duration
    const totalDuration = state.sentences
        .filter(s => s.status === 'done')
        .reduce((sum, s) => sum + s.duration + (s.pause_after_ms / 1000), 0);
    DOM.totalTime.textContent = formatTime(totalDuration);
}

function applyGlobalPause() {
    const pauseMs = parseInt(DOM.globalPause.value);
    state.sentences.forEach(s => s.pause_after_ms = pauseMs);
    renderSentences();
    toast(`Set all pauses to ${pauseMs}ms`, 'info');
}

// ═══════════════════════════════════════════════════════════
//  Config Import / Export
// ═══════════════════════════════════════════════════════════

function exportConfig() {
    if (state.sentences.length === 0) {
        toast('No sentences to save', 'error');
        return;
    }

    const config = {
        version: 1,
        exported_at: new Date().toISOString(),
        voice: DOM.voiceSelect.value,
        language: DOM.languageSelect.value,
        global_settings: getSettings(),
        sentences: state.sentences.map(s => ({
            text: s.text,
            pause_after_ms: s.pause_after_ms,
            config: s.config || {},
        })),
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // Generate filename from first few words of text
    const preview = state.sentences[0]?.text.slice(0, 30).replace(/[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]/g, '_').replace(/_+/g, '_') || 'config';
    a.download = `viterbox_config_${preview}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast(`Saved config for ${state.sentences.length} sentences`, 'success');
}

function importConfig(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const parsed = JSON.parse(event.target.result);

            // ── Format A: Config cache (flat array of config objects) ──
            if (Array.isArray(parsed)) {
                const hasText = parsed.length > 0 && parsed[0]?.text;

                if (hasText) {
                    // Array items include text — create full sentences
                    state.sentences = parsed.filter(s => s.text).map((s, i) => ({
                        id: crypto.randomUUID ? crypto.randomUUID() : `imported-${Date.now()}-${i}`,
                        index: i,
                        text: s.text,
                        status: 'pending',
                        audio_file: null,
                        duration: 0,
                        pause_after_ms: parseInt(s.pause_after_ms) || 500,
                        config: {
                            exaggeration: parseFloat(s.exaggeration) || 0.5,
                            cfg_weight: parseFloat(s.cfg_weight) || 0.5,
                            temperature: parseFloat(s.temperature) || 0.8,
                            speed: parseFloat(s.speed) || 1.0,
                            voice: s.voice || DOM.voiceSelect.value,
                        },
                        showConfig: true,
                    }));
                    DOM.editor.value = state.sentences.map(s => s.text).join('\n');
                    updateCharCount();
                    renderSentences();
                    updateControls();
                    toast(`📂 Loaded ${state.sentences.length} sentences from config`, 'success');
                } else {
                    // Config-only array — apply onto existing sentences
                    if (state.sentences.length === 0) {
                        throw new Error('No sentences loaded. Split text first, then load config.');
                    }
                    const applyCount = Math.min(parsed.length, state.sentences.length);
                    for (let i = 0; i < applyCount; i++) {
                        const cfg = parsed[i];
                        if (!cfg || typeof cfg !== 'object') continue;
                        state.sentences[i].config = {
                            exaggeration: parseFloat(cfg.exaggeration) || 0.5,
                            cfg_weight: parseFloat(cfg.cfg_weight) || 0.5,
                            temperature: parseFloat(cfg.temperature) || 0.8,
                            speed: parseFloat(cfg.speed) || 1.0,
                            voice: cfg.voice || state.sentences[i].config?.voice || DOM.voiceSelect.value,
                        };
                        state.sentences[i].pause_after_ms = parseInt(cfg.pause_after_ms) || state.sentences[i].pause_after_ms;
                        state.sentences[i].showConfig = true;
                    }
                    renderSentences();
                    updateControls();
                    if (parsed.length !== state.sentences.length) {
                        toast(`⚠️ Applied ${applyCount} configs (file has ${parsed.length}, loaded ${state.sentences.length} sentences)`, 'info');
                    } else {
                        toast(`📂 Applied config to ${applyCount} sentences`, 'success');
                    }
                }
                return;
            }

            // ── Format B: Full export config (object with sentences + text) ──
            const config = parsed;

            // Validate structure
            if (!config.sentences || !Array.isArray(config.sentences) || config.sentences.length === 0) {
                throw new Error('Invalid config: no sentences found');
            }

            // Restore language/voice if present
            if (config.language) {
                DOM.languageSelect.value = config.language;
            }
            if (config.voice) {
                DOM.voiceSelect.value = config.voice;
            }

            // Restore global settings if present
            if (config.global_settings) {
                const gs = config.global_settings;
                if (gs.exaggeration != null) $('#setting-exaggeration').value = gs.exaggeration;
                if (gs.cfg_weight != null) $('#setting-cfg').value = gs.cfg_weight;
                if (gs.temperature != null) $('#setting-temperature').value = gs.temperature;
                if (gs.crossfade_ms != null) $('#setting-crossfade').value = gs.crossfade_ms;
                // Update displayed values
                setupSettingsSliders();
                $$('.setting-item input[type="range"]').forEach(slider => {
                    const display = $(`.setting-value[data-for="${slider.id}"]`);
                    if (display) display.textContent = slider.value;
                });
            }

            // Rebuild sentences with fresh IDs and pending status
            state.sentences = config.sentences.map((s, i) => ({
                id: crypto.randomUUID ? crypto.randomUUID() : `imported-${Date.now()}-${i}`,
                index: i,
                text: s.text,
                status: 'pending',
                audio_file: null,
                duration: 0,
                pause_after_ms: s.pause_after_ms ?? 500,
                config: s.config ? {
                    exaggeration: s.config.exaggeration ?? 0.5,
                    cfg_weight: s.config.cfg_weight ?? 0.5,
                    temperature: s.config.temperature ?? 0.8,
                    speed: s.config.speed ?? 1.0,
                    voice: s.config.voice || DOM.voiceSelect.value,
                } : {
                    exaggeration: 0.5,
                    cfg_weight: 0.5,
                    temperature: 0.8,
                    speed: 1.0,
                    voice: DOM.voiceSelect.value,
                },
                showConfig: true,
            }));

            // Populate editor with the imported text
            DOM.editor.value = state.sentences.map(s => s.text).join('\n');
            updateCharCount();

            renderSentences();
            updateControls();
            toast(`📂 Loaded ${state.sentences.length} sentences from config`, 'success');
        } catch (err) {
            toast(`Failed to load config: ${err.message}`, 'error');
        }
    };
    reader.readAsText(file);

    // Reset input so re-selecting the same file triggers change again
    DOM.configFileInput.value = '';
}

function exportSrt() {
    const doneSentences = state.sentences.filter(s => s.status === 'done');
    if (doneSentences.length === 0) {
        toast('No audio to export SRT. Generate sentences first.', 'error');
        return;
    }

    const settings = getSettings();
    const crossfadeMs = settings.crossfade_ms || 50;
    let lines = [];
    let currentTimeMs = 0;

    doneSentences.forEach((s, i) => {
        const durationMs = Math.round(s.duration * 1000);
        const startMs = currentTimeMs;
        const endMs = startMs + durationMs;

        lines.push(`${i + 1}`);
        lines.push(`${formatSrtTime(startMs)} --> ${formatSrtTime(endMs)}`);
        lines.push(s.text);
        lines.push('');

        // Advance by duration + pause (minus crossfade overlap, matching export logic)
        const pauseMs = s.pause_after_ms || 0;
        currentTimeMs = endMs + Math.max(0, pauseMs - crossfadeMs);
    });

    const srtContent = lines.join('\n');
    const blob = new Blob([srtContent], { type: 'text/srt' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `viterbox_export_${Date.now()}.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast(`Exported SRT with ${doneSentences.length} subtitles`, 'success');
}

function formatSrtTime(ms) {
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor((ms % 3600000) / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    const millis = ms % 1000;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')},${String(millis).padStart(3, '0')}`;
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    DOM.toastContainer.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(40px)';
        el.style.transition = '0.3s ease';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}
