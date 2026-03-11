"""
Viterbox Studio - Audiobook Production API
FastAPI backend for the TTS Studio interface.
"""
import os
import sys
import json
import hashlib
import re as _re
import subprocess
import uuid
import time
import asyncio
import concurrent.futures
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from viterbox import Viterbox
from viterbox.tts import (
    _split_text_to_sentences,
    normalize_text,
    crossfade_concat,
    vad_trim,
    apply_fade_out,
    apply_fade_in,
)

# ── Rubber Band Time Stretching ─────────────────────────────────────
# pyrubberband shells out to 'rubberband' CLI — point it to our local binary
import pyrubberband as pyrb
import pyrubberband.pyrb as _pyrb_mod
_RUBBERBAND_EXE = str(Path(__file__).parent / "rubberband_bin" / "rubberband-4.0.0-gpl-executable-windows" / "rubberband.exe")
_pyrb_mod.__RUBBERBAND_UTIL = _RUBBERBAND_EXE  # noqa: private access


# ── Config ──────────────────────────────────────────────────────────
AUDIO_DIR = Path("studio_output")
AUDIO_DIR.mkdir(exist_ok=True)


# ── Session Management ─────────────────────────────────────────────
def _slugify(text: str, max_len: int = 40) -> str:
    """Turn arbitrary text into a filesystem-safe slug."""
    # Keep alphanumeric, Vietnamese characters, and spaces
    slug = _re.sub(r'[^\w\s-]', '', text, flags=_re.UNICODE)
    slug = _re.sub(r'[\s_]+', '-', slug).strip('-')
    return slug[:max_len].rstrip('-') or 'untitled'

def _create_session(text: str) -> tuple[str, Path]:
    """Create a session folder.  Returns (session_id, session_dir)."""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    preview = _slugify(text[:60])
    session_id = f"{timestamp}_{preview}"
    session_dir = AUDIO_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir

def _session_dir(session_id: str | None) -> Path:
    """Resolve the directory for a session.  Falls back to AUDIO_DIR."""
    if session_id:
        d = AUDIO_DIR / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    return AUDIO_DIR

GEMINI_MODEL = "gemini-3.1-pro-preview"  # Model for smart split

# ── Smart Split Config Cache ───────────────────────────────────────
_CONFIG_CACHE_DIR = AUDIO_DIR / ".config_cache"
_CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _config_cache_key(text: str, language: str) -> str:
    """SHA-256 hash of text+language for cache lookup."""
    return hashlib.sha256(f"{language}:{text}".encode("utf-8")).hexdigest()

def _config_cache_get(text: str, language: str) -> list | None:
    """Return cached LLM sentences list if available, else None."""
    key = _config_cache_key(text, language)
    path = _CONFIG_CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _config_cache_put(text: str, language: str, llm_sentences: list):
    """Persist the raw LLM sentence configs to disk."""
    key = _config_cache_key(text, language)
    path = _CONFIG_CACHE_DIR / f"{key}.json"
    try:
        _CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(llm_sentences, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️ Failed to write config cache: {e}")

# ── Load Model ──────────────────────────────────────────────────────
print("=" * 50)
print("🎙️  Loading Viterbox Studio...")
print("=" * 50)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

MODEL = Viterbox.from_pretrained(DEVICE)
print("✅ Model loaded!")
print("=" * 50)

# ── Voice Conditioning Cache ────────────────────────────────────────
_LAST_VOICE_PATH: Optional[str] = None  # Track the last voice used
_LAST_EXAGGERATION: Optional[float] = None  # Track exaggeration (baked into conditioning)

# ── GPU Inference Lock ──────────────────────────────────────────────
# Serializes model access so parallel HTTP requests don't race on CUDA
_GPU_LOCK = asyncio.Lock()

# ── Thread Pool for GPU inference (dedicated, so default pool stays free for I/O)
_GPU_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

# ── Thread Pool for parallel post-processing ───────────────────────
_POST_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="postproc")

# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(title="Viterbox Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ─────────────────────────────────────────────────

class SplitRequest(BaseModel):
    text: str

class SmartSplitRequest(BaseModel):
    text: str
    language: str = "vi"

class GenerateRequest(BaseModel):
    sentence_id: str
    sentence_index: int = 0  # Ordinal position for filename ordering
    text: str
    language: str = "vi"
    voice: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    speed: float = 1.0  # 0.95-1.05, Rubber Band pitch-preserving time stretch
    session_id: Optional[str] = None

class BatchGenerateRequest(BaseModel):
    sentences: List[GenerateRequest]
    session_id: Optional[str] = None

class ExportRequest(BaseModel):
    sentences: list  # [{id, audio_file, pause_after_ms}]
    crossfade_ms: int = 50
    session_id: Optional[str] = None

# ── API Endpoints ───────────────────────────────────────────────────

@app.get("/api/voices")
async def list_voices():
    """List available voice files."""
    wav_dir = Path("wavs")
    voices = []
    if wav_dir.exists():
        for f in sorted(wav_dir.glob("*.wav")) + sorted(wav_dir.glob("*.mp3")):
            voices.append({"name": f.stem, "path": str(f)})
    return {"voices": voices}


@app.post("/api/split")
async def split_text(req: SplitRequest):
    """Split text into sentences."""
    if not req.text.strip():
        return {"sentences": [], "session_id": None}
    
    session_id, _ = _create_session(req.text)
    raw_sentences = _split_text_to_sentences(req.text)
    
    sentences = []
    for i, text in enumerate(raw_sentences):
        sentences.append({
            "id": str(uuid.uuid4())[:8],
            "index": i,
            "text": text,
            "status": "pending",  # pending | generating | done | error
            "audio_file": None,
            "duration": 0,
            "pause_after_ms": 500,
        })
    
    return {"sentences": sentences, "session_id": session_id}


@app.post("/api/smart-split")
async def smart_split_text(req: SmartSplitRequest):
    """Split text with regex first, then use Gemini to assign TTS config per sentence.
    Splitting is deterministic; the AI only tunes parameters."""
    if not req.text.strip():
        return {"sentences": [], "llm_used": False, "session_id": None}
    
    session_id, _ = _create_session(req.text)
    
    # ── Step 1: Split deterministically with regex ──────────────────
    raw_sentences = _split_text_to_sentences(req.text)
    if not raw_sentences:
        return {"sentences": [], "llm_used": False, "session_id": session_id}
    
    # ── Step 2: Check cache ──────────────────────────────────────────
    cached = _config_cache_get(req.text, req.language)
    if cached is not None and isinstance(cached, list) and len(cached) == len(raw_sentences):
        sentences = []
        for i, (text, cfg) in enumerate(zip(raw_sentences, cached)):
            sentences.append({
                "id": str(uuid.uuid4())[:8],
                "index": i,
                "text": text,
                "status": "pending",
                "audio_file": None,
                "duration": 0,
                "pause_after_ms": int(cfg.get("pause_after_ms", 500)),
                "config": {
                    "exaggeration": float(cfg.get("exaggeration", 0.5)),
                    "cfg_weight": float(cfg.get("cfg_weight", 0.5)),
                    "temperature": float(cfg.get("temperature", 0.8)),
                    "speed": float(cfg.get("speed", 1.0)),
                },
            })
        print(f"  ⚡ Config cache hit: {len(sentences)} sentences (skipped Gemini)")
        return {"sentences": sentences, "llm_used": True, "cached": True, "session_id": session_id}
    # ── Step 3: Ask AI for config (compact CSV format) ─────────────────
    # Ask Gemini to return one line per sentence: "index:e,c,t,s,p"
    # This is ~10x smaller than JSON and never truncates.
    lang_name = "Vietnamese" if req.language == "vi" else "English"
    
    try:
        import tempfile, asyncio
        GEMINI_CMD = os.path.expanduser("~") + r"\AppData\Roaming\npm\gemini.cmd"
        start_time = time.time()
        
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(raw_sentences))
        
        prompt = f"""You are a TTS config assistant. Assign speech parameters for {len(raw_sentences)} {lang_name} sentences.

Parameters (ranges):
- E=exaggeration (0.0-2.0): 0.3-0.5 calm, 0.6-0.9 dialogue, 1.0+ dramatic
- C=cfg_weight (0.0-1.0): 0.3-0.5 normal, 0.6-0.8 expressive
- T=temperature (0.1-1.0): 0.6-0.8 normal, lower=formal, higher=casual
- S=speed (0.95-1.05): 0.95-0.97 dramatic, 1.0 normal, 1.02-1.05 fast
- P=pause_after_ms (100-2000): 300-500 mid-paragraph, 700-1000 end-paragraph, 1500+ scene change

OUTPUT FORMAT: One line per sentence, exactly like this:
1:E,C,T,S,P
2:E,C,T,S,P

Example for 3 sentences:
1:0.5,0.5,0.8,1.0,500
2:0.8,0.6,0.7,0.97,1000
3:1.2,0.7,0.9,0.96,1500

RULES:
- Return EXACTLY {len(raw_sentences)} lines, numbered 1 to {len(raw_sentences)}
- NO other text, NO explanation, NO markdown — ONLY the numbered lines
- Each line has exactly 5 comma-separated numbers after the colon

Sentences:
{numbered}"""
        
        prompt_file = os.path.join(tempfile.gettempdir(), "viterbox_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        print(f"  🤖 Calling Gemini CLI (compact mode, {len(raw_sentences)} sentences)...")
        
        proc = await asyncio.create_subprocess_shell(
            f'type "{prompt_file}" | "{GEMINI_CMD}" -m {GEMINI_MODEL}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        
        elapsed = time.time() - start_time
        print(f"  🤖 Gemini CLI responded in {elapsed:.1f}s ({len(stdout_text)} chars)")
        print(f"  📤 First 300 chars: {stdout_text[:300]}")
        
        if proc.returncode != 0:
            raise ValueError(f"Gemini CLI exit code {proc.returncode}")
        
        # Parse compact format: "1:0.5,0.5,0.8,1.0,500"
        llm_configs = {}
        for line in stdout_text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Match pattern: number:values
            match = _re.match(r'^(\d+)\s*:\s*(.+)$', line)
            if not match:
                continue
            idx = int(match.group(1)) - 1  # Convert 1-indexed to 0-indexed
            parts = match.group(2).split(",")
            if len(parts) >= 5:
                try:
                    llm_configs[idx] = {
                        "exaggeration": float(parts[0].strip()),
                        "cfg_weight": float(parts[1].strip()),
                        "temperature": float(parts[2].strip()),
                        "speed": float(parts[3].strip()),
                        "pause_after_ms": int(float(parts[4].strip())),
                    }
                except (ValueError, IndexError):
                    continue
        
        if not llm_configs:
            raise ValueError("No valid config lines parsed")
        
        # Cache as list (include text so cache files are self-contained)
        config_list = [{**llm_configs.get(i, {}), "text": raw_sentences[i]} for i in range(len(raw_sentences))]
        _config_cache_put(req.text, req.language, config_list)
        
        # Merge regex-split sentences with AI configs
        sentences = []
        for i, text in enumerate(raw_sentences):
            cfg = llm_configs.get(i, {})
            sentences.append({
                "id": str(uuid.uuid4())[:8],
                "index": i,
                "text": text,
                "status": "pending",
                "audio_file": None,
                "duration": 0,
                "pause_after_ms": int(cfg.get("pause_after_ms", 500)),
                "config": {
                    "exaggeration": float(cfg.get("exaggeration", 0.5)),
                    "cfg_weight": float(cfg.get("cfg_weight", 0.5)),
                    "temperature": float(cfg.get("temperature", 0.8)),
                    "speed": float(cfg.get("speed", 1.0)),
                },
            })
        
        got = len(llm_configs)
        print(f"  ✅ Smart config: {got}/{len(raw_sentences)} AI-tuned")
        return {"sentences": sentences, "llm_used": True, "cached": False, "session_id": session_id}
    
    except FileNotFoundError:
        print("  ⚠️ Gemini CLI not found, using defaults")
    except Exception as e:
        print(f"  ⚠️ Smart config error: {e}, using defaults")
    
    # Fallback: regex-split sentences with default config
    sentences = []
    for i, text in enumerate(raw_sentences):
        sentences.append({
            "id": str(uuid.uuid4())[:8],
            "index": i,
            "text": text,
            "status": "pending",
            "audio_file": None,
            "duration": 0,
            "pause_after_ms": 500,
        })
    
    return {"sentences": sentences, "llm_used": False, "session_id": session_id}


def _resolve_voice(req_voice: Optional[str]) -> str:
    """Resolve the voice path from request or fallback to default."""
    if req_voice:
        return req_voice
    wav_dir = Path("wavs")
    voices = list(wav_dir.glob("*.wav")) + list(wav_dir.glob("*.mp3"))
    if not voices:
        raise HTTPException(400, "No voice files found in wavs/")
    for v in voices:
        if "voice0" in v.stem:
            return str(v)
    return str(voices[0])


def _gpu_inference(clean_text: str, language: str, voice_path: str,
                   exaggeration: float, cfg_weight: float,
                   temperature: float) -> torch.Tensor:
    """GPU-bound model inference only.  Must be called under _GPU_LOCK."""
    global _LAST_VOICE_PATH, _LAST_EXAGGERATION

    wav = None
    for attempt in range(2):
        try:
            # Only re-prepare conditionals if voice or exaggeration changed
            if voice_path != _LAST_VOICE_PATH or exaggeration != _LAST_EXAGGERATION or MODEL.conds is None:
                print(f"  🔊 Preparing voice conditioning for: {Path(voice_path).name} (exag={exaggeration})")
                MODEL.prepare_conditionals(voice_path, exaggeration)
                _LAST_VOICE_PATH = voice_path
                _LAST_EXAGGERATION = exaggeration

            wav = MODEL.generate(
                text=clean_text,
                language=language,
                audio_prompt=None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                split_sentences=False,
            )
            break
        except RuntimeError as cuda_err:
            if "CUDA" in str(cuda_err) and attempt == 0:
                print(f"  ⚠️ CUDA error on attempt {attempt+1}, recovering: {cuda_err}")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _LAST_VOICE_PATH = None
                _LAST_EXAGGERATION = None
                MODEL.conds = None
                print("  🔄 CUDA state reset, retrying...")
                continue
            else:
                raise

    if wav is None:
        raise RuntimeError("Generation failed after retry")

    return wav


def _postprocess(wav: torch.Tensor, sentence_id: str, sentence_index: int,
                 speed: float, text_preview: str, start_time: float,
                 output_dir: Path = AUDIO_DIR) -> dict:
    """CPU-bound post-processing: trim, time-stretch, save.  No GPU lock needed."""
    # Convert to numpy and trim
    audio_np = wav[0].cpu().numpy()
    audio_np, _ = librosa.effects.trim(audio_np, top_db=30)

    # Apply speed change via Rubber Band
    if speed != 1.0:
        clamped = max(0.95, min(1.05, speed))
        audio_np = pyrb.time_stretch(audio_np, MODEL.sr, clamped)

    # Save to file with ordinal prefix for easy tracking
    filename = f"{sentence_index + 1:03d}_{sentence_id}.wav"
    filepath = output_dir / filename
    sf.write(str(filepath), audio_np, MODEL.sr)

    duration = len(audio_np) / MODEL.sr
    gen_time = time.time() - start_time

    print(f"  ✅ Generated [{filename}] {text_preview[:40]}... ({duration:.1f}s audio in {gen_time:.1f}s)")

    return {
        "sentence_id": sentence_id,
        "audio_file": filename,
        "duration": round(duration, 2),
        "generation_time": round(gen_time, 2),
    }


@app.post("/api/generate")
async def generate_sentence(req: GenerateRequest):
    """Generate audio for a single sentence.
    GPU inference is serialized via lock; post-processing runs freely."""
    try:
        start_time = time.time()
        voice_path = _resolve_voice(req.voice)
        out_dir = _session_dir(req.session_id)

        # Pass raw text to model — normalization and punctuation cleanup
        # are handled internally by MODEL.generate() → _generate_single() → punc_norm()
        clean_text = req.text.strip()

        # GPU inference — serialized, runs in dedicated pool so default pool
        # remains free to serve audio files / other I/O during generation
        loop = asyncio.get_running_loop()
        async with _GPU_LOCK:
            wav = await loop.run_in_executor(
                _GPU_POOL, _gpu_inference, clean_text, req.language,
                voice_path, req.exaggeration, req.cfg_weight, req.temperature,
            )

        # Post-processing — runs in parallel with other requests' GPU work
        result = await loop.run_in_executor(
            _POST_POOL, _postprocess, wav, req.sentence_id,
            req.sentence_index, req.speed, req.text, start_time, out_dir,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/generate-batch")
async def generate_batch(req: BatchGenerateRequest):
    """Generate audio for multiple sentences in one request.
    GPU inference is serialized; post-processing runs in parallel."""
    results = []
    errors = []
    total_start = time.time()
    out_dir = _session_dir(req.session_id)
    print(f"\n🚀 Batch generation: {len(req.sentences)} sentences")

    # Collect post-processing futures so they run concurrently
    loop = asyncio.get_running_loop()
    post_futures = []

    for i, sentence_req in enumerate(req.sentences):
        try:
            start_time = time.time()
            voice_path = _resolve_voice(sentence_req.voice)
            clean_text = sentence_req.text.strip()

            async with _GPU_LOCK:
                wav = await loop.run_in_executor(
                    _GPU_POOL, _gpu_inference, clean_text,
                    sentence_req.language, voice_path,
                    sentence_req.exaggeration, sentence_req.cfg_weight,
                    sentence_req.temperature,
                )

            # Fire post-processing without waiting — runs in thread pool
            fut = loop.run_in_executor(
                _POST_POOL, _postprocess, wav, sentence_req.sentence_id,
                sentence_req.sentence_index, sentence_req.speed,
                sentence_req.text, start_time, out_dir,
            )
            post_futures.append((sentence_req.sentence_id, fut))

        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append({
                "sentence_id": sentence_req.sentence_id,
                "error": str(e),
            })

    # Await all post-processing
    for sid, fut in post_futures:
        try:
            result = await fut
            results.append(result)
        except Exception as e:
            errors.append({"sentence_id": sid, "error": str(e)})

    total_time = time.time() - total_start
    print(f"✅ Batch complete: {len(results)} ok, {len(errors)} errors in {total_time:.1f}s\n")

    return {
        "results": results,
        "errors": errors,
        "total_time": round(total_time, 2),
    }


@app.post("/api/export")
async def export_audio(req: ExportRequest):
    """Stitch all sentence audio files into a single export."""
    try:
        out_dir = _session_dir(req.session_id)
        audio_segments = []
        pause_durations = []
        
        for item in req.sentences:
            audio_path = out_dir / item["audio_file"]
            if not audio_path.exists():
                raise HTTPException(400, f"Audio file not found: {item['audio_file']}")
            
            audio, sr = librosa.load(str(audio_path), sr=MODEL.sr, mono=True)
            audio_segments.append(audio)
            pause_durations.append(item.get("pause_after_ms", 500))
        
        if not audio_segments:
            raise HTTPException(400, "No audio segments to export")
        
        # Stitch with pauses
        if len(audio_segments) == 1:
            merged = audio_segments[0]
        else:
            # Build manually with per-sentence pauses
            merged = audio_segments[0].copy()
            for i in range(1, len(audio_segments)):
                pause_ms = pause_durations[i - 1]
                pause_samples = int(MODEL.sr * pause_ms / 1000)
                
                if pause_samples > 0:
                    silence = np.zeros(pause_samples, dtype=merged.dtype)
                    merged = np.concatenate([merged, silence])
                
                # Apply crossfade
                fade_samples = int(MODEL.sr * req.crossfade_ms / 1000)
                if len(merged) >= fade_samples and len(audio_segments[i]) >= fade_samples:
                    fade_out = np.linspace(1.0, 0.0, fade_samples)
                    fade_in = np.linspace(0.0, 1.0, fade_samples)
                    result_end = merged[-fade_samples:] * fade_out
                    next_start = audio_segments[i][:fade_samples] * fade_in
                    crossfaded = result_end + next_start
                    merged = np.concatenate([
                        merged[:-fade_samples],
                        crossfaded,
                        audio_segments[i][fade_samples:]
                    ])
                else:
                    merged = np.concatenate([merged, audio_segments[i]])
        
        # Apply final fade-out
        merged = apply_fade_out(merged, MODEL.sr, fade_duration=0.015)
        
        # Save export into session folder
        export_name = f"export_{uuid.uuid4().hex[:8]}.wav"
        export_path = out_dir / export_name
        sf.write(str(export_path), merged, MODEL.sr)
        
        duration = len(merged) / MODEL.sr
        print(f"  📦 Exported: {out_dir.name}/{export_name} ({duration:.1f}s)")
        
        return {
            "audio_file": export_name,
            "duration": round(duration, 2),
            "session_id": req.session_id,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/voice-audio/{filename}")
async def get_voice_audio(filename: str):
    """Serve voice reference files for preview playback."""
    wav_dir = Path("wavs")
    filepath = wav_dir / filename
    if not filepath.exists():
        raise HTTPException(404, "Voice file not found")
    media_type = "audio/mpeg" if filepath.suffix == ".mp3" else "audio/wav"
    return FileResponse(str(filepath), media_type=media_type)


@app.get("/api/audio/{filename:path}")
async def get_audio(filename: str):
    """Serve generated audio files.  Supports session paths like session_id/file.wav."""
    filepath = AUDIO_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(filepath), media_type="audio/wav")


# ── Serve Studio Frontend ───────────────────────────────────────────
app.mount("/", StaticFiles(directory="studio", html=True), name="studio")


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)
