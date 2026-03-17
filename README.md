<div align="center">

# 🎙️ Viterbox TTS Studio

### Công cụ Chuyên nghiệp cho Text-to-Speech Tiếng Việt & Sản xuất Audiobook

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-dolly--vn%2Fviterbox-orange)](https://huggingface.co/dolly-vn/viterbox)

**Viterbox** là mô hình Text-to-Speech tiếng Việt chất lượng cao, được fine-tune từ [Chatterbox](https://github.com/resemble-ai/chatterbox) trên **3.000+ giờ** dữ liệu tiếng Việt.  
**Viterbox Studio** là giao diện sản xuất audiobook đi kèm — chia câu, tinh chỉnh từng câu, tạo audio, nghe thử và xuất file — tất cả ngay trên trình duyệt.

[Tính năng](#-tính-năng) • [Bắt đầu nhanh](#-bắt-đầu-nhanh) • [Studio](#️-viterbox-studio) • [Python API](#-python-api) • [CLI](#️-dòng-lệnh) • [Tham số](#️-tham-số)

</div>

---

## ✨ Tính năng

| | Tính năng | Mô tả |
|---|---------|-------|
| 🇻🇳 | **Tiếng Việt tự nhiên** | Phát âm chuẩn, ngữ điệu tự nhiên, hỗ trợ đầy đủ dấu thanh |
| 🎯 | **Zero-shot Voice Cloning** | Nhân bản giọng nói chỉ với 3–10 giây audio mẫu |
| 🌍 | **Đa ngôn ngữ** | Tiếng Việt + 23 ngôn ngữ khác kế thừa từ Chatterbox |
| 🎚️ | **Điều chỉnh từng câu** | Tinh chỉnh exaggeration, CFG, temperature, tốc độ & cao độ cho mỗi câu |
| 🤖 | **Smart Split (AI)** | Chia câu thông minh bằng Gemini, tự động gợi ý tham số TTS phù hợp |
| ⚡ | **Tăng tốc GPU** | Inference FP16 trên CUDA, hỗ trợ fallback CPU |
| 🎛️ | **Hậu xử lý** | Co giãn thời gian & thay đổi cao độ qua Rubber Band, cắt im lặng bằng VAD, crossfade |
| 💾 | **Quản lý phiên** | Tự động tổ chức thư mục output, lưu/tải cấu hình câu dạng JSON |
| 📄 | **Xuất SRT** | Tạo file phụ đề song song với audio cuối cùng |

---

## 📊 Kiến trúc Model

Viterbox được xây dựng trên kiến trúc **Chatterbox** của Resemble AI:

```
Văn bản ─→ [ T3 Transformer (520M) ] ─→ Speech Tokens ─→ [ S3Gen Vocoder ] ─→ Audio 24 kHz
                      ↑                                          ↑
            Text Tokenizer                            Voice Encoder (speaker embedding)
         (2.549 tokens, bao gồm                      từ audio mẫu tham chiếu
          1.845 tokens tiếng Việt)
```

| Thành phần | Vai trò | Kích thước |
|------------|---------|------------|
| **T3** | Văn bản → speech tokens (transformer dựa trên Llama) | 520M tham số |
| **S3Gen** | Tokens → dạng sóng (CFM decoder + HiFi-GAN vocoder) | ~150M tham số |
| **Voice Encoder** | Trích xuất speaker embedding từ audio mẫu | ~5M tham số |

### Dữ liệu huấn luyện

Fine-tune trên **3.000+ giờ** dữ liệu tiếng Việt chất lượng cao:

| Dataset | Mô tả | Thời lượng |
|---------|-------|------------|
| **ViVoice** | Giọng đọc đa dạng, nhiều vùng miền | ~1.000h |
| **PhoAudiobook** | Sách nói tiếng Việt chuyên nghiệp | ~1.200h |
| **Dolly-Audio** | Dữ liệu nội bộ, đa phong cách | ~800h |

---

## 🚀 Bắt đầu nhanh

### Yêu cầu hệ thống

| | Tối thiểu | Khuyến nghị |
|---|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **CUDA** | 11.8 | 12.0+ |
| **RAM** | 8 GB | 16 GB |
| **VRAM** | 6 GB | 8 GB+ |

### Chạy nhanh trên Windows

```bat
run.bat
```

Script sẽ tự động tạo virtual environment, cài đặt toàn bộ dependencies (có CUDA), và khởi chạy Studio tại **http://localhost:7861**.

Dùng `run.bat --reinstall` để cài lại toàn bộ packages, hoặc `run.bat --cpu` cho chế độ chỉ dùng CPU.

### Cài đặt thủ công

```bash
# Clone repo
git clone https://github.com/iamdinhthuan/viterbox-tts.git
cd viterbox-tts

# Tạo & kích hoạt virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Cài đặt PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Cài đặt dependencies
pip install -r requirements.txt
pip install -e .

# Khởi chạy Viterbox Studio
python studio_api.py
```

Mở **http://localhost:7861** trên trình duyệt.

---

## 🖥️ Viterbox Studio

Viterbox Studio là giao diện **sản xuất audiobook đầy đủ tính năng**, phục vụ dưới dạng web app cục bộ (FastAPI + HTML/CSS/JS).

### Quy trình làm việc

```
 ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐     ┌────────────┐
 │  Dán văn bản │ ──→ │  Chia câu /       │ ──→ │  Tạo audio & │ ──→ │  Xuất file │
 │  vào editor  │     │  Smart Split (AI)  │     │  nghe thử    │     │  WAV / SRT │
 └──────────────┘     └───────────────────┘     └──────────────┘     └────────────┘
```

### Các tính năng chính

- **Script Editor** — Dán văn bản bất kỳ độ dài; hiển thị số ký tự real-time.
- **Chia câu / Smart Split (AI)** — Chia câu bằng regex hoặc dùng Gemini AI phân đoạn thông minh, tự động tinh chỉnh exaggeration, CFG weight, temperature và tốc độ cho từng câu dựa trên nội dung và cảm xúc.
- **Sentence Timeline** — Mỗi câu hiển thị dưới dạng thẻ riêng với:
  - Chỉnh sửa nội dung trực tiếp
  - Tùy chỉnh voice, exaggeration, CFG, temperature, tốc độ riêng từng câu
  - Nút tạo / tạo lại / nghe thử
- **Tạo hàng loạt** — "Generate All" xử lý tất cả câu tuần tự (GPU inference tuần tự, hậu xử lý chạy song song trên thread pool).
- **Phát toàn bộ** — Nghe lại toàn bộ audio đã tạo với thanh tiến trình.
- **Xuất file** — Ghép tất cả câu thành một file WAV duy nhất với khoảng nghỉ và crossfade tùy chỉnh. Hỗ trợ xuất file phụ đề SRT.
- **Quản lý phiên** — Lưu/tải cấu hình câu dạng JSON để chỉnh sửa lặp lại qua nhiều phiên làm việc.
- **Quản lý giọng** — Chọn từ giọng mẫu có sẵn, nghe thử, hoặc import file `.wav` / `.mp3` của bạn.

### Danh sách API

| Phương thức | Endpoint | Mô tả |
|-------------|----------|-------|
| `GET` | `/api/voices` | Danh sách file giọng mẫu có sẵn |
| `POST` | `/api/voices/import` | Upload giọng mẫu mới |
| `POST` | `/api/split` | Chia câu bằng regex |
| `POST` | `/api/smart-split` | Chia câu bằng AI với cấu hình riêng từng câu |
| `POST` | `/api/generate` | Tạo audio cho một câu |
| `POST` | `/api/generate/batch` | Tạo audio cho nhiều câu |
| `POST` | `/api/export` | Ghép các câu thành file WAV xuất |
| `GET` | `/audio/{filename}` | Phục vụ file audio đã tạo |
| `GET` | `/voices/{filename}` | Phục vụ file giọng mẫu |

---

## 🐍 Python API

```python
from viterbox import Viterbox

# Tải model (tự động download từ HuggingFace lần đầu)
tts = Viterbox.from_pretrained("cuda")

# Tạo audio cơ bản (giọng mặc định)
audio = tts.generate("Xin chào, tôi là Viterbox!")
tts.save_audio(audio, "hello.wav")

# Voice cloning với tham số tùy chỉnh
audio = tts.generate(
    text="Tôi có thể nói bằng giọng của bạn!",
    language="vi",
    audio_prompt="reference.wav",   # 3-10 giây audio sạch
    exaggeration=0.3,               # giọng trầm tĩnh, tự nhiên
    cfg_weight=0.7,                 # bám sát giọng mẫu
    temperature=0.8,
)
tts.save_audio(audio, "cloned.wav")

# Xử lý văn bản dài với chia câu tự động
text = """
Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương.
Đất nước có hình chữ S với chiều dài hơn 1600 km.
Thủ đô Hà Nội là trung tâm chính trị và văn hóa của cả nước.
"""

audio = tts.generate(
    text=text,
    language="vi",
    sentence_pause_ms=600,
    crossfade_ms=50,
)
tts.save_audio(audio, "vietnam.wav")
```

---

## ⌨️ Dòng lệnh

```bash
# Tạo audio đơn giản
python inference.py --text "Xin chào ạ, em là trợ lý ảo của bạn." --output output.wav

# Với voice cloning và tùy chỉnh tham số
python inference.py \
    --text "Việt Nam là một đất nước xinh đẹp." \
    --lang vi \
    --ref wavs/voice1.wav \
    --exaggeration 0.5 \
    --cfg-weight 0.5 \
    --temperature 0.8 \
    --sentence-pause 0.5 \
    --output output.wav
```

---

## 🎛️ Tham số

### Tham số tạo giọng nói

| Tham số | Mô tả | Phạm vi | Mặc định |
|---------|-------|---------|----------|
| `text` | Văn bản cần đọc | — | *(bắt buộc)* |
| `language` | Mã ngôn ngữ | `"vi"`, `"en"` | `"vi"` |
| `audio_prompt` | Audio mẫu cho voice cloning | đường dẫn file | `None` |
| `exaggeration` | Mức biểu cảm — cao hơn = biểu cảm hơn | 0.0 – 2.0 | 0.5 |
| `cfg_weight` | Độ bám giọng — cao hơn = giống giọng mẫu hơn | 0.0 – 1.0 | 0.5 |
| `temperature` | Độ ngẫu nhiên — cao hơn = đa dạng hơn | 0.1 – 1.0 | 0.8 |
| `top_p` | Ngưỡng nucleus sampling | 0.0 – 1.0 | 1.0 |
| `repetition_penalty` | Phạt token lặp lại | 1.0 – 2.0 | 2.0 |

### Tham số hậu xử lý

| Tham số | Mô tả | Phạm vi | Mặc định |
|---------|-------|---------|----------|
| `speed` | Tỷ lệ co giãn thời gian qua Rubber Band | 0.5 – 2.0 | 1.0 |
| `sentence_pause_ms` | Khoảng im lặng giữa các câu (ms) | 0 – 2000 | 500 |
| `crossfade_ms` | Crossfade khi ghép câu (ms) | 0 – 200 | 50 |

### Hướng dẫn tinh chỉnh

- **`exaggeration`** — Tăng cho lối đọc kịch tính (đối thoại, câu cảm thán). Giảm cho giọng đọc bình tĩnh, khách quan.
- **`cfg_weight`** — Tăng để giọng tạo ra giống giọng mẫu hơn. Giảm để âm thanh tự nhiên hơn theo phong cách model.
- **`temperature`** — Tăng để có sự đa dạng giữa các lần tạo. Giảm để kết quả ổn định, nhất quán.
- **`speed`** — Hậu xử lý qua Rubber Band (co giãn thời gian giữ nguyên cao độ). Dưới 1.0 = chậm lại, trên 1.0 = nhanh hơn.

---


## 🔧 File Model

Trọng số model được lưu trữ trên HuggingFace Hub: [`dolly-vn/viterbox`](https://huggingface.co/dolly-vn/viterbox)

Chúng sẽ được **tự động tải về** khi chạy lần đầu vào thư mục `pretrained/`.

| File | Mô tả | Kích thước |
|------|-------|------------|
| `t3_ml24ls_v2.safetensors` | T3 transformer (đã fine-tune) | ~2 GB |
| `s3gen.pt` | Vocoder S3Gen | ~1 GB |
| `ve.pt` | Voice Encoder | ~20 MB |
| `tokenizer_vi_expanded.json` | Tokenizer với bộ từ vựng tiếng Việt | ~50 KB |
| `conds.pt` | Conditioning giọng mặc định | ~1 MB |

---

## ⚠️ Lưu ý & Hạn chế

- **Audio mẫu** nên sạch (không nhiễu nền), mono, dài 3–10 giây để đạt kết quả tốt nhất.
- **VRAM**: Cần khoảng ~6 GB. Nếu GPU không đủ bộ nhớ, dùng chế độ `--cpu` (chậm hơn).
- **Dấu thanh**: Văn bản có dấu đầy đủ sẽ cho phát âm tốt nhất.
- **Streaming**: Chưa hỗ trợ inference streaming thời gian thực.
- **Văn bản dài**: Với văn bản rất dài (>500 từ), nên dùng quy trình chia câu từng câu của Studio để đảm bảo chất lượng.

---

## 🔒 Sử dụng có trách nhiệm

### Mục đích sử dụng

- 📖 Sản xuất audiobook & podcast
- 🎓 Tạo nội dung e-learning
- ♿ Công cụ hỗ trợ người khiếm thị
- 🤖 Trợ lý ảo & chatbot
- 🔬 Nghiên cứu & phát triển TTS

### Nghiêm cấm

- ❌ **KHÔNG** tạo deepfake hoặc nội dung lừa đảo
- ❌ **KHÔNG** nhân bản giọng nói mà không có sự đồng ý của chủ sở hữu
- ❌ **KHÔNG** tạo nội dung vi phạm bản quyền hoặc pháp luật

---

## 📄 Giấy phép

**CC BY-NC 4.0** — Creative Commons Attribution-NonCommercial 4.0 International

- ✅ Miễn phí cho mục đích **phi thương mại**
- ✅ Chia sẻ & chỉnh sửa với ghi nguồn
- ❌ **Cấm sử dụng thương mại** khi chưa có giấy phép riêng

Liên hệ cấp phép thương mại: **[contextbox.ai](https://contextbox.ai)**

---

## 📚 Trích dẫn

```bibtex
@misc{viterbox2025,
  author       = {Dolly VN, ContextBoxAI},
  title        = {Viterbox: Vietnamese Text-to-Speech with Voice Cloning},
  year         = {2025},
  publisher    = {HuggingFace},
  url          = {https://huggingface.co/dolly-vn/viterbox}
}
```

---

<div align="center">

**Made with ❤️ by [Dolly VN](https://github.com/dolly-vn) @ [ContextBoxAI](https://contextbox.ai)**

[⬆ Về đầu trang](#️-viterbox-tts-studio)

</div>
