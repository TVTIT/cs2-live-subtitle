# CS2 Live Subtitles (Sherpa-ONNX + Local LLM Gemma 3 4B + PyQt6)

Ứng dụng phụ đề tiếng Anh + dịch tiếng Việt thời gian thực chạy hoàn toàn cục bộ (đa nền tảng **Linux & Windows**), nhận âm thanh từ Extension **Tab Audio Streamer** hoặc Userscript **YouTube Live Subtitles** qua WebSocket.

## Cấu trúc thư mục

- `main.py` — Khởi chạy CLI cài đặt, nạp model ASR, bật WebSocket Server, luồng dịch Local LLM và mở 2 cửa sổ phụ đề EN / VI.
- `config.py` — Đọc/ghi `config.json`, danh sách preset ASR (**NVIDIA Nemotron Speech Streaming 0.6B**, Zipformer, Moonshine) và cấu hình Local LLM (**Gemma 3 4B**).
- `model_manager.py` — Tải/giải nén và tự động nhận diện model ASR.
- `asr_engine.py` — Lớp bọc `sherpa-onnx` cho Nemotron Streaming (`feature_dim=128`, `greedy_search`), Zipformer (`feature_dim=80`, hotwords) và Moonshine + Silero VAD.
- `llm_translator.py` — Luồng dịch Anh → Việt bất đồng bộ gọi Local LLM qua API chuẩn OpenAI (`/v1/chat/completions`), kèm system prompt thuật ngữ CS2, ngữ cảnh 2 câu trước đó, gom nhịp dịch nháp và dịch ngay khi hết câu.
- `ws_server.py` — WebSocket Server nhận luồng `16kHz Mono Float32LE` từ trình duyệt trên luồng nền.
- `subtitle_overlay.py` — Cửa sổ phụ đề nổi PyQt6 (giống hệt nhau cho cả EN & VI: luôn on-top, tự cuộn không tràn mép dưới, nền đen xám 60%, nút `A-`/`A+`, ẩn taskbar `_`).
- `requirements.txt` — Thư viện Python cần thiết (`sherpa-onnx>=2.0.0`).

## Bước 1: Chạy Local LLM Gemma 3 4B trên GPU (AMD RX 6600 — Vulkan)

Chọn **1 trong 2** cách sau (đều chạy tốt trên cả Fedora Linux và Windows):

### Cách A — Dùng `llama-server` (`llama.cpp` Vulkan — khuyên dùng, nhẹ và nhanh)
```bash
llama-server -hf ggml-org/gemma-3-4b-it-qat-GGUF:Q4_0 --host 127.0.0.1 --port 8080 -c 2048 -ngl 99
```
*(Script mặc định đã trỏ sẵn về `http://127.0.0.1:8080/v1/chat/completions` nên không cần sửa `config.json`.)*

### Cách B — Dùng Ollama
```bash
ollama pull gemma3:4b-it-qat
ollama serve
```
Sau đó mở `config.json` và đổi 2 dòng sau:
```json
"llm_api_url": "http://127.0.0.1:11434/v1/chat/completions",
"llm_model": "gemma3:4b-it-qat"
```

## Bước 2: Cài đặt và chạy script Python

```bash
python3 -m venv venv
# Linux:
source venv/bin/activate
# Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

pip install -U -r requirements.txt
python main.py
```

## Tùy chỉnh nhanh trong `config.json`

- `"enable_vi_translation"`: `true` (bật cửa sổ tiếng Việt thứ 2) hoặc `false` (chỉ hiện tiếng Anh).
- `"llm_translate_interval"`: `1.0` (số giây tối thiểu giữa các lần dịch nháp câu đang nói dở; câu chốt `is_final` luôn được dịch ngay lập tức).
- `"llm_context_sentences"`: `2` (số câu tiếng Anh trước đó gửi kèm để Gemma 3 hiểu mạch trận đấu).
- `"llm_system_prompt"`: Để trống để dùng bảng luật thuật ngữ CS2 mặc định, hoặc tự viết prompt riêng.
