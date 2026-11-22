"""Quản lý cấu hình và danh sách link tải model cố định cho sherpa-onnx + Local LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = APP_ROOT / "config.json"
MODELS_ROOT = APP_ROOT / "models"

SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)
SILERO_VAD_PATH = MODELS_ROOT / "silero_vad.onnx"

# Danh sách link tải cố định từ GitHub Releases chính thức của k2-fsa/sherpa-onnx
PRESET_MODELS: list[dict[str, str]] = [
    {
        "id": "nemotron-en-0.6b-560ms",
        "name": "NVIDIA Nemotron Speech Streaming EN 0.6B (560ms, int8) [Mới 2026, độ chính xác cao]",
        "model_type": "nemotron",
        "dir_name": "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
        "size_note": "~670 MB (int8), có sẵn viết hoa & dấu câu, cần sherpa-onnx mới nhất (không dùng hotwords.txt)",
    },
    {
        "id": "zipformer-en-gigaspeech",
        "name": "Streaming Zipformer EN 70M (LibriSpeech + GigaSpeech) [Nhẹ, hỗ trợ hotwords.txt]",
        "model_type": "zipformer",
        "dir_name": "sherpa-onnx-streaming-zipformer-en-2023-06-21",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        "size_note": "~180 MB (int8) / ~340 MB (fp32), hỗ trợ hotwords.txt",
    },
    {
        "id": "zipformer-en-2023-06-26",
        "name": "Streaming Zipformer EN 66M (LibriSpeech, bản int8 gọn nhẹ)",
        "model_type": "zipformer",
        "dir_name": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "size_note": "~70 MB (int8) / ~253 MB (fp32), hỗ trợ hotwords.txt",
    },
    {
        "id": "zipformer-en-20m",
        "name": "Streaming Zipformer EN 20M (Siêu nhẹ cho CPU yếu)",
        "model_type": "zipformer",
        "dir_name": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        "size_note": "~42 MB (int8), hỗ trợ hotwords.txt",
    },
    {
        "id": "moonshine-v2-tiny-en",
        "name": "Moonshine v2 Tiny EN (Quantized, tự tải kèm Silero VAD)",
        "model_type": "moonshine",
        "dir_name": "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2",
        "size_note": "Gọn nhẹ, tự ngắt câu qua Silero VAD (không dùng hotwords.txt)",
    },
    {
        "id": "moonshine-v1-tiny-en",
        "name": "Moonshine v1 Tiny EN (int8, tự tải kèm Silero VAD)",
        "model_type": "moonshine",
        "dir_name": "sherpa-onnx-moonshine-tiny-en-int8",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2",
        "size_note": "Gọn nhẹ, tự ngắt câu qua Silero VAD (không dùng hotwords.txt)",
    },
]

DEFAULT_CONFIG: dict[str, Any] = {
    "model_dir": "",
    "model_type": "nemotron",  # "nemotron", "zipformer", hoặc "moonshine"
    "prefer_int8": True,
    "num_threads": 4,
    "ws_host": "127.0.0.1",
    "ws_port": 8765,
    "hotwords_file": "",  # Chỉ áp dụng cho Zipformer; Nemotron/Moonshine tự động bỏ qua
    "hotwords_score": 2.0,
    "font_size": 22,
    # Cấu hình cửa sổ phụ đề Tiếng Việt thứ 2 (dịch qua Local LLM Gemma 3 4B)
    "enable_vi_translation": True,
    # Mặc định trỏ tới llama-server (Vulkan). Nếu dùng Ollama, đổi thành:
    # "http://127.0.0.1:11434/v1/chat/completions" và llm_model thành "gemma3:4b-it-qat"
    "llm_api_url": "http://127.0.0.1:8080/v1/chat/completions",
    "llm_model": "gemma-3-4b-it-qat",
    "llm_api_key": "",
    "llm_translate_interval": 1.0,  # Chu kỳ dịch nháp câu đang nói dở (giây); hết câu dịch ngay lập tức
    "llm_context_sentences": 2,  # Số câu tiếng Anh trước đó gửi kèm làm ngữ cảnh
    "llm_system_prompt": "",  # Để trống để dùng system prompt CS2 mặc định
}


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """Đọc file config.json; nếu chưa có hoặc lỗi thì trả về cấu hình mặc định."""
    cfg = dict(DEFAULT_CONFIG)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update(data)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Cảnh báo] Không đọc được {path.name} ({exc}). Dùng cấu hình mặc định.")
    return cfg


def save_config(cfg: dict[str, Any], path: Path = CONFIG_FILE) -> None:
    """Lưu cấu hình ra file config.json."""
    path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
