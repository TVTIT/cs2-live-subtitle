"""Điểm chạy chính (CLI cấu hình -> Load sherpa-onnx -> WebSocket Server & 2 Overlay EN/VI)."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from asr_engine import SherpaASREngine
from config import load_config, save_config
from llm_translator import LocalLLMTranslator
from model_manager import resolve_model_interactive
from subtitle_overlay import SubtitleOverlay
from ws_server import AudioWebSocketWorker, check_port_available


def prompt_ws_address(cfg: dict) -> tuple[str, int]:
    """Kiểm tra mở cổng WebSocket; nếu bị chiếm thì cho người dùng nhập ip:port mới."""
    host = str(cfg.get("ws_host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(cfg.get("ws_port", 8765))

    while True:
        ok, err = check_port_available(host, port)
        if ok:
            print(f"[WebSocket] Sẵn sàng mở server tại ws://{host}:{port}")
            cfg["ws_host"] = host
            cfg["ws_port"] = port
            return host, port

        print(f"[Lỗi WebSocket] Không thể mở cổng tại {host}:{port} ({err}).")
        raw = input("Nhập địa chỉ ip:port mới (ví dụ 127.0.0.1:8766): ").strip()
        if ":" not in raw:
            print("Vui lòng nhập đúng định dạng ip:port (ví dụ 127.0.0.1:8766).")
            continue
        new_host, new_port_str = raw.rsplit(":", 1)
        if not new_port_str.isdigit() or not (1 <= int(new_port_str) <= 65535):
            print("Port phải là số nguyên từ 1 đến 65535.")
            continue
        host = new_host.strip() or "127.0.0.1"
        port = int(new_port_str)


def resolve_hotwords_path(cfg: dict, model_type: str) -> str:
    """Đọc đường dẫn file hotwords.txt từ config (tùy chọn, chỉ áp dụng cho Zipformer)."""
    if model_type != "zipformer":
        return ""
    hw = str(cfg.get("hotwords_file", "")).strip()
    if hw:
        p = Path(hw).expanduser()
        if p.is_file():
            return str(p)
        print(f"[Cảnh báo] File hotwords '{p}' chưa tồn tại — tạm chạy không có hotwords.")
    return ""


def main() -> int:
    print("============================================================")
    print("  CS2 LIVE SUBTITLES (Sherpa-ONNX + Local LLM Gemma 3 4B)   ")
    print("============================================================")

    # Bước 1: Đọc config.json và kiểm tra/tải model ASR
    cfg = load_config()
    _, model_info = resolve_model_interactive(cfg)

    # Bước 2: Kiểm tra địa chỉ WebSocket Server (nhập ip:port mới nếu cổng bị chiếm)
    host, port = prompt_ws_address(cfg)
    save_config(cfg)

    # Bước 3: Load model AI ASR trước khi chờ trình duyệt kết nối
    print(f"\n[AI] Đang nạp model ({model_info['model_type']}) vào CPU...")
    hotwords_path = resolve_hotwords_path(cfg, model_info["model_type"])
    engine = SherpaASREngine(
        model_info=model_info,
        num_threads=int(cfg.get("num_threads", 4)),
        hotwords_file=hotwords_path,
        hotwords_score=float(cfg.get("hotwords_score", 2.0)),
    )
    print("[AI] Nạp model thành công!")

    # Bước 4: Mở cửa sổ phụ đề tiếng Anh (+ cửa sổ tiếng Việt nếu bật) và khởi động các luồng nền
    app = QApplication(sys.argv)
    enable_vi = bool(cfg.get("enable_vi_translation", True))
    font_size = int(cfg.get("font_size", 22))

    closing_flag = {"done": False}
    worker_holder: dict[str, AudioWebSocketWorker | None] = {"worker": None}
    translator_holder: dict[str, LocalLLMTranslator | None] = {"translator": None}

    def _shutdown_all() -> None:
        if closing_flag["done"]:
            return
        closing_flag["done"] = True
        w = worker_holder.get("worker")
        if w is not None:
            w.stop()
        t = translator_holder.get("translator")
        if t is not None:
            t.stop()
        app.quit()

    # Cửa sổ 1: Phụ đề Tiếng Anh (nằm phía trên cửa sổ Tiếng Việt khi bật song ngữ)
    overlay_en = SubtitleOverlay(
        font_size=font_size,
        on_close_callback=_shutdown_all,
        window_title="CS2 Live Subtitles (EN)",
        initial_text="Sẵn sàng nhận phụ đề tiếng Anh...",
        bottom_offset=200 if enable_vi else 48,
    )

    overlay_vi: SubtitleOverlay | None = None
    translator: LocalLLMTranslator | None = None

    if enable_vi:
        llm_url = str(cfg.get("llm_api_url", "http://127.0.0.1:8080/v1/chat/completions"))
        llm_model = str(cfg.get("llm_model", "gemma-3-4b-it-qat"))
        overlay_vi = SubtitleOverlay(
            font_size=font_size,
            on_close_callback=_shutdown_all,
            window_title=f"CS2 Live Subtitles (VI - {llm_model})",
            initial_text="Sẵn sàng dịch phụ đề tiếng Việt (Local LLM)...",
            bottom_offset=48,
        )
        translator = LocalLLMTranslator(
            api_url=llm_url,
            model_name=llm_model,
            on_translated=lambda text, is_final: overlay_vi.subtitle_signal.emit(text, is_final),
            on_status=lambda msg: overlay_vi.status_signal.emit(msg),
            api_key=str(cfg.get("llm_api_key", "")),
            translate_interval=float(cfg.get("llm_translate_interval", 1.0)),
            context_sentences=int(cfg.get("llm_context_sentences", 2)),
            system_prompt=str(cfg.get("llm_system_prompt", "")),
        )
        translator_holder["translator"] = translator
        translator.start()
        print(f"[Dịch VI] Đã bật Local LLM ({llm_model}) tại {llm_url}")

    def _on_en_subtitle(text: str, is_final: bool) -> None:
        overlay_en.subtitle_signal.emit(text, is_final)
        if translator is not None:
            translator.submit(text, is_final)

    def _on_ws_reset() -> None:
        if translator is not None:
            translator.reset()

    worker = AudioWebSocketWorker(
        host=host,
        port=port,
        engine=engine,
        on_subtitle=_on_en_subtitle,
        on_status=lambda msg: overlay_en.status_signal.emit(msg),
        on_reset=_on_ws_reset,
    )
    worker_holder["worker"] = worker
    worker.start()

    # Cho phép nhấn Ctrl+C trong terminal để thoát sạch
    signal.signal(signal.SIGINT, lambda *_: _shutdown_all())
    keepalive_timer = QTimer()
    keepalive_timer.timeout.connect(lambda: None)
    keepalive_timer.start(250)

    print(
        f"\n[Sẵn sàng] Mở tab xem giải CS2 trên trình duyệt và bấm nút bật phụ đề "
        f"(trỏ tới ws://{host}:{port})."
    )
    overlay_en.show()
    if overlay_vi is not None:
        overlay_vi.show()

    exit_code = app.exec()
    _shutdown_all()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
