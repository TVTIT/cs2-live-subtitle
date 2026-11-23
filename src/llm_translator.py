"""Bộ dịch phụ đề Anh -> Việt cục bộ qua Local LLM (tối ưu cho Translate Gemma 4 Sub E4B & Gemma 4 E4B)."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Callable

TranslatedCallback = Callable[[str, bool], None]  # (vi_text, is_final)
StatusCallback = Callable[[str], None]

# Định dạng system prompt chuẩn của 17slever17/translate-gemma-4-sub-e4b
# (tương thích tốt với cả Gemma 4 E4B và Gemma 3 4B gốc)
DEFAULT_CS2_SYSTEM_PROMPT = (
    "TASK: Translate English subtitles into Vietnamese.\n"
    "RULES: Live Counter-Strike 2 (CS2) esports commentary. Keep common CS2 terms in English: "
    "AWP, eco, force buy, clutch, retake, flank, smoke, flash, molly, headshot, wallbang, ace, "
    "plant, defuse, site A, site B, mid, CT, T, Major, map.\n"
    "STYLE: friendly.\n"
    "Translate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only. "
    "Preserve meaning, tone, slang, profanity, uncertainty, repetitions and incomplete speech. "
    "Return only the final translation without labels or commentary."
)

_PREFIX_CLEAN_RE = re.compile(
    r"^(?:\[(?:CURRENT_TRANSLATION|TRANSLATION|CURRENT_SOURCE)\]\s*|"
    r"(?:bản dịch(?:\s+tiếng việt)?|dịch(?:\s+sang\s+tiếng việt)?|translation)\s*:\s*)",
    flags=re.IGNORECASE,
)


def _clean_llm_output(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    # Bỏ thẻ <think>...</think> hoặc <thought>...</thought> nếu server chưa tắt thinking
    text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", text, flags=re.DOTALL).strip()
    # Gom nhiều dòng thành 1 dòng gọn gàng cho cửa sổ phụ đề
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = " ".join(lines)
    text = _PREFIX_CLEAN_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“", "”"):
        text = text[1:-1].strip()
    return text


class LocalLLMTranslator:
    """Chạy luồng dịch bất đồng bộ qua llama-server / Ollama không chặn ASR hay GUI."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        on_translated: TranslatedCallback,
        on_status: StatusCallback,
        api_key: str = "",
        translate_interval: float = 1.0,
        context_sentences: int = 2,
        system_prompt: str = "",
        timeout_sec: float = 12.0,
    ) -> None:
        self.api_url = api_url.strip()
        self.model_name = model_name.strip() or "translate-gemma-4-sub-e4b"
        self.api_key = api_key.strip()
        self.on_translated = on_translated
        self.on_status = on_status
        self.translate_interval = max(0.3, float(translate_interval))
        self.system_prompt = system_prompt.strip() or DEFAULT_CS2_SYSTEM_PROMPT
        self.timeout_sec = max(3.0, float(timeout_sec))

        # Lưu cặp (câu tiếng Anh trước đó, bản dịch tiếng Việt trước đó)
        self._context: deque[tuple[str, str]] = deque(maxlen=max(0, int(context_sentences)))
        self._final_queue: queue.Queue[str] = queue.Queue(maxsize=32)

        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_event = threading.Event()
        self._pending_partial: str = ""
        self._last_translated_partial: str = ""
        self._last_partial_time: float = 0.0
        self._generation_id: int = 0
        self._connected_once: bool = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wakeup.set()

    def reset(self) -> None:
        """Xóa hàng đợi và ngữ cảnh cũ khi kết nối lại tab mới."""
        with self._lock:
            self._generation_id += 1
            self._pending_partial = ""
            self._last_translated_partial = ""
            self._context.clear()
        while True:
            try:
                self._final_queue.get_nowait()
            except queue.Empty:
                break

    def submit(self, text: str, is_final: bool) -> None:
        """Nhận câu tiếng Anh từ ASR: chốt câu đưa vào hàng đợi FIFO, câu nháp chỉ giữ bản mới nhất."""
        cleaned = text.strip()
        if not cleaned or self._stop_event.is_set():
            return

        if is_final:
            with self._lock:
                self._generation_id += 1
                self._pending_partial = ""
                self._last_translated_partial = ""
            if self._final_queue.full():
                try:
                    self._final_queue.get_nowait()
                except queue.Empty:
                    pass
            self._final_queue.put_nowait(cleaned)
            self._wakeup.set()
        else:
            with self._lock:
                if cleaned != self._pending_partial:
                    self._pending_partial = cleaned
                    self._wakeup.set()

    def _check_server_health(self) -> None:
        """Kiểm tra nhanh kết nối tới llama-server / Ollama lúc khởi động."""
        base = self.api_url
        if "/v1/chat/completions" in base:
            models_url = base.replace("/v1/chat/completions", "/v1/models")
        else:
            models_url = base

        req = urllib.request.Request(models_url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if 200 <= resp.status < 300:
                    self._connected_once = True
                    self.on_status(f"Local LLM sẵn sàng ({self.model_name})")
                    return
        except Exception:
            pass
        self.on_status(f"Chờ Local LLM ({self.model_name}) tại {self.api_url} ...")

    def _build_user_prompt(self, current_text: str) -> str:
        """Đóng gói user prompt theo đúng 3 thẻ [PREVIOUS_SOURCE], [PREVIOUS_TRANSLATION], [CURRENT_SOURCE]."""
        with self._lock:
            recent = list(self._context)

        if recent:
            prev_src = " ".join(en for en, _ in recent)
            prev_vi = " ".join(vi for _, vi in recent)
            return (
                f"[PREVIOUS_SOURCE]\n{prev_src}\n"
                f"[PREVIOUS_TRANSLATION]\n{prev_vi}\n"
                f"[CURRENT_SOURCE]\n{current_text}"
            )
        return f"[CURRENT_SOURCE]\n{current_text}"

    def _call_llm(self, text: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_prompt(text)},
            ],
            "temperature": 0.1,
            "max_tokens": 160,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        choices = body.get("choices") or []
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        return _clean_llm_output(str(content))

    def _worker_loop(self) -> None:
        self._check_server_health()

        while not self._stop_event.is_set():
            # 1. Ưu tiên tuyệt đối các câu đã chốt (is_final=True)
            try:
                final_en = self._final_queue.get_nowait()
            except queue.Empty:
                final_en = ""

            if final_en:
                try:
                    t0 = time.monotonic()
                    vi_text = self._call_llm(final_en)
                    dt_ms = int((time.monotonic() - t0) * 1000)
                    if vi_text:
                        self.on_translated(vi_text, True)
                        with self._lock:
                            self._context.append((final_en, vi_text))
                    self._connected_once = True
                    self.on_status(f"Local LLM ({self.model_name}) • {dt_ms} ms")
                except urllib.error.URLError as exc:
                    self.on_status(f"Không kết nối được Local LLM ({exc.reason})")
                    time.sleep(1.0)
                except Exception as exc:
                    self.on_status(f"Lỗi dịch Local LLM: {exc}")
                continue

            # 2. Xử lý câu đang nói dở (is_final=False) theo nhịp throttle
            partial_en = ""
            gen_snapshot = 0
            wait_sec = 0.2
            now = time.monotonic()

            with self._lock:
                if (
                    self._pending_partial
                    and self._pending_partial != self._last_translated_partial
                ):
                    elapsed = now - self._last_partial_time
                    if elapsed >= self.translate_interval:
                        partial_en = self._pending_partial
                        gen_snapshot = self._generation_id
                    else:
                        wait_sec = max(0.05, self.translate_interval - elapsed)

            if partial_en:
                try:
                    t0 = time.monotonic()
                    vi_text = self._call_llm(partial_en)
                    dt_ms = int((time.monotonic() - t0) * 1000)
                    with self._lock:
                        self._last_partial_time = time.monotonic()
                        self._last_translated_partial = partial_en
                        still_valid = gen_snapshot == self._generation_id
                    if vi_text and still_valid:
                        self.on_translated(vi_text, False)
                    self._connected_once = True
                    self.on_status(f"Local LLM ({self.model_name}) • {dt_ms} ms")
                except urllib.error.URLError as exc:
                    with self._lock:
                        self._last_partial_time = time.monotonic()
                    self.on_status(f"Không kết nối được Local LLM ({exc.reason})")
                    time.sleep(1.0)
                except Exception as exc:
                    with self._lock:
                        self._last_partial_time = time.monotonic()
                    self.on_status(f"Lỗi dịch Local LLM: {exc}")
                continue

            self._wakeup.wait(timeout=wait_sec)
            self._wakeup.clear()
