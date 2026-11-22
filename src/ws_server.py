"""WebSocket Server nhận luồng 16kHz Mono Float32LE từ Browser Extension / Userscript."""

from __future__ import annotations

import asyncio
import json
import queue
import socket
import threading
from typing import Callable

import numpy as np
import websockets

from asr_engine import SherpaASREngine

StatusCallback = Callable[[str], None]
SubtitleCallback = Callable[[str, bool], None]
ResetCallback = Callable[[], None]


def check_port_available(host: str, port: int) -> tuple[bool, str]:
    """Kiểm tra nhanh xem có thể mở WebSocket Server tại host:port hay không."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True, ""
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()


class AudioWebSocketWorker:
    """Chạy WebSocket Server (asyncio) + luồng suy luận AI tách biệt với GUI."""

    def __init__(
        self,
        host: str,
        port: int,
        engine: SherpaASREngine,
        on_subtitle: SubtitleCallback,
        on_status: StatusCallback,
        on_reset: ResetCallback | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.engine = engine
        self.on_subtitle = on_subtitle
        self.on_status = on_status
        self.on_reset = on_reset

        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=120)
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._asr_thread: threading.Thread | None = None

    def start(self) -> None:
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()

        self._ws_thread = threading.Thread(target=self._ws_thread_main, daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass

    def _asr_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self.engine.process_audio(item, self.on_subtitle)
            except Exception as exc:
                self.on_status(f"Lỗi suy luận AI: {exc}")

    def _ws_thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_server())
        except RuntimeError:
            pass
        finally:
            self._loop.close()

    async def _run_server(self) -> None:
        async def _handler(ws: websockets.ServerConnection) -> None:
            self.on_status("Đã kết nối trình duyệt — đang nhận âm thanh...")
            self.engine.reset()
            if self.on_reset is not None:
                self.on_reset()
            try:
                async for message in ws:
                    if self._stop_event.is_set():
                        break
                    if isinstance(message, str):
                        try:
                            meta = json.loads(message)
                            sr = meta.get("sampleRate", 16000)
                            self.on_status(f"Đang nghe từ trình duyệt ({sr}Hz Mono)")
                        except json.JSONDecodeError:
                            pass
                    else:
                        samples = np.frombuffer(message, dtype=np.float32).copy()
                        if self._audio_queue.full():
                            # Xả bớt 1 gói cũ nếu hàng đợi đầy để giữ độ trễ thấp nhất
                            try:
                                self._audio_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self._audio_queue.put_nowait(samples)
            except websockets.ConnectionClosed:
                pass
            finally:
                self.engine.reset()
                if self.on_reset is not None:
                    self.on_reset()
                if not self._stop_event.is_set():
                    self.on_status(
                        f"Trình duyệt đã ngắt kết nối — đang chờ tại ws://{self.host}:{self.port}"
                    )

        async with websockets.serve(
            _handler,
            self.host,
            self.port,
            max_size=2**20,
            ping_interval=20,
            ping_timeout=20,
        ):
            self.on_status(f"Đang chờ kết nối tại ws://{self.host}:{self.port} ...")
            while not self._stop_event.is_set():
                await asyncio.sleep(0.2)
