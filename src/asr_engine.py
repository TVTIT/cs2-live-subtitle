"""Bộ máy nhận diện giọng nói bọc sherpa-onnx (Nemotron Streaming, Zipformer & Moonshine + VAD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import sherpa_onnx

from model_manager import ensure_bpe_vocab, ensure_silero_vad

SubtitleCallback = Callable[[str, bool], None]  # (text, is_final)


class SherpaASREngine:
    """Giao diện thống nhất cho Nemotron Streaming, Streaming Zipformer và Moonshine (v1/v2)."""

    def __init__(
        self,
        model_info: dict[str, Any],
        num_threads: int = 4,
        hotwords_file: str = "",
        hotwords_score: float = 2.0,
    ) -> None:
        self.model_type: str = model_info["model_type"]
        self.sample_rate = 16000
        self.online_recognizer = None
        self.online_stream = None
        self.offline_recognizer = None
        self.vad = None
        self._vad_window_size = 512
        self._vad_pending = np.empty(0, dtype=np.float32)
        self._last_text = ""

        if self.model_type == "nemotron":
            if hotwords_file:
                print(
                    "[Thông báo] Model Nemotron Streaming trong sherpa-onnx hiện chỉ hỗ trợ "
                    "greedy_search — đã tự động bỏ qua hotwords.txt."
                )
            self._init_nemotron(model_info=model_info, num_threads=num_threads)
        elif self.model_type == "zipformer":
            self._init_zipformer(
                model_info=model_info,
                num_threads=num_threads,
                hotwords_file=hotwords_file,
                hotwords_score=hotwords_score,
            )
        elif self.model_type in ("moonshine_v1", "moonshine_v2"):
            if hotwords_file:
                print(
                    "[Thông báo] Model Moonshine không hỗ trợ hotwords.txt trong sherpa-onnx — "
                    "đã tự động bỏ qua hotwords."
                )
            self._init_moonshine(model_info=model_info, num_threads=num_threads)
        else:
            raise ValueError(f"Loại model không được hỗ trợ: {self.model_type}")

    def _init_nemotron(self, model_info: dict[str, Any], num_threads: int) -> None:
        kwargs: dict[str, Any] = {
            "tokens": model_info["tokens"],
            "encoder": model_info["encoder"],
            "decoder": model_info["decoder"],
            "joiner": model_info["joiner"],
            "num_threads": num_threads,
            "sample_rate": self.sample_rate,
            "feature_dim": 128,
            "decoding_method": "greedy_search",
            "enable_endpoint_detection": True,
            "rule1_min_trailing_silence": 2.0,
            "rule2_min_trailing_silence": 0.9,
            "rule3_min_utterance_length": 15.0,
            "model_type": "nemotron",
            "provider": "cpu",
        }
        try:
            self.online_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        except TypeError:
            # Dự phòng nếu phiên bản sherpa-onnx tự nhận diện model_type qua metadata ONNX
            kwargs.pop("model_type", None)
            self.online_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        self.online_stream = self.online_recognizer.create_stream()

    def _init_zipformer(
        self,
        model_info: dict[str, Any],
        num_threads: int,
        hotwords_file: str,
        hotwords_score: float,
    ) -> None:
        use_hotwords = False
        bpe_vocab = ""
        hw_path = Path(hotwords_file).expanduser() if hotwords_file else None

        if hw_path is not None:
            if hw_path.is_file():
                bpe_vocab = ensure_bpe_vocab(model_info)
                if bpe_vocab:
                    use_hotwords = True
                else:
                    print(
                        "[Cảnh báo] Thiếu bpe.vocab nên tạm tắt hotwords.txt "
                        "(hãy cài sentencepiece hoặc đặt bpe.vocab vào thư mục model)."
                    )
            else:
                print(f"[Cảnh báo] Không tìm thấy file hotwords: {hw_path}. Chạy không có hotwords.")

        kwargs: dict[str, Any] = {
            "tokens": model_info["tokens"],
            "encoder": model_info["encoder"],
            "decoder": model_info["decoder"],
            "joiner": model_info["joiner"],
            "num_threads": num_threads,
            "sample_rate": self.sample_rate,
            "feature_dim": 80,
            "enable_endpoint_detection": True,
            "rule1_min_trailing_silence": 2.0,
            "rule2_min_trailing_silence": 0.9,
            "rule3_min_utterance_length": 15.0,
            "provider": "cpu",
        }

        if use_hotwords and hw_path is not None:
            kwargs["decoding_method"] = "modified_beam_search"
            kwargs["max_active_paths"] = 4
            kwargs["modeling_unit"] = "bpe"
            kwargs["bpe_vocab"] = bpe_vocab
            kwargs["hotwords_file"] = str(hw_path)
            kwargs["hotwords_score"] = float(hotwords_score)
            print(f"[ASR] Đã bật Hotwords: {hw_path} (score={hotwords_score})")
        else:
            kwargs["decoding_method"] = "greedy_search"

        self.online_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        self.online_stream = self.online_recognizer.create_stream()

    def _init_moonshine(self, model_info: dict[str, Any], num_threads: int) -> None:
        vad_path = ensure_silero_vad()
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(vad_path)
        vad_cfg.silero_vad.threshold = 0.45
        vad_cfg.silero_vad.min_silence_duration = 0.35
        vad_cfg.silero_vad.min_speech_duration = 0.25
        vad_cfg.silero_vad.max_speech_duration = 4.0  # Cắt tối đa 4s để caster nói nhanh vẫn hiện sớm
        vad_cfg.silero_vad.window_size = self._vad_window_size
        vad_cfg.sample_rate = self.sample_rate
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=30)

        if self.model_type == "moonshine_v2":
            self.offline_recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                encoder=model_info["encoder"],
                decoder=model_info["decoder"],
                tokens=model_info["tokens"],
                num_threads=num_threads,
                provider="cpu",
            )
        else:
            self.offline_recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=model_info["preprocessor"],
                encoder=model_info["encoder"],
                uncached_decoder=model_info["uncached_decoder"],
                cached_decoder=model_info["cached_decoder"],
                tokens=model_info["tokens"],
                num_threads=num_threads,
                provider="cpu",
            )

    def reset(self) -> None:
        """Đặt lại trạng thái stream khi Extension/Userscript ngắt kết nối hoặc đổi tab."""
        self._last_text = ""
        if self.online_recognizer is not None and self.online_stream is not None:
            self.online_recognizer.reset(self.online_stream)
        if self.vad is not None:
            self.vad.reset()
            self._vad_pending = np.empty(0, dtype=np.float32)

    def process_audio(self, samples: np.ndarray, on_subtitle: SubtitleCallback) -> None:
        """Nhận mảng mẫu float32 @ 16kHz Mono và gọi callback khi có chữ mới."""
        if samples.size == 0:
            return

        if self.model_type in ("zipformer", "nemotron"):
            assert self.online_recognizer is not None and self.online_stream is not None
            self.online_stream.accept_waveform(self.sample_rate, samples)
            while self.online_recognizer.is_ready(self.online_stream):
                self.online_recognizer.decode_stream(self.online_stream)

            text = self.online_recognizer.get_result(self.online_stream).strip()
            is_endpoint = self.online_recognizer.is_endpoint(self.online_stream)

            if text and text != self._last_text:
                self._last_text = text
                on_subtitle(text, False)

            if is_endpoint:
                if text:
                    on_subtitle(text, True)
                self._last_text = ""
                self.online_recognizer.reset(self.online_stream)
        else:
            assert self.vad is not None and self.offline_recognizer is not None
            self._vad_pending = np.concatenate([self._vad_pending, samples])
            while len(self._vad_pending) >= self._vad_window_size:
                window = self._vad_pending[: self._vad_window_size]
                self._vad_pending = self._vad_pending[self._vad_window_size :]
                self.vad.accept_waveform(window)

            while not self.vad.empty():
                seg = self.vad.front
                self.vad.pop()
                stream = self.offline_recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, seg.samples)
                self.offline_recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    on_subtitle(text, True)
