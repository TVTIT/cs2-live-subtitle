"""Kiểm tra, tải xuống, giải nén model và tự động nhận diện file ONNX."""

from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from config import (
    MODELS_ROOT,
    PRESET_MODELS,
    SILERO_VAD_PATH,
    SILERO_VAD_URL,
)


def _pick_onnx(files: list[Path], prefer_int8: bool) -> Path | None:
    if not files:
        return None
    int8_files = [f for f in files if ".int8." in f.name or "int8" in f.name.lower()]
    fp32_files = [f for f in files if f not in int8_files]
    if prefer_int8 and int8_files:
        return sorted(int8_files)[0]
    if fp32_files:
        return sorted(fp32_files)[0]
    return sorted(files)[0]


def inspect_model_dir(model_dir: Path, prefer_int8: bool = True) -> dict[str, Any] | None:
    """Tự động quét thư mục model để phát hiện Nemotron Streaming, Zipformer hoặc Moonshine (v1/v2)."""
    if not model_dir.is_dir():
        return None

    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        return None

    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        return None

    # 1. Kiểm tra Online Transducer (Nemotron Streaming 0.6B hoặc Streaming Zipformer)
    encoders = [f for f in onnx_files if "encoder" in f.name.lower()]
    decoders = [
        f
        for f in onnx_files
        if "decoder" in f.name.lower()
        and "cached" not in f.name.lower()
        and "merged" not in f.name.lower()
    ]
    joiners = [f for f in onnx_files if "joiner" in f.name.lower()]
    if encoders and decoders and joiners:
        enc = _pick_onnx(encoders, prefer_int8)
        dec = _pick_onnx(decoders, prefer_int8)
        joi = _pick_onnx(joiners, prefer_int8)

        # Phân biệt Nemotron Speech Streaming với Zipformer dựa trên tên thư mục
        if "nemotron" in model_dir.name.lower():
            return {
                "model_type": "nemotron",
                "encoder": str(enc),
                "decoder": str(dec),
                "joiner": str(joi),
                "tokens": str(tokens),
            }

        bpe_vocab = model_dir / "bpe.vocab"
        bpe_model = model_dir / "bpe.model"
        return {
            "model_type": "zipformer",
            "encoder": str(enc),
            "decoder": str(dec),
            "joiner": str(joi),
            "tokens": str(tokens),
            "bpe_vocab": str(bpe_vocab) if bpe_vocab.is_file() else "",
            "bpe_model": str(bpe_model) if bpe_model.is_file() else "",
        }

    # 2. Kiểm tra Moonshine v1 (preprocess + encode + uncached_decode + cached_decode)
    preprocessors = [f for f in onnx_files if "preprocess" in f.name.lower()]
    uncached = [f for f in onnx_files if "uncached" in f.name.lower()]
    cached = [f for f in onnx_files if "cached" in f.name.lower() and "uncached" not in f.name.lower()]
    ms_encoders = [f for f in onnx_files if "encode" in f.name.lower()]
    if preprocessors and uncached and cached and ms_encoders:
        return {
            "model_type": "moonshine_v1",
            "preprocessor": str(_pick_onnx(preprocessors, prefer_int8)),
            "encoder": str(_pick_onnx(ms_encoders, prefer_int8)),
            "uncached_decoder": str(_pick_onnx(uncached, prefer_int8)),
            "cached_decoder": str(_pick_onnx(cached, prefer_int8)),
            "tokens": str(tokens),
        }

    # 3. Kiểm tra Moonshine v2 (encoder_model + decoder_model_merged)
    merged_decoders = [f for f in onnx_files if "decoder" in f.name.lower()]
    if ms_encoders and merged_decoders:
        return {
            "model_type": "moonshine_v2",
            "encoder": str(_pick_onnx(ms_encoders, prefer_int8)),
            "decoder": str(_pick_onnx(merged_decoders, prefer_int8)),
            "tokens": str(tokens),
        }

    return None


def ensure_bpe_vocab(model_info: dict[str, Any]) -> str:
    """Tự động xuất bpe.vocab từ bpe.model nếu model Zipformer chưa có sẵn bpe.vocab."""
    if model_info.get("bpe_vocab") and Path(model_info["bpe_vocab"]).is_file():
        return model_info["bpe_vocab"]

    bpe_model_path = model_info.get("bpe_model", "")
    if not bpe_model_path or not Path(bpe_model_path).is_file():
        return ""

    out_vocab = Path(bpe_model_path).with_name("bpe.vocab")
    try:
        import sentencepiece as spm  # type: ignore

        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model_path)
        with out_vocab.open("w", encoding="utf-8") as f:
            for i in range(sp.get_piece_size()):
                piece = sp.id_to_piece(i)
                score = sp.get_score(i)
                f.write(f"{piece}\t{score}\n")
        print(f"[Hotwords] Đã tự động tạo file {out_vocab.name} từ {Path(bpe_model_path).name}.")
        model_info["bpe_vocab"] = str(out_vocab)
        return str(out_vocab)
    except ImportError:
        print(
            "[Cảnh báo] Cần thư viện 'sentencepiece' để tự tạo bpe.vocab cho hotwords.txt "
            "(chạy: pip install sentencepiece)."
        )
        return ""
    except Exception as exc:
        print(f"[Cảnh báo] Không thể xuất bpe.vocab từ bpe.model: {exc}")
        return ""


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    print(f"Đang tải: {url}")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Tiến độ: {pct:5.1f}% ({mb_done:.1f} / {mb_total:.1f} MB)")
        else:
            mb_done = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Đã tải: {mb_done:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
        print()
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def ensure_silero_vad() -> Path:
    """Đảm bảo file silero_vad.onnx đã có trên máy (dùng cho Moonshine)."""
    if SILERO_VAD_PATH.is_file():
        return SILERO_VAD_PATH
    print("\n[VAD] Model Moonshine cần thêm silero_vad.onnx (~2 MB) để tự động ngắt câu.")
    _download_file(SILERO_VAD_URL, SILERO_VAD_PATH)
    return SILERO_VAD_PATH


def download_preset_model(preset: dict[str, str]) -> Path:
    """Tải và giải nén model từ danh sách preset."""
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    target_dir = MODELS_ROOT / preset["dir_name"]
    if inspect_model_dir(target_dir) is not None:
        print(f"Model đã có sẵn tại: {target_dir}")
        return target_dir

    archive_path = MODELS_ROOT / f"{preset['dir_name']}.tar.bz2"
    _download_file(preset["url"], archive_path)

    print("Đang giải nén model...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=MODELS_ROOT)

    archive_path.unlink(missing_ok=True)

    if inspect_model_dir(target_dir) is None:
        raise RuntimeError(
            f"Giải nén xong nhưng không tìm thấy đủ file model trong {target_dir}."
        )
    print(f"Đã tải và giải nén xong tại: {target_dir}")
    return target_dir


def resolve_model_interactive(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Kiểm tra model trong config; nếu chưa có thì hiển thị menu CLI cho người dùng chọn."""
    prefer_int8 = bool(cfg.get("prefer_int8", True))
    saved_dir = str(cfg.get("model_dir", "")).strip()

    if saved_dir:
        candidate = Path(saved_dir).expanduser()
        info = inspect_model_dir(candidate, prefer_int8=prefer_int8)
        if info is not None:
            print(f"[Model] Dùng model đã lưu: {candidate} ({info['model_type']})")
            return candidate, info
        print(f"[Cảnh báo] Thư mục model trong config không tồn tại hoặc thiếu file: {candidate}")

    while True:
        print("\n=== CHỌN MODEL AI NHẬN DIỆN TIẾNG ANH (SHERPA-ONNX) ===")
        for idx, item in enumerate(PRESET_MODELS, start=1):
            print(f"  {idx}. {item['name']}")
            print(f"     └─ {item['size_note']}")
        manual_idx = len(PRESET_MODELS) + 1
        print(f"  {manual_idx}. Tự nhập đường dẫn thư mục model (hoặc tên thư mục trong ./models/)")

        choice_raw = input(f"\nNhập lựa chọn [1-{manual_idx}] (mặc định 1): ").strip() or "1"
        if not choice_raw.isdigit():
            print("Lựa chọn không hợp lệ, vui lòng nhập số.")
            continue

        choice = int(choice_raw)
        if 1 <= choice <= len(PRESET_MODELS):
            preset = PRESET_MODELS[choice - 1]
            try:
                model_dir = download_preset_model(preset)
                info = inspect_model_dir(model_dir, prefer_int8=prefer_int8)
                if info is not None:
                    cfg["model_dir"] = str(model_dir)
                    cfg["model_type"] = info["model_type"]
                    return model_dir, info
            except Exception as exc:
                print(f"[Lỗi tải model] {exc}")
        elif choice == manual_idx:
            raw_path = input(
                "Nhập đường dẫn thư mục model (hoặc tên thư mục trong ./models/): "
            ).strip().strip('"').strip("'")
            if not raw_path:
                continue
            p = Path(raw_path).expanduser()
            if not p.is_dir() and (MODELS_ROOT / raw_path).is_dir():
                p = MODELS_ROOT / raw_path
            info = inspect_model_dir(p, prefer_int8=prefer_int8)
            if info is None:
                print(
                    "[Lỗi] Không nhận diện được model sherpa-onnx hợp lệ trong thư mục này "
                    "(cần có tokens.txt và các file .onnx tương ứng)."
                )
                continue
            cfg["model_dir"] = str(p.resolve())
            cfg["model_type"] = info["model_type"]
            return p.resolve(), info
        else:
            print("Số lựa chọn nằm ngoài phạm vi.")
