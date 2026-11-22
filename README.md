# CS2 Live Subtitles (Local ASR + Local LLM Translation)

Script Python tạo **phụ đề tiếng Anh thời gian thực** và **dịch song song sang tiếng Việt** chạy hoàn toàn cục bộ trên máy cá nhân (hỗ trợ cả **Windows** và **Linux**), không tốn phí API và không gửi dữ liệu ra máy chủ ngoài.

### Điểm nổi bật
- **Nhận diện giọng nói độ trễ thấp trên CPU:** Dùng [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) với các model streaming mới như **NVIDIA Nemotron Speech Streaming 0.6B** (tự động viết hoa và chấm câu), Streaming Zipformer hoặc Moonshine.
- **Dịch tiếng Việt thông minh trên GPU bằng Local LLM:** Kết nối trực tiếp tới `llama-server` (`llama.cpp`, hỗ trợ Vulkan/CUDA) chạy **Gemma 4 E4B** (hoặc Gemma 3 4B), kèm ngữ cảnh 2 câu trước đó và bảng thuật ngữ CS2 để dịch tự nhiên, giữ nguyên tiếng lóng esport (*AWP, eco, clutch, retake, smoke, flash...*).
- **Hai cửa sổ phụ đề nổi độc lập (PyQt6):** Hiển thị riêng cửa sổ **Tiếng Anh (EN)** và **Tiếng Việt (VI)** với nền đen xám mờ 60%, chữ trắng rõ nét, tự cuộn xuống dòng mới nhất, luôn nổi trên cùng (*always-on-top*), có nút chỉnh cỡ chữ (`A-`/`A+`) và nút ẩn xuống thanh tác vụ (`_`).
- **Bắt âm thanh trực tiếp từ YouTube qua Userscript:** Không phá vỡ chế độ toàn màn hình (Fullscreen) của YouTube và chạy mượt trên cả Chrome, Edge, Brave lẫn Firefox.

---

## Yêu cầu chuẩn bị

- Đã cài đặt **Python 3.10+** (`python` / `python3`) và thêm vào `PATH` của hệ điều hành.
- Đã cài đặt **`llama-server`** (`llama.cpp`) và thêm vào `PATH` (xem mục cài đặt nhanh bên dưới).
- Trình duyệt có cài **Violentmonkey** hoặc **Tampermonkey**.

---

## 1. Cài đặt script Python và thư viện qua `venv`

Mở terminal tại thư mục chứa mã nguồn của dự án rồi chạy các lệnh tương ứng với hệ điều hành của bạn:

### Linux (Bash)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U -r requirements.txt
```
> [!NOTE]
> **Lưu ý trên Windows:** Nếu PowerShell báo lỗi không cho phép chạy script kích hoạt `venv`, hãy chạy lệnh sau một lần duy nhất rồi kích hoạt lại:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

---

## 2. Cài đặt nhanh `llama-server` (`llama.cpp`)

`llama-server` dùng để chạy model **Gemma 4 E4B** trên GPU (khuyên dùng bản **Vulkan** cho card AMD/Intel, hoặc bản **CUDA** cho card NVIDIA).

### Linux (Bash)
- **Tải bản build sẵn:** Vào trang [Releases của ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) → chọn bản `bXXXX` mới nhất → tải file `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz` → giải nén và thêm thư mục chứa `llama-server` vào `PATH`.

### Windows (PowerShell)
- Vào trang [Releases của ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) → chọn bản `bXXXX` mới nhất → tải file `llama-bXXXX-bin-win-vulkan-x64.zip` (hoặc bản `win-cuda` nếu dùng GPU NVIDIA) → giải nén (ví dụ `C:\llama-cpp`) và thêm đường dẫn đó vào `PATH` của Windows.

### Kiểm tra GPU đã được nhận diện
```bash
llama-server --list-devices
```
Nếu thấy tên card màn hình của bạn (ví dụ `Vulkan0: AMD Radeon RX 6600`) là đã sẵn sàng.

---

## 3. Khởi động `llama-server` để dịch tiếng Việt bằng Gemma 4 E4B (không bắt buộc)

Mở một cửa sổ terminal riêng và chạy lệnh sau để bật máy chủ dịch cục bộ (lần đầu tiên sẽ tự tải model từ Hugging Face; cờ `enable_thinking: false` giúp tắt chế độ suy luận để dịch nhanh theo thời gian thực):

### Linux (Bash)
```bash
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL --host 127.0.0.1 --port 8080 -c 2048 -ngl 99 --chat-template-kwargs '{"enable_thinking": false}'
```

### Windows (PowerShell)
```powershell
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL --host 127.0.0.1 --port 8080 -c 2048 -ngl 99 --chat-template-kwargs '{\"enable_thinking\": false}'
```

> [!TIP]
> **Tùy chọn đổi thư mục lưu model:**
> - Nếu bạn đã tải sẵn file `.gguf` về máy, thay `-hf ...` bằng `-m "/duong/dan/toi/file-model.gguf"`.
> - Hoặc đặt biến môi trường `LLAMA_CACHE` (Linux: `export LLAMA_CACHE="/duong/dan/models"`, Windows: `setx LLAMA_CACHE "D:\AI-Models"`) trước khi chạy lệnh `-hf`.

Đợi cho đến khi thấy thông báo `server is listening on http://127.0.0.1:8080` nghĩa là `llama-server` đã load xong. Giữ nguyên terminal này mở.

---

## 4. Chạy script Python lần đầu tiên

Mở một cửa sổ terminal khác tại thư mục dự án, kích hoạt `venv` và khởi chạy `main.py`:

### Linux (Bash)
```bash
source venv/bin/activate
python main.py
```

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

### Các bước thiết lập ở lần chạy đầu tiên
1. **Chọn model nhận diện giọng nói (ASR):** Menu CLI sẽ hiện ra — nhập **`1`** (rồi nhấn Enter) để tự động tải và giải nén **NVIDIA Nemotron Speech Streaming EN 0.6B (~670 MB)**. Nếu bạn đã tải sẵn model vào ổ cứng, chọn mục cuối để nhập đường dẫn thư mục model.
2. **Xác nhận cổng WebSocket:** Script tự kiểm tra cổng mặc định `127.0.0.1:8765` (nếu cổng bị chiếm, nhập địa chỉ `ip:port` mới như `127.0.0.1:8766`). Mọi thiết lập sẽ được tự động lưu vào file `config.json` cho các lần chạy sau.
3. **Chờ nạp model:** Sau vài giây nạp model vào CPU, **2 cửa sổ phụ đề nổi (EN và VI)** sẽ tự động xuất hiện ở phía dưới màn hình với trạng thái sẵn sàng kết nối.

> [!NOTE]
> Nếu bạn không dùng `llama-server` để chạy model LLM dịch tiếng Việt, bạn có thể ẩn cửa sổ phụ đề tiếng Việt xuống và chức năng phụ đề tiếng Anh vẫn hoạt động bình thường

---

## 5. Cài đặt Userscript (Violentmonkey / Tampermonkey) & Kết nối WebSocket

### Bước 1: Cài đặt tiện ích quản lý Userscript
Cài một trong hai tiện ích mở rộng sau cho trình duyệt của bạn (hỗ trợ Chrome, Edge, Brave, Firefox):
- [Violentmonkey](https://violentmonkey.github.io/)
- [Tampermonkey](https://www.tampermonkey.net/)

### Bước 2: Cài đặt file Userscript bằng link trực tiếp
Nhấn trực tiếp vào link cài đặt dưới đây:

**[Cài đặt YouTube Audio Streamer Userscript](https://example.com/userscript-ytb-audio-streamer.user.js)**

**Violentmonkey** hoặc **Tampermonkey** sẽ tự động nhận diện và mở trang xác nhận — bạn chỉ cần bấm **Confirm installation / Cài đặt** là xong.

### Bước 3: Kết nối với WebSocket Server của script Python
1. Đảm bảo `llama-server` và `python main.py` đều đang chạy.
2. Mở video hoặc livestream YouTube cần xem (nên tải lại trang nếu bạn vừa mới cài Userscript).
3. Bấm nút **`AI SUB`** nằm ngay trên thanh điều khiển góc dưới bên phải của trình phát YouTube, hoặc nhấn phím tắt **`Alt + S`**.
4. Kiểm tra trạng thái qua chấm màu trên nút `AI SUB`:
   - **Xanh lá:** Đã kết nối thành công tới `ws://127.0.0.1:8765` và đang truyền âm thanh sang script Python.
   - **Vàng / Đỏ:** Chưa kết nối được - hãy kiểm tra xem `python main.py` đã chạy chưa.
5. **Nếu bạn đổi cổng WebSocket trong script Python:** Bấm vào biểu tượng **Violentmonkey / Tampermonkey** trên thanh công cụ trình duyệt → chọn mục **`Đổi địa chỉ WebSocket Python`** của userscript → nhập đúng địa chỉ `ws://ip:port` mới (ví dụ `ws://127.0.0.1:8766`).
