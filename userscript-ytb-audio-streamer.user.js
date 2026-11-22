// ==UserScript==
// @name         YouTube Audio Streamer
// @namespace    https://example.com
// @version      1.0.0
// @description  Bắt âm thanh trực tiếp từ video YouTube (16kHz Mono Float32LE) bằng Web Audio API và gửi qua WebSocket cục bộ sang script Python sherpa-onnx.
// @match        https://www.youtube.com/*
// @match        https://m.youtube.com/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_MS = 100;
  const CHUNK_SAMPLES = Math.round((TARGET_SAMPLE_RATE * CHUNK_MS) / 1000); // 1600 mẫu
  const DEFAULT_WS_URL = "ws://127.0.0.1:8765";

  let wsUrl = typeof GM_getValue === "function" ? GM_getValue("wsUrl", DEFAULT_WS_URL) : DEFAULT_WS_URL;
  let isStreaming = false;
  let ws = null;
  let reconnectTimer = null;

  // Web Audio state (tái sử dụng 1 AudioContext & cache MediaElementSource theo từng thẻ <video>)
  let audioCtx = null;
  let processorNode = null;
  let silentGain = null;
  const videoSourceCache = new WeakMap();
  let activeVideo = null;
  let activeSourceNode = null;

  // Bộ đệm downsampling tuyến tính từ sampleRate gốc (44.1/48kHz) -> 16kHz
  let dsPhase = 0;
  let dsPrevSample = 0;
  let outBuffer = new Float32Array(CHUNK_SAMPLES);
  let outOffset = 0;

  // UI elements (dùng createElement thuần để tương thích tuyệt đối YouTube Trusted Types)
  let controlBtn = null;
  let badgeDot = null;
  let statusToast = null;
  let toastTimer = null;

  function showToast(text, isError = false) {
    if (!statusToast) {
      statusToast = document.createElement("div");
      Object.assign(statusToast.style, {
        position: "fixed",
        top: "16px",
        right: "16px",
        zIndex: "2147483647",
        padding: "8px 14px",
        borderRadius: "8px",
        fontSize: "13px",
        fontFamily: "Roboto, Arial, sans-serif",
        fontWeight: "500",
        color: "#f8fafc",
        backgroundColor: "rgba(15, 23, 42, 0.88)",
        border: "1px solid rgba(255, 255, 255, 0.15)",
        pointerEvents: "none",
        transition: "opacity 0.2s ease",
        opacity: "0",
      });
      document.documentElement.appendChild(statusToast);
    }
    statusToast.textContent = text;
    statusToast.style.borderColor = isError ? "rgba(239, 68, 68, 0.7)" : "rgba(56, 189, 248, 0.6)";
    statusToast.style.opacity = "1";
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      if (statusToast) statusToast.style.opacity = "0";
    }, 2600);
  }

  function updateButtonUI(state) {
    if (!controlBtn || !badgeDot) return;
    if (!isStreaming) {
      controlBtn.title = "Bật gửi âm thanh phụ đề AI (Alt+S)";
      badgeDot.style.backgroundColor = "#94a3b8";
      controlBtn.style.opacity = "0.85";
      return;
    }
    controlBtn.style.opacity = "1";
    if (state === "connected") {
      controlBtn.title = `Đang gửi âm thanh tới ${wsUrl} (Alt+S để tắt)`;
      badgeDot.style.backgroundColor = "#22c55e";
    } else if (state === "connecting") {
      controlBtn.title = `Đang kết nối tới ${wsUrl}... (Alt+S để tắt)`;
      badgeDot.style.backgroundColor = "#f59e0b";
    } else {
      controlBtn.title = `Lỗi kết nối ${wsUrl} — đang thử lại... (Alt+S để tắt)`;
      badgeDot.style.backgroundColor = "#ef4444";
    }
  }

  function findMainVideo() {
    const main = document.querySelector("video.html5-main-video");
    if (main) return main;
    const videos = Array.from(document.querySelectorAll("video"));
    return videos.find((v) => v.readyState > 0) || videos[0] || null;
  }

  function ensureAudioPipeline(video) {
    if (!audioCtx) {
      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextCtor();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }

    if (!processorNode) {
      // 4096 mẫu (~85ms @ 48kHz), tương thích cả Chrome/Edge/Firefox và vượt Trusted Types YouTube
      processorNode = audioCtx.createScriptProcessor(4096, 2, 1);
      silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;

      processorNode.onaudioprocess = (event) => {
        if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (!activeVideo || activeVideo.paused || activeVideo.ended) return;

        const inputBuf = event.inputBuffer;
        const ch0 = inputBuf.getChannelData(0);
        const ch1 = inputBuf.numberOfChannels > 1 ? inputBuf.getChannelData(1) : null;
        const frameCount = ch0.length;
        const ratio = audioCtx.sampleRate / TARGET_SAMPLE_RATE;

        // Trộn stereo -> mono
        const mono = new Float32Array(frameCount);
        if (ch1) {
          for (let i = 0; i < frameCount; i++) {
            mono[i] = 0.5 * (ch0[i] + ch1[i]);
          }
        } else {
          mono.set(ch0);
        }

        // Nội suy tuyến tính về đúng 16kHz và đóng gói 100ms (1600 mẫu)
        while (dsPhase < frameCount) {
          const idx = Math.floor(dsPhase);
          const frac = dsPhase - idx;
          const s0 = idx === 0 ? dsPrevSample : mono[idx - 1];
          const s1 = mono[idx];
          outBuffer[outOffset++] = s0 + (s1 - s0) * frac;

          if (outOffset >= CHUNK_SAMPLES) {
            if (ws.bufferedAmount < 262144) {
              ws.send(outBuffer.buffer);
            }
            outBuffer = new Float32Array(CHUNK_SAMPLES);
            outOffset = 0;
          }
          dsPhase += ratio;
        }
        dsPhase -= frameCount;
        dsPrevSample = mono[frameCount - 1];
      };

      // Nối qua GainNode(0) để ScriptProcessor luôn chạy mà không phát tiếng lặp
      processorNode.connect(silentGain);
      silentGain.connect(audioCtx.destination);
    }

    // Nếu chuyển sang thẻ <video> mới, chỉ ngắt nhánh gửi sang processor (giữ nguyên nhánh ra loa!)
    if (activeVideo !== video) {
      if (activeSourceNode && processorNode) {
        try {
          activeSourceNode.disconnect(processorNode);
        } catch (_) {}
      }

      let sourceNode = videoSourceCache.get(video);
      if (!sourceNode) {
        sourceNode = audioCtx.createMediaElementSource(video);
        // Luôn nối trực tiếp ra loa/tai nghe ở chất lượng gốc 48kHz stereo
        sourceNode.connect(audioCtx.destination);
        videoSourceCache.set(video, sourceNode);
      }

      sourceNode.connect(processorNode);
      activeVideo = video;
      activeSourceNode = sourceNode;
    }
  }

  function connectWebSocket() {
    if (!isStreaming) return;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      try {
        ws.close();
      } catch (_) {}
      ws = null;
    }

    updateButtonUI("connecting");
    try {
      ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
    } catch (err) {
      updateButtonUI("error");
      showToast(`URL WebSocket không hợp lệ: ${err.message}`, true);
      return;
    }

    ws.onopen = () => {
      updateButtonUI("connected");
      ws.send(
        JSON.stringify({
          type: "config",
          sampleRate: TARGET_SAMPLE_RATE,
          channels: 1,
          format: "f32le",
        })
      );
      showToast("Đã kết nối phụ đề AI (YouTube Audio)");
    };

    ws.onerror = () => {
      updateButtonUI("error");
    };

    ws.onclose = () => {
      if (!isStreaming) return;
      updateButtonUI("connecting");
      reconnectTimer = setTimeout(connectWebSocket, 1500);
    };
  }

  function startStreaming() {
    const video = findMainVideo();
    if (!video) {
      showToast("Không tìm thấy trình phát video YouTube trên trang này.", true);
      return;
    }

    try {
      dsPhase = 0;
      dsPrevSample = 0;
      outOffset = 0;
      outBuffer = new Float32Array(CHUNK_SAMPLES);
      ensureAudioPipeline(video);
    } catch (err) {
      showToast(`Lỗi Web Audio API: ${err.message}`, true);
      return;
    }

    isStreaming = true;
    updateButtonUI("connecting");
    connectWebSocket();
  }

  function stopStreaming() {
    isStreaming = false;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      try {
        ws.close();
      } catch (_) {}
      ws = null;
    }
    updateButtonUI("disconnected");
    showToast("Đã dừng gửi âm thanh phụ đề AI");
  }

  function toggleStreaming() {
    if (isStreaming) {
      stopStreaming();
    } else {
      startStreaming();
    }
  }

  function injectYouTubeControlButton() {
    const rightControls = document.querySelector(".ytp-right-controls");
    if (!rightControls) return;
    if (document.getElementById("cs2-sub-yt-btn")) return;

    const btn = document.createElement("button");
    btn.id = "cs2-sub-yt-btn";
    btn.className = "ytp-button";
    btn.type = "button";
    Object.assign(btn.style, {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "5px",
      width: "auto",
      padding: "0 8px",
      fontSize: "12px",
      fontWeight: "700",
      fontFamily: "Roboto, Arial, sans-serif",
      color: "#ffffff",
      verticalAlign: "top",
      cursor: "pointer",
    });

    const dot = document.createElement("span");
    Object.assign(dot.style, {
      width: "7px",
      height: "7px",
      borderRadius: "50%",
      backgroundColor: "#94a3b8",
      display: "inline-block",
    });

    const label = document.createElement("span");
    label.textContent = "AI SUB";

    btn.appendChild(dot);
    btn.appendChild(label);
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleStreaming();
    });

    rightControls.insertBefore(btn, rightControls.firstChild);
    controlBtn = btn;
    badgeDot = dot;
    updateButtonUI(ws && ws.readyState === WebSocket.OPEN ? "connected" : "disconnected");
  }

  // Tự gắn lại nút điều khiển & theo dõi thẻ <video> mới khi YouTube chuyển trang (SPA)
  const domObserver = new MutationObserver(() => {
    injectYouTubeControlButton();
    if (isStreaming) {
      const video = findMainVideo();
      if (video && video !== activeVideo) {
        try {
          ensureAudioPipeline(video);
        } catch (_) {}
      }
    }
  });
  domObserver.observe(document.documentElement, { childList: true, subtree: true });

  window.addEventListener("yt-navigate-finish", () => {
    setTimeout(() => {
      injectYouTubeControlButton();
      if (isStreaming) {
        const video = findMainVideo();
        if (video && video !== activeVideo) {
          try {
            ensureAudioPipeline(video);
          } catch (_) {}
        }
      }
    }, 400);
  });

  // Phím tắt Alt + S để bật/tắt nhanh kể cả khi đang xem toàn màn hình
  window.addEventListener("keydown", (e) => {
    if (e.altKey && !e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "s") {
      e.preventDefault();
      toggleStreaming();
    }
  });

  // Menu trong Tampermonkey / Violentmonkey
  if (typeof GM_registerMenuCommand === "function") {
    GM_registerMenuCommand("Bật / Tắt gửi âm thanh phụ đề (Alt+S)", toggleStreaming);
    GM_registerMenuCommand("Đổi địa chỉ WebSocket Python", () => {
      const nextUrl = window.prompt("Nhập địa chỉ WebSocket server Python:", wsUrl);
      if (nextUrl && nextUrl.trim()) {
        wsUrl = nextUrl.trim();
        if (typeof GM_setValue === "function") {
          GM_setValue("wsUrl", wsUrl);
        }
        showToast(`Đã lưu WebSocket: ${wsUrl}`);
        if (isStreaming) {
          connectWebSocket();
        }
      }
    });
  }

  injectYouTubeControlButton();
})();