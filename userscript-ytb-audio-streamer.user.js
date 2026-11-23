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

  const getStored = (key, fallback) =>
    typeof GM_getValue === "function" ? GM_getValue(key, fallback) : fallback;
  const setStored = (key, val) => {
    if (typeof GM_setValue === "function") GM_setValue(key, val);
  };

  let wsUrl = getStored("wsUrl", DEFAULT_WS_URL);
  let fontSize = Number(getStored("subFontSize", 22)) || 22;
  let showEN = Boolean(getStored("showEN", true));
  let showVI = Boolean(getStored("showVI", true));
  let offsetX = Number(getStored("subOffsetX", 0)) || 0;
  let offsetY = Number(getStored("subOffsetY", 0)) || 0;

  let isStreaming = false;
  let ws = null;
  let reconnectTimer = null;

  // Trạng thái chế độ hiển thị do server Python (config.json -> display_mode) gửi xuống
  let serverDisplayMode = "userscript"; // "userscript" | "window" | "both"
  let serverEnableVI = true;

  // Bộ đệm hiển thị phụ đề EN & VI (giữ câu chốt gần nhất + câu đang nói dở)
  const enState = { lastFinal: "", partial: "" };
  const viState = { lastFinal: "", partial: "" };

  // Web Audio state
  let audioCtx = null;
  let processorNode = null;
  let silentGain = null;
  const videoSourceCache = new WeakMap();
  let activeVideo = null;
  let activeSourceNode = null;

  // Downsampling 44.1/48kHz -> 16kHz
  let dsPhase = 0;
  let dsPrevSample = 0;
  let outBuffer = new Float32Array(CHUNK_SAMPLES);
  let outOffset = 0;

  // UI elements (dùng createElement/textContent thuần để tương thích tuyệt đối YouTube Trusted Types)
  let controlBtn = null;
  let badgeDot = null;
  let statusToast = null;
  let toastTimer = null;

  // In-player subtitle overlay elements
  let overlayBox = null;
  let overlayToolbar = null;
  let overlayStatusEl = null;
  let enLineEl = null;
  let viLineEl = null;
  let btnToggleEn = null;
  let btnToggleVi = null;

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
      controlBtn.title = `Đang kết nối ${wsUrl} (Alt+S để tắt)`;
      badgeDot.style.backgroundColor = "#22c55e";
    } else if (state === "connecting") {
      controlBtn.title = `Đang kết nối tới ${wsUrl}... (Alt+S để tắt)`;
      badgeDot.style.backgroundColor = "#f59e0b";
    } else {
      controlBtn.title = `Lỗi kết nối ${wsUrl} — đang thử lại... (Alt+S để tắt)`;
      badgeDot.style.backgroundColor = "#ef4444";
    }
  }

  function shouldShowInPlayerOverlay() {
    return isStreaming && (serverDisplayMode === "userscript" || serverDisplayMode === "both");
  }

  function makeMiniBtn(label, title, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    Object.assign(b.style, {
      background: "rgba(255, 255, 255, 0.14)",
      color: "#f1f5f9",
      border: "none",
      borderRadius: "4px",
      padding: "2px 6px",
      fontSize: "11px",
      fontWeight: "600",
      cursor: "pointer",
      lineHeight: "1.3",
    });
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });
    return b;
  }

  function applyOverlayPositionAndStyle() {
    if (!overlayBox) return;
    overlayBox.style.transform = `translate(calc(-50% + ${offsetX}px), ${offsetY}px)`;
    if (enLineEl) {
      enLineEl.style.fontSize = `${Math.max(13, fontSize - 3)}px`;
    }
    if (viLineEl) {
      viLineEl.style.fontSize = `${fontSize}px`;
    }
    if (btnToggleEn) {
      btnToggleEn.style.opacity = showEN ? "1" : "0.45";
    }
    if (btnToggleVi) {
      btnToggleVi.style.opacity = showVI ? "1" : "0.45";
      btnToggleVi.style.display = serverEnableVI ? "inline-block" : "none";
    }
  }

  function renderSubtitles() {
    if (!overlayBox || !enLineEl || !viLineEl) return;

    if (!shouldShowInPlayerOverlay()) {
      overlayBox.style.display = "none";
      return;
    }
    overlayBox.style.display = "flex";

    const enText = enState.partial || enState.lastFinal;
    const viText = viState.partial || viState.lastFinal;

    if (showEN && enText) {
      enLineEl.textContent = enText;
      enLineEl.style.display = "block";
    } else if (showEN && !viText) {
      enLineEl.textContent = "Đang nghe phụ đề tiếng Anh...";
      enLineEl.style.display = "block";
    } else {
      enLineEl.style.display = "none";
    }

    if (serverEnableVI && showVI && viText) {
      viLineEl.textContent = viText;
      viLineEl.style.display = "block";
    } else {
      viLineEl.style.display = "none";
    }
  }

  function ensureInPlayerOverlay() {
    const player =
      document.getElementById("movie_player") ||
      document.querySelector(".html5-video-player");
    if (!player) return;

    if (overlayBox && overlayBox.parentElement === player) {
      return;
    }

    if (!overlayBox) {
      overlayBox = document.createElement("div");
      overlayBox.id = "cs2-yt-sub-overlay";
      Object.assign(overlayBox.style, {
        position: "absolute",
        left: "50%",
        bottom: "70px",
        zIndex: "60",
        display: "none",
        flexDirection: "column",
        alignItems: "center",
        maxWidth: "82%",
        minWidth: "320px",
        padding: "8px 16px 10px 16px",
        borderRadius: "10px",
        backgroundColor: "rgba(24, 25, 30, 0.72)",
        border: "1px solid rgba(255, 255, 255, 0.16)",
        boxShadow: "0 4px 18px rgba(0, 0, 0, 0.45)",
        fontFamily: "Roboto, Arial, sans-serif",
        textAlign: "center",
        userSelect: "none",
        cursor: "grab",
        pointerEvents: "auto",
      });

      // Thanh điều khiển siêu gọn chỉ hiện rõ khi rê chuột vào khung phụ đề
      overlayToolbar = document.createElement("div");
      Object.assign(overlayToolbar.style, {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "5px",
        marginBottom: "4px",
        opacity: "0.25",
        transition: "opacity 0.18s ease",
      });

      overlayStatusEl = document.createElement("span");
      overlayStatusEl.textContent = "AI SUB";
      Object.assign(overlayStatusEl.style, {
        fontSize: "10.5px",
        color: "#cbd5e1",
        marginRight: "4px",
      });

      const btnSmaller = makeMiniBtn("A-", "Giảm cỡ chữ", () => {
        fontSize = Math.max(14, fontSize - 2);
        setStored("subFontSize", fontSize);
        applyOverlayPositionAndStyle();
      });
      const btnLarger = makeMiniBtn("A+", "Tăng cỡ chữ", () => {
        fontSize = Math.min(44, fontSize + 2);
        setStored("subFontSize", fontSize);
        applyOverlayPositionAndStyle();
      });
      btnToggleEn = makeMiniBtn("EN", "Ẩn / Hiện tiếng Anh", () => {
        showEN = !showEN;
        setStored("showEN", showEN);
        applyOverlayPositionAndStyle();
        renderSubtitles();
      });
      btnToggleVi = makeMiniBtn("VI", "Ẩn / Hiện tiếng Việt", () => {
        showVI = !showVI;
        setStored("showVI", showVI);
        applyOverlayPositionAndStyle();
        renderSubtitles();
      });
      const btnResetPos = makeMiniBtn("↺", "Đặt lại vị trí giữa màn hình", () => {
        offsetX = 0;
        offsetY = 0;
        setStored("subOffsetX", 0);
        setStored("subOffsetY", 0);
        applyOverlayPositionAndStyle();
      });

      overlayToolbar.appendChild(overlayStatusEl);
      overlayToolbar.appendChild(btnSmaller);
      overlayToolbar.appendChild(btnLarger);
      overlayToolbar.appendChild(btnToggleEn);
      overlayToolbar.appendChild(btnToggleVi);
      overlayToolbar.appendChild(btnResetPos);

      enLineEl = document.createElement("div");
      Object.assign(enLineEl.style, {
        color: "#cbd5e1",
        fontWeight: "500",
        lineHeight: "1.35",
        textShadow: "0 1px 4px rgba(0, 0, 0, 0.9)",
        wordBreak: "break-word",
      });

      viLineEl = document.createElement("div");
      Object.assign(viLineEl.style, {
        color: "#ffffff",
        fontWeight: "700",
        lineHeight: "1.38",
        marginTop: "4px",
        textShadow: "0 1px 5px rgba(0, 0, 0, 0.95)",
        wordBreak: "break-word",
      });

      overlayBox.appendChild(overlayToolbar);
      overlayBox.appendChild(enLineEl);
      overlayBox.appendChild(viLineEl);

      overlayBox.addEventListener("mouseenter", () => {
        if (overlayToolbar) overlayToolbar.style.opacity = "1";
      });
      overlayBox.addEventListener("mouseleave", () => {
        if (overlayToolbar) overlayToolbar.style.opacity = "0.25";
      });

      // Ngăn click lên khung phụ đề làm pause video YouTube + hỗ trợ kéo di chuyển vị trí
      let dragging = false;
      let startX = 0;
      let startY = 0;
      let baseOffX = 0;
      let baseOffY = 0;

      overlayBox.addEventListener("click", (e) => e.stopPropagation());
      overlayBox.addEventListener("dblclick", (e) => e.stopPropagation());
      overlayBox.addEventListener("mousedown", (e) => {
        if (e.button !== 0 || e.target.tagName === "BUTTON") return;
        e.stopPropagation();
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        baseOffX = offsetX;
        baseOffY = offsetY;
        overlayBox.style.cursor = "grabbing";
      });
      window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        offsetX = Math.round(baseOffX + (e.clientX - startX));
        offsetY = Math.round(baseOffY + (e.clientY - startY));
        applyOverlayPositionAndStyle();
      });
      window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        if (overlayBox) overlayBox.style.cursor = "grab";
        setStored("subOffsetX", offsetX);
        setStored("subOffsetY", offsetY);
      });
    }

    player.appendChild(overlayBox);
    applyOverlayPositionAndStyle();
    renderSubtitles();
  }

  function resetSubtitleBuffers() {
    enState.lastFinal = "";
    enState.partial = "";
    viState.lastFinal = "";
    viState.partial = "";
    renderSubtitles();
  }

  function handleServerMessage(rawData) {
    if (typeof rawData !== "string") return;
    let msg;
    try {
      msg = JSON.parse(rawData);
    } catch (_) {
      return;
    }

    if (msg.type === "mode") {
      serverDisplayMode = msg.display_mode || "userscript";
      serverEnableVI = msg.enable_vi !== false;
      if (typeof msg.font_size === "number" && !getStored("subFontSize", 0)) {
        fontSize = msg.font_size;
      }
      ensureInPlayerOverlay();
      applyOverlayPositionAndStyle();
      renderSubtitles();
      return;
    }

    if (msg.type === "reset") {
      resetSubtitleBuffers();
      return;
    }

    if (msg.type === "status") {
      if (overlayStatusEl && msg.text) {
        overlayStatusEl.textContent = msg.text;
      }
      return;
    }

    if (msg.type === "subtitle") {
      ensureInPlayerOverlay();
      const text = String(msg.text || "").trim();
      if (!text) return;
      const target = msg.lang === "vi" ? viState : enState;
      if (msg.is_final) {
        target.lastFinal = text;
        target.partial = "";
      } else {
        target.partial = text;
      }
      renderSubtitles();
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
      // 4096 mẫu (~85ms @ 48kHz), tương thích Chrome/Edge/Firefox và vượt Trusted Types YouTube
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

        const mono = new Float32Array(frameCount);
        if (ch1) {
          for (let i = 0; i < frameCount; i++) {
            mono[i] = 0.5 * (ch0[i] + ch1[i]);
          }
        } else {
          mono.set(ch0);
        }

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

      processorNode.connect(silentGain);
      silentGain.connect(audioCtx.destination);
    }

    if (activeVideo !== video) {
      if (activeSourceNode && processorNode) {
        try {
          activeSourceNode.disconnect(processorNode);
        } catch (_) {}
      }

      let sourceNode = videoSourceCache.get(video);
      if (!sourceNode) {
        sourceNode = audioCtx.createMediaElementSource(video);
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
      ensureInPlayerOverlay();
      showToast("Đã kết nối phụ đề AI (YouTube)");
    };

    ws.onmessage = (event) => {
      handleServerMessage(event.data);
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
    resetSubtitleBuffers();
    ensureInPlayerOverlay();
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
    if (overlayBox) {
      overlayBox.style.display = "none";
    }
    updateButtonUI("disconnected");
    showToast("Đã dừng phụ đề AI");
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

  const domObserver = new MutationObserver(() => {
    injectYouTubeControlButton();
    if (isStreaming) {
      ensureInPlayerOverlay();
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
        ensureInPlayerOverlay();
        const video = findMainVideo();
        if (video && video !== activeVideo) {
          try {
            ensureAudioPipeline(video);
          } catch (_) {}
        }
      }
    }, 400);
  });

  document.addEventListener("fullscreenchange", () => {
    if (isStreaming) {
      ensureInPlayerOverlay();
      renderSubtitles();
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.altKey && !e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "s") {
      e.preventDefault();
      toggleStreaming();
    }
  });

  if (typeof GM_registerMenuCommand === "function") {
    GM_registerMenuCommand("Bật / Tắt phụ đề AI (Alt+S)", toggleStreaming);
    GM_registerMenuCommand("Đổi địa chỉ WebSocket Python", () => {
      const nextUrl = window.prompt("Nhập địa chỉ WebSocket server Python:", wsUrl);
      if (nextUrl && nextUrl.trim()) {
        wsUrl = nextUrl.trim();
        setStored("wsUrl", wsUrl);
        showToast(`Đã lưu WebSocket: ${wsUrl}`);
        if (isStreaming) {
          connectWebSocket();
        }
      }
    });
  }

  injectYouTubeControlButton();
})();