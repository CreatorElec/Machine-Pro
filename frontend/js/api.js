/* API 封装: REST 请求 + WebSocket 训练通道 */
"use strict";

const Api = (() => {
  async function request(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let msg = res.statusText;
      try {
        const data = await res.json();
        msg = data.detail || msg;
      } catch (_) { /* ignore */ }
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return res.json();
  }

  const get = (path) => request(path);
  const post = (path, payload) =>
    request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

  function upload(file) {
    const fd = new FormData();
    fd.append("file", file);
    return request("/api/upload", { method: "POST", body: fd });
  }

  /**
   * WebSocket 多算法对比训练。
   * handlers: { onStart, onProgress, onLoss, onResult, onDone, onError }
   * 返回 { close() } 供中途取消。
   */
  function compareWS(payload, handlers) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/train`);
    ws.onopen = () => ws.send(JSON.stringify(payload));
    ws.onmessage = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch (_) { return; }
      const map = {
        start: handlers.onStart,
        progress: handlers.onProgress,
        loss: handlers.onLoss,
        result: handlers.onResult,
        done: handlers.onDone,
        error: handlers.onError,
      };
      const fn = map[ev.type];
      if (fn) fn(ev);
    };
    ws.onerror = () => handlers.onError && handlers.onError({ message: "WebSocket 连接失败" });
    return { close: () => ws.close() };
  }

  return { get, post, upload, compareWS };
})();
