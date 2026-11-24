// Isolated world: fetches frames from the local VanishCam app (the page's CSP would
// block this) and hands them to page.js.
(() => {
  const ENDPOINT = "http://127.0.0.1:47809/frame";
  const CHANNEL = "vanishcam";
  let consumers = 0;
  let running = false;
  let lastId = 0;

  const post = (message, transfer = []) =>
    window.postMessage({ channel: CHANNEL, ...message }, location.origin, transfer);

  async function pump() {
    running = true;
    let online = null;
    while (consumers > 0) {
      try {
        const response = await fetch(`${ENDPOINT}?after=${lastId}`, {
          headers: { "X-VanishCam": "1" },
          cache: "no-store",
        });
        if (online !== true) post({ type: "status", online: (online = true) });
        if (response.status === 200) {
          lastId = Number(response.headers.get("X-Frame-Id")) || lastId + 1;
          const buffer = await response.arrayBuffer();
          post({ type: "frame", buffer }, [buffer]);
        }
      } catch {
        if (online !== false) post({ type: "status", online: (online = false) });
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
    running = false;
  }

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (event.source !== window || !data || data.channel !== CHANNEL) return;
    if (data.type === "start") {
      consumers += 1;
      if (!running) pump();
    } else if (data.type === "stop") {
      consumers = Math.max(0, consumers - 1);
    }
  });
})();
