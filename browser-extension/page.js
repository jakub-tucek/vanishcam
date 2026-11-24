// Page world: adds a "VanishCam" camera to Meet's device list and serves it from a
// canvas that bridge.js fills with frames from the local VanishCam app.
(() => {
  const media = navigator.mediaDevices;
  if (!media || media.__vanishcam) return;
  media.__vanishcam = true;

  const CHANNEL = "vanishcam";
  const DEVICE_ID = "vanishcam";
  const LABEL = "VanishCam";
  const FPS = 30;
  const originalEnumerate = media.enumerateDevices.bind(media);
  const originalGetUserMedia = media.getUserMedia.bind(media);

  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const context = canvas.getContext("2d");
  let decoding = false;

  const post = (type) => window.postMessage({ channel: CHANNEL, type }, location.origin);

  function drawPlaceholder(text) {
    context.fillStyle = "#1a1d22";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#e6ecef";
    context.font = "600 40px -apple-system, Helvetica, sans-serif";
    context.textAlign = "center";
    context.fillText(text, canvas.width / 2, canvas.height / 2);
  }
  drawPlaceholder("Start the VanishCam app");

  window.addEventListener("message", async (event) => {
    const data = event.data;
    if (event.source !== window || !data || data.channel !== CHANNEL) return;
    if (data.type === "status" && !data.online) {
      drawPlaceholder("Start the VanishCam app");
    } else if (data.type === "frame" && !decoding) {
      decoding = true;
      try {
        const bitmap = await createImageBitmap(new Blob([data.buffer], { type: "image/jpeg" }));
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
        }
        context.drawImage(bitmap, 0, 0);
        bitmap.close();
      } finally {
        decoding = false;
      }
    }
  });

  function deviceInfo(kind) {
    const info = {
      deviceId: DEVICE_ID,
      groupId: DEVICE_ID,
      kind,
      label: LABEL,
      toJSON() {
        return { deviceId: DEVICE_ID, groupId: DEVICE_ID, kind, label: LABEL };
      },
    };
    const prototype = kind === "videoinput" && window.InputDeviceInfo
      ? InputDeviceInfo.prototype
      : MediaDeviceInfo.prototype;
    return Object.setPrototypeOf(info, prototype);
  }

  function wrapTrack(track) {
    const settings = track.getSettings.bind(track);
    const clone = track.clone.bind(track);
    const stop = track.stop.bind(track);
    let stopped = false;
    Object.defineProperty(track, "label", { value: LABEL });
    track.getSettings = () => ({
      ...settings(),
      deviceId: DEVICE_ID,
      groupId: DEVICE_ID,
      width: canvas.width,
      height: canvas.height,
      frameRate: FPS,
    });
    track.getCapabilities = () => ({
      deviceId: DEVICE_ID,
      groupId: DEVICE_ID,
      width: { min: 1, max: canvas.width },
      height: { min: 1, max: canvas.height },
      frameRate: { min: 1, max: FPS },
    });
    track.applyConstraints = async () => {};
    track.clone = () => {
      post("start");
      return wrapTrack(clone());
    };
    track.stop = () => {
      if (!stopped) {
        stopped = true;
        post("stop");
      }
      stop();
    };
    track.addEventListener("ended", () => {
      if (!stopped) {
        stopped = true;
        post("stop");
      }
    });
    return track;
  }

  function wantsVanishCam(video) {
    if (!video || typeof video !== "object") return false;
    return JSON.stringify([video.deviceId ?? null, video.advanced ?? null]).includes(DEVICE_ID);
  }

  media.enumerateDevices = async function () {
    const devices = await originalEnumerate();
    return [...devices, deviceInfo("videoinput")];
  };

  media.getUserMedia = async function (constraints = {}) {
    if (!wantsVanishCam(constraints.video)) return originalGetUserMedia(constraints);
    const stream = new MediaStream();
    if (constraints.audio) {
      const audio = await originalGetUserMedia({ audio: constraints.audio });
      audio.getAudioTracks().forEach((track) => stream.addTrack(track));
    }
    post("start");
    stream.addTrack(wrapTrack(canvas.captureStream(FPS).getVideoTracks()[0]));
    return stream;
  };
})();
