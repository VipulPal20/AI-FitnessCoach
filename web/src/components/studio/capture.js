// Frame capture shared by the live camera and video sources.

const FRAME_WIDTH = 480;
let canvas = null;

/** Downscale the current video frame to 480 px wide (aspect kept) as a JPEG blob. */
export function captureFrame(video) {
  if (!video?.videoWidth || !video?.videoHeight) return Promise.resolve(null);
  canvas = canvas || document.createElement('canvas');
  const width = Math.min(FRAME_WIDTH, video.videoWidth);
  const height = Math.round((width * video.videoHeight) / video.videoWidth);
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(video, 0, 0, width, height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.82));
}

let audio = null;

/** Short rising two-note chime for a completed rep. */
export function playChime() {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === 'suspended') audio.resume();
    const now = audio.currentTime;
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, now);
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.12);
    gain.gain.setValueAtTime(0.22, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
    osc.connect(gain).connect(audio.destination);
    osc.start(now);
    osc.stop(now + 0.3);
  } catch {
    // Audio is optional.
  }
}

/** Ask for a camera, falling back to looser constraints on picky desktop webcams. */
export async function openCamera(facingMode) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This browser cannot access a camera. Use Chrome, Edge or Firefox over http://127.0.0.1 or https.');
  }
  const attempts = [
    { video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
    { video: true, audio: false },
  ];
  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      lastError = err;
      if (err?.name === 'NotAllowedError') break;
    }
  }
  if (lastError?.name === 'NotAllowedError') {
    throw new Error('Camera permission was blocked. Allow camera access in the address bar, then try again.');
  }
  if (lastError?.name === 'NotFoundError') throw new Error('No camera was found on this device.');
  if (lastError?.name === 'NotReadableError') throw new Error('The camera is busy in another app. Close it there and try again.');
  throw new Error(lastError?.message || 'The camera could not be started.');
}
