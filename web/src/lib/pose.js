// MediaPipe's 33-landmark body model: which joints connect into bones.
export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
];

// Joints whose angle labels are drawn on the overlay: [landmark, angle key].
const ANGLE_TAGS = [
  [13, 'left_elbow'], [14, 'right_elbow'],
  [25, 'left_knee'], [26, 'right_knee'],
  [23, 'left_hip'], [24, 'right_hip'],
];

const VISIBLE = 0.35;

/**
 * Draw landmarks (pixel coordinates in the source frame) over a media element.
 * The canvas is sized to the element's rendered box so the overlay lines up
 * with object-fit: contain media.
 */
export function drawSkeleton(canvas, source, landmarks, angles, { frame, mirrored = false, showBones = true, showAngles = true } = {}) {
  if (!canvas || !source) return;
  const box = source.getBoundingClientRect();
  if (!box.width || !box.height) return;

  const natW = source.videoWidth || source.naturalWidth || box.width;
  const natH = source.videoHeight || source.naturalHeight || box.height;
  // Fit the frame inside the element the same way object-fit: contain does.
  const scale = Math.min(box.width / natW, box.height / natH);
  const drawW = natW * scale;
  const drawH = natH * scale;
  const offsetX = (box.width - drawW) / 2;
  const offsetY = (box.height - drawH) / 2;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width * dpr);
  canvas.height = Math.round(box.height * dpr);
  canvas.style.width = `${box.width}px`;
  canvas.style.height = `${box.height}px`;

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  if (!landmarks?.length) return;

  // The server's landmarks are in pixels of the frame it received, which is
  // usually a downscaled copy of the source.
  const frameW = frame?.width || natW;
  const frameH = frame?.height || natH;
  const byId = new Map(landmarks.map((lm) => [lm.id, lm]));
  const point = (lm) => {
    const x = (lm.x / frameW) * drawW;
    return [offsetX + (mirrored ? drawW - x : x), offsetY + (lm.y / frameH) * drawH];
  };

  if (showBones) {
    ctx.lineCap = 'round';
    ctx.lineWidth = Math.max(3, drawW / 160);
    ctx.strokeStyle = '#c6f432';
    for (const [a, b] of POSE_CONNECTIONS) {
      const p = byId.get(a);
      const q = byId.get(b);
      if (!p || !q || p.visibility < VISIBLE || q.visibility < VISIBLE) continue;
      const [x1, y1] = point(p);
      const [x2, y2] = point(q);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  const radius = Math.max(3.5, drawW / 200);
  for (const lm of landmarks) {
    if (lm.visibility < VISIBLE || lm.id < 11) continue;
    const [x, y] = point(lm);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#121412';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#c6f432';
    ctx.stroke();
  }

  if (showAngles && angles) {
    // Shrink the tags on small previews so neighbouring joints stay readable.
    const size = Math.max(10, Math.min(13, drawW / 40));
    const pad = size * 0.45;
    const height = size * 1.7;
    ctx.font = `600 ${size}px "Instrument Sans", system-ui, sans-serif`;
    ctx.textBaseline = 'middle';
    for (const [id, key] of ANGLE_TAGS) {
      const lm = byId.get(id);
      const value = angles[key];
      if (!lm || lm.visibility < VISIBLE || value === undefined) continue;
      const [x, y] = point(lm);
      const text = `${Math.round(value)}°`;
      const width = ctx.measureText(text).width + pad * 2;
      ctx.fillStyle = '#121412';
      ctx.beginPath();
      ctx.roundRect(x + radius + 4, y - height / 2, width, height, height / 2);
      ctx.fill();
      ctx.fillStyle = '#f3f3ef';
      ctx.fillText(text, x + radius + 4 + pad, y);
    }
  }
}

export function clearCanvas(canvas) {
  if (!canvas) return;
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

/** Angle at b (degrees) formed by points a-b-c, each [x, y, (z)]. */
export function jointAngle(a, b, c) {
  const ab = a.map((v, i) => v - b[i]);
  const cb = c.map((v, i) => v - b[i]);
  const dot = ab.reduce((s, v, i) => s + v * cb[i], 0);
  const norm = Math.hypot(...ab) * Math.hypot(...cb);
  if (!norm) return null;
  return (Math.acos(Math.max(-1, Math.min(1, dot / norm))) * 180) / Math.PI;
}
