import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../lib/api.js';
import { label } from '../../lib/actions.js';
import { useCoach } from '../../lib/coach.jsx';
import { clearCanvas, drawSkeleton } from '../../lib/pose.js';
import { captureFrame, openCamera } from './capture.js';

const SEND_INTERVAL_MS = 100;

/** Sends frames from a <video> roughly every 100 ms, one request at a time. */
function useFrameLoop(videoRef, canvasRef, { active, clock, mirrored, view }) {
  const { sendFrame } = useCoach();
  const busy = useRef(false);
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(async () => {
      const video = videoRef.current;
      if (busy.current || !video || video.readyState < 2) return;
      busy.current = true;
      try {
        const timestamp = clock();
        const blob = await captureFrame(video);
        if (!blob) return;
        const data = await sendFrame(blob, timestamp);
        if (data) {
          drawSkeleton(canvasRef.current, video, data.landmarks, data.angles, {
            frame: data.image, mirrored, showBones: view.skeleton, showAngles: view.angles,
          });
        }
      } finally {
        busy.current = false;
      }
    }, SEND_INTERVAL_MS);
    return () => clearInterval(id);
  }, [active, clock, mirrored, view.skeleton, view.angles, videoRef, canvasRef, sendFrame]);
}

export function LiveSource({ view }) {
  const { setSourceRunning, cameraRequest } = useCoach();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const startedAt = useRef(0);
  const [facing, setFacing] = useState('user');
  const [streaming, setStreaming] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStreaming(false);
    setSourceRunning('live', false);
    clearCanvas(canvasRef.current);
  }, [setSourceRunning]);

  const start = useCallback(async (mode = facing) => {
    stop();
    setError('');
    setStarting(true);
    try {
      const stream = await openCamera(mode);
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play().catch(() => {});
      startedAt.current = performance.now();
      setStreaming(true);
      setSourceRunning('live', true);
    } catch (err) {
      setError(err.message);
    } finally {
      setStarting(false);
    }
  }, [facing, stop, setSourceRunning]);

  useEffect(() => stop, [stop]);
  useEffect(() => { if (cameraRequest > 0) start(); }, [cameraRequest]); // eslint-disable-line react-hooks/exhaustive-deps

  // Real capture times, so the server's 1.9 s window spans 1.9 s of movement.
  const clock = useCallback(() => (performance.now() - startedAt.current) / 1000, []);
  const mirrored = facing === 'user';
  useFrameLoop(videoRef, canvasRef, { active: streaming, clock, mirrored, view });

  const flip = () => {
    const next = facing === 'user' ? 'environment' : 'user';
    setFacing(next);
    if (streaming) start(next);
  };

  return (
    <div className="source">
      <div className="viewport">
        <video ref={videoRef} className={mirrored ? 'is-mirrored' : ''} playsInline muted />
        <canvas ref={canvasRef} className="overlay" />
        {!streaming && (
          <div className="viewport-empty">
            <h3>Stand back about two metres.</h3>
            <p>Your whole body should fit in the frame. Frames go to your own computer&apos;s server and nowhere else.</p>
            <button type="button" className="btn btn-accent btn-lg" onClick={() => start()} disabled={starting}>
              {starting ? 'Opening camera…' : 'Turn on camera'}
            </button>
            {error && <p className="inline-error" role="alert">{error}</p>}
          </div>
        )}
      </div>
      <div className="source-bar">
        <button type="button" className="btn btn-ghost" onClick={stop} disabled={!streaming}>Stop camera</button>
        <button type="button" className="btn btn-ghost" onClick={flip}>Flip camera</button>
      </div>
    </div>
  );
}

function DropZone({ accept, onFile, title, hint, children }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  return (
    <div
      className={`dropzone ${over ? 'is-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onFile(file);
      }}
    >
      <h3>{title}</h3>
      <p>{hint}</p>
      <div className="dropzone-actions">
        <button type="button" className="btn btn-accent" onClick={() => inputRef.current?.click()}>Choose a file</button>
        {children}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => { const file = e.target.files?.[0]; if (file) onFile(file); e.target.value = ''; }}
      />
    </div>
  );
}

export function PhotoSource({ view }) {
  const { sendFrame } = useCoach();
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const [src, setSrc] = useState('');
  const [result, setResult] = useState(null);
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(false);

  const redraw = useCallback(() => {
    if (!result) return;
    drawSkeleton(canvasRef.current, imgRef.current, result.landmarks, result.angles, {
      frame: result.image, showBones: view.skeleton, showAngles: view.angles,
    });
  }, [result, view.skeleton, view.angles]);

  useEffect(() => {
    redraw();
    const el = imgRef.current;
    if (!el) return undefined;
    const observer = new ResizeObserver(redraw);
    observer.observe(el);
    return () => observer.disconnect();
  }, [redraw]);

  const analyse = async (blob, url) => {
    setSrc(url);
    setResult(null);
    setNote('');
    setLoading(true);
    const data = await sendFrame(blob);
    setLoading(false);
    if (!data) { setNote('The server could not analyse this photo.'); return; }
    setResult(data);
    const joints = (data.landmarks || []).filter((lm) => lm.visibility > 0.35).length;
    setNote(data.detected
      ? `Found ${joints} of 33 joints and measured the angles. Naming the exercise and counting reps need about two seconds of movement, so use the camera or a video for those.`
      : 'No person found in this photo. Try one where the whole body is in view.');
  };

  const loadSample = async (url) => {
    const blob = await (await fetch(url)).blob();
    analyse(blob, url);
  };

  if (!src) {
    return (
      <div className="source">
        <DropZone
          accept="image/jpeg,image/png,image/webp"
          onFile={(file) => analyse(file, URL.createObjectURL(file))}
          title="Check a pose in a photo."
          hint="JPG, PNG or WEBP up to 10 MB. You’ll get the skeleton and joint angles."
        >
          <button type="button" className="btn btn-ghost" onClick={() => loadSample('/media/squat-bottom.jpg')}>Try a squat photo</button>
          <button type="button" className="btn btn-ghost" onClick={() => loadSample('/media/deadlift-hinge.jpg')}>Try a deadlift photo</button>
        </DropZone>
      </div>
    );
  }

  return (
    <div className="source">
      <div className="viewport">
        <img ref={imgRef} src={src} alt="Your uploaded pose" onLoad={redraw} />
        <canvas ref={canvasRef} className="overlay" />
        {loading && <div className="viewport-loading">Finding joints…</div>}
      </div>
      {note && <p className="source-note">{note}</p>}
      <div className="source-bar">
        <button type="button" className="btn btn-ghost" onClick={() => { setSrc(''); setResult(null); setNote(''); }}>Use another photo</button>
      </div>
    </div>
  );
}

export function VideoSource({ view }) {
  const { chooseTarget, setSourceRunning, counters } = useCoach();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [src, setSrc] = useState('');
  const [playing, setPlaying] = useState(false);
  const [note, setNote] = useState('');
  const countersRef = useRef(counters);
  countersRef.current = counters;

  useEffect(() => () => setSourceRunning('video', false), [setSourceRunning]);

  const open = async (url, targetAction) => {
    setNote('');
    // Each video starts from zero so its count belongs to that video.
    await api.reset().catch(() => {});
    if (targetAction !== undefined) await chooseTarget(targetAction);
    setSrc(url);
  };

  // The video's own clock is the timestamp: exact, even if frames are dropped.
  const clock = useCallback(() => videoRef.current?.currentTime ?? 0, []);
  useFrameLoop(videoRef, canvasRef, { active: playing, clock, mirrored: false, view });

  const onEnded = () => {
    setPlaying(false);
    setSourceRunning('video', false);
    const counted = Object.entries(countersRef.current || {}).filter(([, n]) => n > 0);
    setNote(counted.length
      ? `Finished: ${counted.map(([action, n]) => `${n} × ${label(action).toLowerCase()}`).join(', ')}. Press play to watch again (the count carries on), or reset.`
      : 'Finished with no complete reps. Pick the exercise under “Count reps for” and press play again.');
  };

  if (!src) {
    return (
      <div className="source">
        <DropZone
          accept="video/mp4,video/webm,video/ogg,video/quicktime"
          onFile={(file) => open(URL.createObjectURL(file))}
          title="Count reps in a recording."
          hint="MP4, WEBM or MOV, filmed with the whole body in frame. It plays once and counts as it goes."
        >
          <button type="button" className="btn btn-ghost" onClick={() => open('/media/squat-clip.webm', 'squat')}>
            Try the squat clip
          </button>
        </DropZone>
      </div>
    );
  }

  return (
    <div className="source">
      <div className="viewport">
        <video
          ref={videoRef}
          src={src}
          controls
          autoPlay
          muted
          playsInline
          onPlay={() => { setPlaying(true); setSourceRunning('video', true); setNote(''); }}
          onPause={() => { setPlaying(false); setSourceRunning('video', false); }}
          onEnded={onEnded}
        />
        <canvas ref={canvasRef} className="overlay" />
      </div>
      {note && <p className="source-note">{note}</p>}
      <div className="source-bar">
        <button type="button" className="btn btn-ghost" onClick={() => { setSrc(''); setPlaying(false); setSourceRunning('video', false); setNote(''); }}>
          Use another video
        </button>
      </div>
    </div>
  );
}
