import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api.js';

const CoachContext = createContext(null);

/**
 * One place for everything the backend engine knows about the session:
 * the latest analysed frame, rep counters, the rep target, and the clock.
 * The engine is a single shared instance on the server, so the page keeps
 * one mirror of it here.
 */
export function CoachProvider({ children }) {
  const [health, setHealth] = useState({ status: 'checking' });
  const [frame, setFrame] = useState(null);
  const [counters, setCounters] = useState({});
  const [target, setTarget] = useState(null);
  const [selectedExercise, setSelectedExercise] = useState(null);
  const [studioMode, setStudioMode] = useState('live');
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const [running, setRunning] = useState(false);
  const [lastRep, setLastRep] = useState(null);
  const [error, setError] = useState(null);
  // Bumped when another part of the page (the library) asks the studio to open the camera.
  const [cameraRequest, setCameraRequest] = useState(0);
  const requestCamera = useCallback(() => setCameraRequest((n) => n + 1), []);
  const runningSources = useRef(new Set());

  useEffect(() => {
    let cancelled = false;
    api.health()
      .then((data) => !cancelled && setHealth({ status: 'ok', ...data }))
      .catch((err) => !cancelled && setHealth({ status: 'offline', message: err.message }));
    // Every page load starts in auto-detect; the server keeps state across reloads.
    api.setTarget(null).catch(() => {});
    // Rep counts do survive a reload on the server, so show them straight away.
    api.stats()
      .then((data) => !cancelled && setCounters(data.counters || {}))
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => setSessionSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  // Live camera and video playback each report whether they are streaming.
  const setSourceRunning = useCallback((source, isRunning) => {
    const set = runningSources.current;
    if (isRunning) set.add(source); else set.delete(source);
    setRunning(set.size > 0);
  }, []);

  const sendFrame = useCallback(async (blob, timestamp) => {
    try {
      const data = await api.processFrame(blob, timestamp);
      setFrame(data);
      setCounters(data.counters || {});
      if (data.target_action !== undefined) setTarget(data.target_action);
      if (data.rep_event) setLastRep({ action: data.rep_event, at: performance.now() });
      setError(null);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }, []);

  const chooseTarget = useCallback(async (action) => {
    setTarget(action || null);
    try {
      await api.setTarget(action || null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  const reset = useCallback(async () => {
    try {
      const data = await api.reset();
      setCounters(data.counters || {});
    } catch (err) {
      setError(err.message);
    }
    setFrame(null);
    setLastRep(null);
    setSessionSeconds(0);
  }, []);

  const value = useMemo(() => ({
    health, frame, setFrame, counters, target, chooseTarget, reset, sendFrame,
    selectedExercise, setSelectedExercise, studioMode, setStudioMode,
    sessionSeconds, running, setSourceRunning, lastRep, error, setError, cameraRequest, requestCamera,
  }), [health, frame, counters, target, chooseTarget, reset, sendFrame, selectedExercise,
    studioMode, sessionSeconds, running, setSourceRunning, lastRep, error, cameraRequest, requestCamera]);

  return <CoachContext.Provider value={value}>{children}</CoachContext.Provider>;
}

export function useCoach() {
  const context = useContext(CoachContext);
  if (!context) throw new Error('useCoach must be used inside <CoachProvider>');
  return context;
}

export function scrollToId(id) {
  const el = document.getElementById(id);
  if (!el) return;
  if (window.__lenis) window.__lenis.scrollTo(el, { offset: -72 });
  else el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
