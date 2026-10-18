import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useRef, useState } from 'react';
import { api } from '../../lib/api.js';
import { label } from '../../lib/actions.js';
import { useCoach } from '../../lib/coach.jsx';
import { Reveal } from '../../ui/motion.jsx';
import Cockpit from './Cockpit.jsx';
import { LiveSource, PhotoSource, VideoSource } from './Sources.jsx';
import { playChime } from './capture.js';

const MODES = [
  ['live', 'Live camera'],
  ['video', 'Video'],
  ['photo', 'Photo'],
];

export default function Studio() {
  const { studioMode, setStudioMode, lastRep, reset, selectedExercise, setSelectedExercise, chooseTarget } = useCoach();
  const [view, setView] = useState({ skeleton: true, angles: true, sound: true });
  const [burst, setBurst] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const soundRef = useRef(view.sound);
  soundRef.current = view.sound;

  useEffect(() => {
    if (!lastRep) return undefined;
    if (soundRef.current) playChime();
    setBurst(lastRep);
    const id = setTimeout(() => setBurst(null), 900);
    return () => clearTimeout(id);
  }, [lastRep]);

  const toggle = (key) => setView((v) => ({ ...v, [key]: !v[key] }));

  return (
    <section id="studio" className="studio">
      <div className="studio-head">
        <Reveal as="h2" className="section-title">The studio.</Reveal>
        <Reveal as="p" className="section-lede" delay={0.08}>
          Train in front of the camera, run a recording through it, or check a single pose. The live numbers update about ten times a second.
        </Reveal>
      </div>

      <div className="studio-toolbar">
        <div className="segmented" role="tablist" aria-label="Input">
          {MODES.map(([id, text]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={studioMode === id}
              className={studioMode === id ? 'is-on' : ''}
              onClick={() => setStudioMode(id)}
            >
              {studioMode === id && <motion.span layoutId="studio-mode" className="segmented-bg" transition={{ type: 'spring', stiffness: 400, damping: 34 }} />}
              <span>{text}</span>
            </button>
          ))}
        </div>
        <div className="studio-toggles">
          {[['skeleton', 'Skeleton'], ['angles', 'Angles'], ['sound', 'Rep sound']].map(([key, text]) => (
            <label key={key} className="switch">
              <input type="checkbox" checked={view[key]} onChange={() => toggle(key)} />
              <span className="switch-track" aria-hidden="true" />
              {text}
            </label>
          ))}
          <button type="button" className="btn btn-ghost btn-sm" aria-expanded={settingsOpen} onClick={() => setSettingsOpen((o) => !o)}>
            Sensitivity
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={reset}>Reset</button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {settingsOpen && <SettingsPanel />}
      </AnimatePresence>

      {selectedExercise && (
        <div className="studio-target">
          {selectedExercise.image && <img src={`/exercise-media/${selectedExercise.image}`} alt="" width="56" height="56" />}
          <p>
            Training <strong>{selectedExercise.name}</strong>
            {selectedExercise.rep_action
              ? `, counted as ${label(selectedExercise.rep_action).toLowerCase()} reps.`
              : '. Reps for this one aren’t counted yet, but the skeleton and angles still work.'}
          </p>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setSelectedExercise(null); chooseTarget(null); }}>Clear</button>
        </div>
      )}

      <div className="studio-grid">
        <div className="studio-stage">
          {studioMode === 'live' && <LiveSource view={view} />}
          {studioMode === 'video' && <VideoSource view={view} />}
          {studioMode === 'photo' && <PhotoSource view={view} />}
          <AnimatePresence>
            {burst && (
              <motion.div
                key={burst.at}
                className="rep-burst"
                initial={{ scale: 0.4, opacity: 0, rotate: -8 }}
                animate={{ scale: 1, opacity: 1, rotate: 0 }}
                exit={{ scale: 1.2, opacity: 0 }}
                transition={{ type: 'spring', stiffness: 420, damping: 16 }}
              >
                <strong>+1</strong>
                <span>{label(burst.action)}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <Cockpit />
      </div>
    </section>
  );
}

function SettingsPanel() {
  const [values, setValues] = useState({ min_detection_confidence: 0.5, min_tracking_confidence: 0.5, classification_threshold: 0.7 });
  const timer = useRef(null);
  const update = (key, value) => {
    const next = { ...values, [key]: value };
    setValues(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => api.settings(next).catch(() => {}), 250);
  };
  const sliders = [
    ['min_detection_confidence', 'Person detection', 'How sure the pose model must be that someone is in frame.'],
    ['min_tracking_confidence', 'Joint tracking', 'How sure it must be about each joint before using it.'],
    ['classification_threshold', 'Exercise naming', 'How sure the classifier must be before it names an exercise. It never goes below 0.70.'],
  ];
  return (
    <motion.div
      className="settings"
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="settings-inner">
        {sliders.map(([key, name, hint]) => (
          <label key={key} className="slider">
            <span className="slider-head">
              <span>{name}</span>
              <span className="tnum">{values[key].toFixed(2)}</span>
            </span>
            <input type="range" min="0.1" max="0.95" step="0.05" value={values[key]} onChange={(e) => update(key, Number(e.target.value))} />
            <span className="slider-hint">{hint}</span>
          </label>
        ))}
      </div>
    </motion.div>
  );
}
