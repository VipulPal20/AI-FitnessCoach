import { motion, useReducedMotion } from 'motion/react';
import { Suspense, lazy, useState } from 'react';
import { scrollToId } from '../lib/coach.jsx';
import { stageLabel } from '../lib/actions.js';
import { Magnetic, SplitHeadline } from '../ui/motion.jsx';

const SkeletonScene = lazy(() => import('./SkeletonScene.jsx'));

const CLIPS = [
  ['squat', 'Squat'],
  ['deadlift', 'Deadlift'],
];

export default function Hero() {
  const reduce = useReducedMotion();
  const [move, setMove] = useState('squat');
  const [readout, setReadout] = useState({ angle: null, stage: null, reps: 0, label: 'Knee' });

  return (
    <section id="top" className="hero">
      <div className="hero-copy">
        <SplitHeadline className="hero-title" text={'Your camera\nis the coach.'} delay={0.15} />
        <motion.p
          className="hero-lede"
          initial={{ opacity: 0, y: reduce ? 0 : 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          FitnessCoach finds 33 joints in your body, works out which exercise you are doing and counts every rep.
          No wearables, no sign-up. Just a webcam and some floor space.
        </motion.p>
        <motion.div
          className="hero-actions"
          initial={{ opacity: 0, y: reduce ? 0 : 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <Magnetic>
            <button type="button" className="btn btn-ink btn-lg" onClick={() => scrollToId('studio')}>
              Start training
              <span className="btn-arrow" aria-hidden="true">→</span>
            </button>
          </Magnetic>
          <button type="button" className="text-link" onClick={() => scrollToId('how')}>
            See how it works
          </button>
        </motion.div>
      </div>

      <motion.figure
        className="hero-stage"
        data-ui
        initial={{ opacity: 0, scale: reduce ? 1 : 0.94 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.25, duration: 1, ease: [0.22, 1, 0.36, 1] }}
      >
        <Suspense fallback={<div className="skeleton-scene is-loading" />}>
          <SkeletonScene move={move} onReadout={setReadout} />
        </Suspense>

        <div className="hero-readout" aria-live="off">
          <div>
            <span className="readout-key">{readout.label}</span>
            <span className="readout-value tnum">{readout.angle ? `${Math.round(readout.angle)}°` : '—'}</span>
          </div>
          <div>
            <span className="readout-key">Stage</span>
            <span className={`readout-value stage-${readout.stage || 'rest'}`}>{stageLabel(readout.stage)}</span>
          </div>
          <div>
            <span className="readout-key">Reps</span>
            <motion.span key={readout.reps} className="readout-value tnum" initial={{ scale: reduce ? 1 : 1.5 }} animate={{ scale: 1 }}>
              {readout.reps}
            </motion.span>
          </div>
        </div>

        <div className="hero-switch" role="group" aria-label="Captured movement">
          {CLIPS.map(([id, text]) => (
            <button key={id} type="button" className={move === id ? 'is-on' : ''} aria-pressed={move === id} onClick={() => setMove(id)}>
              {move === id && <motion.span layoutId="hero-switch" className="hero-switch-bg" />}
              <span>{text}</span>
            </button>
          ))}
        </div>

        <figcaption>
          A real {move} from the training footage, replayed from MediaPipe&apos;s 3D landmarks. Move your cursor to turn the figure.
        </figcaption>
      </motion.figure>
    </section>
  );
}
