import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useEffect, useRef, useState } from 'react';
import { POSE_CONNECTIONS } from '../lib/pose.js';
import { Reveal } from '../ui/motion.jsx';

const STEPS = [
  {
    id: 'joints',
    title: 'It finds 33 joints.',
    body: 'Every frame goes through Google’s MediaPipe pose model, which places 33 landmarks on the body: shoulders, elbows, wrists, hips, knees, ankles and feet. Below, the joints it found in a frame of the training footage.',
  },
  {
    id: 'angles',
    title: 'It measures your angles.',
    body: 'Three joints make an angle. Hip, knee and ankle give the knee angle; shoulder, elbow and wrist give the elbow. At the bottom of this squat the left knee is bent to 86°. Standing tall it reads about 178°.',
  },
  {
    id: 'window',
    title: 'It watches 1.9 seconds.',
    body: 'One photo can’t tell a squat from a lunge. The classifier looks at the last 1.9 seconds of movement, resampled to 20 poses, and reads how the whole body travels before it names the exercise.',
  },
  {
    id: 'count',
    title: 'Then it counts.',
    body: 'A rep is a full round trip: start position, working position, and back. It has to take at least 0.4 seconds, so a wobble never counts. If you tell it which exercise you’re doing, it counts that one directly.',
  },
];

function useStepObserver(refs, setActive) {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => {
        if (entry.isIntersecting) setActive(Number(entry.target.dataset.index));
      }),
      { rootMargin: '-40% 0px -45% 0px' },
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, [refs, setActive]);
}

export default function HowItWorks() {
  const [active, setActive] = useState(0);
  const [poses, setPoses] = useState(null);
  const stepRefs = useRef([]);
  useStepObserver(stepRefs, setActive);

  useEffect(() => {
    fetch('/media/poses.json').then((r) => r.json()).then(setPoses).catch(() => setPoses({}));
  }, []);

  return (
    <section id="how" className="how">
      <div className="how-head">
        <Reveal as="h2" className="section-title">How a webcam learns to count.</Reveal>
        <Reveal as="p" className="section-lede" delay={0.08}>
          No sensors and no markers. Just geometry, a short memory and a few small decision trees, trained on 120 clips of people lifting.
        </Reveal>
      </div>

      <div className="how-body">
        <div className="how-visual" aria-hidden="true">
          <div className="how-visual-frame">
            <AnimatePresence mode="wait">
              <motion.div
                key={STEPS[active].id}
                className="how-visual-inner"
                initial={{ opacity: 0, scale: 1.03 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
              >
                <StepVisual step={STEPS[active].id} poses={poses} />
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        <ol className="how-steps">
          {STEPS.map((step, index) => (
            <li
              key={step.id}
              ref={(el) => { stepRefs.current[index] = el; }}
              data-index={index}
              className={`how-step ${active === index ? 'is-active' : ''}`}
            >
              <h3>{step.title}</h3>
              <p>{step.body}</p>
              <div className="how-step-visual" aria-hidden="true">
                <StepVisual step={step.id} poses={poses} />
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function StepVisual({ step, poses }) {
  if (step === 'joints') return <PosePhoto src="/media/squat-stand.jpg" pose={poses?.['squat-stand.jpg']} />;
  if (step === 'angles') return <PosePhoto src="/media/squat-bottom.jpg" pose={poses?.['squat-bottom.jpg']} angles={['left_knee', 'left_hip', 'right_knee']} />;
  if (step === 'window') return <WindowVisual />;
  return <CountVisual poses={poses} />;
}

const ANGLE_JOINT = { left_knee: 25, right_knee: 26, left_hip: 23, right_hip: 24, left_elbow: 13, right_elbow: 14 };

/** Photo with the landmarks MediaPipe detected in it, drawn bone by bone. */
function PosePhoto({ src, pose, angles = [], compact = false }) {
  const reduce = useReducedMotion();
  const points = pose?.landmarks || [];
  const visible = (i) => points[i] && points[i][2] >= 0.35;
  const xy = (i) => [points[i][0] * 1280, points[i][1] * 720];
  return (
    <div className={`pose-photo ${compact ? 'is-compact' : ''}`}>
      <img src={src} alt="" loading="lazy" />
      <svg viewBox="0 0 1280 720" preserveAspectRatio="xMidYMid slice">
        {POSE_CONNECTIONS.map(([a, b], i) => (visible(a) && visible(b) ? (
          <motion.line
            key={`${a}-${b}`}
            x1={xy(a)[0]} y1={xy(a)[1]} x2={xy(b)[0]} y2={xy(b)[1]}
            className="pose-bone"
            initial={{ pathLength: reduce ? 1 : 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.5, delay: reduce ? 0 : 0.15 + i * 0.035 }}
          />
        ) : null))}
        {points.map((p, i) => (i >= 11 && visible(i) ? (
          <motion.circle
            key={i}
            cx={xy(i)[0]} cy={xy(i)[1]} r={compact ? 9 : 7}
            className="pose-joint"
            initial={{ scale: reduce ? 1 : 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 18, delay: reduce ? 0 : 0.1 + i * 0.02 }}
          />
        ) : null))}
        {angles.map((key, i) => {
          const joint = ANGLE_JOINT[key];
          const value = pose?.angles?.[key];
          if (!visible(joint) || value === undefined) return null;
          const [x, y] = xy(joint);
          const left = key.startsWith('left');
          return (
            <motion.g key={key} initial={{ opacity: 0, x: left ? 20 : -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.9 + i * 0.12 }}>
              <rect x={left ? x + 22 : x - 150} y={y - 26} width="128" height="52" rx="26" className="pose-tag" />
              <text x={left ? x + 86 : x - 86} y={y + 9} textAnchor="middle" className="pose-tag-text">
                {Math.round(value)}°
              </text>
            </motion.g>
          );
        })}
      </svg>
    </div>
  );
}

function WindowVisual() {
  return (
    <div className="window-visual">
      <video src="/media/squat-clip.webm" poster="/media/squat-stand.jpg" autoPlay muted loop playsInline />
      <div className="window-strip">
        {Array.from({ length: 20 }, (_, i) => (
          <span key={i} style={{ animationDelay: `${i * 0.095}s` }} />
        ))}
      </div>
      <p className="window-caption">20 poses · 1.9 s</p>
    </div>
  );
}

function CountVisual({ poses }) {
  const reduce = useReducedMotion();
  const frames = [
    ['squat-stand.jpg', 'Start'],
    ['squat-bottom.jpg', 'Effort'],
    ['squat-stand.jpg', 'Start'],
  ];
  return (
    <div className="count-visual">
      {frames.map(([name, text], i) => (
        <motion.div
          key={i}
          className="count-frame"
          initial={{ opacity: 0, y: reduce ? 0 : 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 + i * 0.25, duration: 0.6 }}
        >
          <PosePhoto src={`/media/${name}`} pose={poses?.[name]} compact />
          <span>{text}</span>
        </motion.div>
      ))}
      <motion.span
        className="count-plus"
        initial={{ scale: 0, rotate: -20 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ delay: reduce ? 0 : 1.05, type: 'spring', stiffness: 300, damping: 14 }}
      >
        +1
      </motion.span>
    </div>
  );
}
