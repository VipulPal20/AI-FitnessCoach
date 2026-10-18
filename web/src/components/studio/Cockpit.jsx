import { motion } from 'motion/react';
import { REP_ACTIONS, REP_GOALS, STATUS_TEXT, cueFor, label, stageLabel } from '../../lib/actions.js';
import { useCoach } from '../../lib/coach.jsx';
import { SpringNumber } from '../../ui/motion.jsx';

const ANGLES = [
  ['left_elbow', 'Left elbow'], ['right_elbow', 'Right elbow'],
  ['left_knee', 'Left knee'], ['right_knee', 'Right knee'],
  ['left_hip', 'Left hip'], ['right_hip', 'Right hip'],
];

export default function Cockpit() {
  const { frame, counters, target, chooseTarget, error, studioMode } = useCoach();
  const detected = frame?.current_action && frame.current_action !== 'unknown' ? frame.current_action : null;
  const repAction = target || detected;
  const reps = repAction ? Number(counters[repAction] || 0) : 0;
  const goal = repAction ? REP_GOALS[repAction] : 12;
  const stage = repAction ? frame?.stages?.[repAction] : null;
  const angles = frame?.angles || {};
  const ranked = Object.entries(frame?.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const buffer = frame?.buffer_progress || '0/20';

  let coach = 'Turn on the camera, upload a video, or pick an exercise from the library.';
  if (error) coach = `Server error: ${error}`;
  else if (studioMode === 'photo') coach = frame?.detected ? 'Pose measured. Reps need movement, so switch to the camera or a video to count them.' : 'Pick a photo to measure the pose.';
  // Losing the body beats everything; otherwise a known exercise gets form cues
  // rather than the classifier's own progress messages.
  else if (frame && !frame.detected && STATUS_TEXT[frame.message]) coach = STATUS_TEXT[frame.message];
  else if (frame?.message?.startsWith('Partial pose')) coach = STATUS_TEXT[frame.message];
  else if (repAction) coach = cueFor(repAction, stage) || coach;
  else if (frame) coach = STATUS_TEXT[frame.message] || 'Start moving and it will pick up the exercise.';

  return (
    <aside className="cockpit" data-ui aria-label="Live results">
      <div className="cockpit-main">
        <div className="cockpit-head">
          <h3 className="cockpit-action">{repAction ? label(repAction) : 'Ready when you are'}</h3>
          <span className={`stage-chip stage-${stage || 'rest'}`}>{stageLabel(stage)}</span>
        </div>
        {target && detected && detected !== target && (
          <p className="cockpit-sees">The AI thinks this looks like a {label(detected).toLowerCase()}.</p>
        )}
        <div className="cockpit-reps">
          <SpringNumber value={reps} className="cockpit-rep-number tnum" />
          <span className="cockpit-rep-goal">of {goal} reps</span>
        </div>
        <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={goal} aria-valuenow={reps}>
          <motion.span animate={{ scaleX: Math.min(1, reps / goal) }} transition={{ type: 'spring', stiffness: 160, damping: 22 }} />
        </div>

        <label className="picker">
          <span>Count reps for</span>
          <select value={target || ''} onChange={(e) => chooseTarget(e.target.value || null)}>
            <option value="">Auto-detect with AI</option>
            {REP_ACTIONS.map((action) => (
              <option key={action} value={action}>{label(action)}</option>
            ))}
          </select>
        </label>

        <p className="cockpit-coach" aria-live="polite">{coach}</p>
      </div>

      <div className="cockpit-block">
        <h4>Joint angles</h4>
        <dl className="angle-grid">
          {ANGLES.map(([key, name]) => (
            <div key={key}>
              <dt>{name}</dt>
              <dd className="tnum">{angles[key] !== undefined ? `${Math.round(angles[key])}°` : '—'}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="cockpit-block">
        <div className="cockpit-block-head">
          <h4>What the AI sees</h4>
          <span className="tnum">window {buffer}</span>
        </div>
        <ul className="prob-list">
          {[0, 1, 2].map((i) => {
            const [action, p] = ranked[i] || [null, 0];
            const pct = Math.round(Number(p || 0) * 100);
            return (
              <li key={i}>
                <span>{action && pct > 0 ? label(action) : '—'}</span>
                <span className="prob-track"><motion.span animate={{ scaleX: pct / 100 }} transition={{ duration: 0.25 }} /></span>
                <span className="tnum">{pct}%</span>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
