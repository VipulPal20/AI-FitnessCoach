import { motion } from 'motion/react';
import { REP_ACTIONS, REP_GOALS, estimateKcal, label } from '../lib/actions.js';
import { scrollToId, useCoach } from '../lib/coach.jsx';
import { Reveal, SpringNumber } from '../ui/motion.jsx';

function clock(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, '0');
  const s = String(seconds % 60).padStart(2, '0');
  return `${m}:${s}`;
}

export default function Session() {
  const { counters, sessionSeconds, reset, running } = useCoach();
  const total = REP_ACTIONS.reduce((sum, action) => sum + Number(counters[action] || 0), 0);
  const ordered = [...REP_ACTIONS].sort((a, b) => Number(counters[b] || 0) - Number(counters[a] || 0));

  return (
    <section id="session" className="session">
      <div className="session-summary">
        <Reveal as="h2" className="section-title">This session.</Reveal>
        <div className="session-big" data-ui>
          <SpringNumber value={total} className="session-total tnum" />
          <span className="session-total-label">{total === 1 ? 'rep' : 'reps'} counted</span>
        </div>
        <dl className="session-facts" data-ui>
          <div><dt>Time training</dt><dd className="tnum">{clock(sessionSeconds)}</dd></div>
          <div><dt>Estimated burn</dt><dd className="tnum">{estimateKcal(counters, sessionSeconds)} kcal</dd></div>
          <div><dt>Status</dt><dd>{running ? 'Tracking now' : 'Paused'}</dd></div>
        </dl>
        <div className="session-actions">
          <button type="button" className="btn btn-ink" onClick={() => scrollToId('studio')}>{total ? 'Keep going' : 'Start a set'}</button>
          <button type="button" className="text-link" onClick={reset}>Reset the session</button>
        </div>
        <p className="fine-print">The calorie figure is a rough per-rep estimate, not a measurement.</p>
      </div>

      <ul className="session-list" data-ui>
        {ordered.map((action) => {
          const count = Number(counters[action] || 0);
          const goal = REP_GOALS[action];
          return (
            <motion.li key={action} layout transition={{ type: 'spring', stiffness: 300, damping: 30 }} className={count ? 'has-reps' : ''}>
              <span className="session-name">{label(action)}</span>
              <span className="session-bar"><motion.span animate={{ scaleX: Math.min(1, count / goal) }} transition={{ type: 'spring', stiffness: 140, damping: 20 }} /></span>
              <span className="session-count tnum">{count}<small>/{goal}</small></span>
            </motion.li>
          );
        })}
      </ul>
    </section>
  );
}
