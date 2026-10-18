import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { label } from '../lib/actions.js';
import { CountUp, Reveal } from '../ui/motion.jsx';

export default function ModelNotes() {
  const [metrics, setMetrics] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch('/api/v1/model/metrics')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setMetrics)
      .catch(() => setFailed(true));
  }, []);

  const test = metrics?.splits?.test;
  const rows = test
    ? Object.entries(test.f1).filter(([action]) => action !== 'unknown').sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section id="model" className="model">
      <div className="model-copy">
        <Reveal as="h2" className="section-title">How good is the AI, honestly?</Reveal>
        <Reveal as="p" className="section-lede" delay={0.08}>
          We tested it on recordings it never saw during training. It names the exercise correctly a bit more than half the time.
          It is strong on squats, push-ups and bench presses, and it struggles with lunges and with telling shoulder presses from pull-ups.
        </Reveal>
        <Reveal as="p" className="model-advice" delay={0.14}>
          That&apos;s why the studio has <strong>Count reps for</strong>. Tell it the exercise and it skips the guessing.
          It then counts from your joint angles, which is far more reliable.
        </Reveal>
      </div>

      <div className="model-figures" data-ui>
        {failed && <p className="fine-print">Metrics could not be loaded from the server.</p>}
        {test && (
          <>
            <div className="model-headline">
              <span className="model-big tnum"><CountUp to={test.accuracy * 100} suffix="%" /></span>
              <span>correct on {test.windows} held-out clips from {test.groups} recordings</span>
            </div>
            <table className="f1-table">
              <caption className="visually-hidden">F1 score per exercise on the held-out test set</caption>
              <thead>
                <tr><th scope="col">Exercise</th><th scope="col">F1 score</th><th scope="col"><span className="visually-hidden">Bar</span></th></tr>
              </thead>
              <tbody>
                {rows.map(([action, f1], i) => (
                  <tr key={action}>
                    <th scope="row">{label(action)}</th>
                    <td className="tnum">{f1.toFixed(2)}</td>
                    <td>
                      <span className="f1-track">
                        <motion.span
                          initial={{ scaleX: 0 }}
                          whileInView={{ scaleX: Math.max(f1, 0.01) }}
                          viewport={{ once: true }}
                          transition={{ delay: i * 0.05, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                          className={f1 >= 0.7 ? 'is-good' : ''}
                        />
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="fine-print">
              F1 balances precision and recall (1.00 is perfect). Trained on {metrics.usable_clips} usable clips ({metrics.windows} windows) with grouped splits,
              so no recording appears in both training and testing.
            </p>
          </>
        )}
      </div>
    </section>
  );
}
