import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useEffect, useState } from 'react';
import { scrollToId, useCoach } from '../lib/coach.jsx';

const LINKS = [
  ['how', 'How it works'],
  ['studio', 'Studio'],
  ['session', 'Session'],
  ['library', 'Library'],
  ['model', 'Accuracy'],
];
const DARK_SECTIONS = new Set(['studio']);

export default function Navbar() {
  const { health, running, counters } = useCoach();
  const reduce = useReducedMotion();
  const [scrolled, setScrolled] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [active, setActive] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);

  // Compact once the page moves; slide away on the way down, return on the way up.
  useEffect(() => {
    let last = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      setScrolled(y > 24);
      if (Math.abs(y - last) > 6) {
        setHidden(y > last && y > 240);
        last = y;
      }
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Track which section sits in the middle of the viewport.
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => entry.isIntersecting && setActive(entry.target.id)),
      { rootMargin: '-45% 0px -50% 0px' },
    );
    ['top', ...LINKS.map(([id]) => id)].forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : '';
    if (menuOpen) window.__lenis?.stop(); else window.__lenis?.start();
  }, [menuOpen]);

  const go = (id) => {
    setMenuOpen(false);
    scrollToId(id);
  };

  const totalReps = Object.values(counters || {}).reduce((sum, n) => sum + Number(n || 0), 0);
  const dark = DARK_SECTIONS.has(active) && scrolled;
  const statusText = health.status === 'ok' ? 'AI ready' : health.status === 'checking' ? 'Connecting' : 'Server offline';

  return (
    <>
      <motion.header
        className={`nav ${scrolled ? 'is-scrolled' : ''} ${dark ? 'is-dark' : ''}`}
        animate={{ y: hidden && !menuOpen ? '-110%' : '0%' }}
        transition={{ duration: reduce ? 0 : 0.35, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="nav-inner">
          <a className="wordmark" href="#top" onClick={(e) => { e.preventDefault(); go('top'); }}>
            <span className="wordmark-mark" aria-hidden="true" />
            FitnessCoach
          </a>

          <nav className="nav-links" aria-label="Sections">
            {LINKS.map(([id, text]) => (
              <a
                key={id}
                href={`#${id}`}
                className={active === id ? 'is-active' : ''}
                aria-current={active === id ? 'true' : undefined}
                onClick={(e) => { e.preventDefault(); go(id); }}
              >
                {text}
                {active === id && <motion.span layoutId="nav-marker" className="nav-marker" transition={{ type: 'spring', stiffness: 380, damping: 32 }} />}
              </a>
            ))}
          </nav>

          <div className="nav-actions">
            <AnimatePresence mode="popLayout" initial={false}>
              {running ? (
                <motion.span
                  key="live"
                  className="nav-live"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                >
                  <span className="live-dot" aria-hidden="true" />
                  <span className="tnum">{totalReps}</span> reps
                </motion.span>
              ) : (
                <motion.span
                  key="status"
                  className={`nav-status status-${health.status}`}
                  title={health.message || ''}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <span className="status-dot" aria-hidden="true" />
                  {statusText}
                </motion.span>
              )}
            </AnimatePresence>
            <button type="button" className="btn btn-accent nav-cta" onClick={() => go('studio')}>
              Start training
            </button>
            <button
              type="button"
              className={`menu-toggle ${menuOpen ? 'is-open' : ''}`}
              aria-expanded={menuOpen}
              aria-controls="mobile-menu"
              aria-label={menuOpen ? 'Close menu' : 'Open menu'}
              onClick={() => setMenuOpen((open) => !open)}
            >
              <span /><span />
            </button>
          </div>
        </div>
      </motion.header>

      <AnimatePresence>
        {menuOpen && (
          <motion.div
            id="mobile-menu"
            className="mobile-menu"
            initial={{ clipPath: 'inset(0 0 100% 0)' }}
            animate={{ clipPath: 'inset(0 0 0% 0)' }}
            exit={{ clipPath: 'inset(0 0 100% 0)' }}
            transition={{ duration: reduce ? 0 : 0.55, ease: [0.76, 0, 0.24, 1] }}
          >
            <nav aria-label="Sections">
              {LINKS.map(([id, text], i) => (
                <motion.a
                  key={id}
                  href={`#${id}`}
                  onClick={(e) => { e.preventDefault(); go(id); }}
                  initial={{ y: reduce ? 0 : 60, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.18 + i * 0.06, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                  {text}
                </motion.a>
              ))}
            </nav>
            <p className="mobile-menu-foot">{statusText}{running ? ` · ${totalReps} reps this session` : ''}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
