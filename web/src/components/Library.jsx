import { AnimatePresence, motion } from 'motion/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, mediaUrl } from '../lib/api.js';
import { label } from '../lib/actions.js';
import { scrollToId, useCoach } from '../lib/coach.jsx';
import { Reveal } from '../ui/motion.jsx';

const PER_PAGE = 24;
const CREDIT = '© Gym visual — gymvisual.com';

export default function Library() {
  const { health } = useCoach();
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({ category: '', equipment: '', target: '', rep_counting: false });
  const [options, setOptions] = useState({ categories: {}, equipment_types: {}, targets: {} });
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [status, setStatus] = useState('loading');
  const [openId, setOpenId] = useState(null);
  const debounced = useDebounced(query, 250);

  useEffect(() => {
    api.categories().then(setOptions).catch(() => {});
  }, []);

  const load = useCallback(async (nextPage, append) => {
    setStatus('loading');
    try {
      const data = await api.exercises({ ...filters, search: debounced, page: nextPage, per_page: PER_PAGE });
      setItems((prev) => (append ? [...prev, ...data.exercises] : data.exercises));
      setTotal(data.total);
      setPage(nextPage);
      setStatus('ready');
    } catch {
      setStatus('error');
    }
  }, [filters, debounced]);

  useEffect(() => { load(1, false); }, [load]);

  const setFilter = (key, value) => setFilters((f) => ({ ...f, [key]: value }));
  const hasMore = items.length < total;
  const exerciseCount = health.total_exercises || 1324;

  return (
    <section id="library" className="library">
      <div className="library-head">
        <Reveal as="h2" className="section-title">{exerciseCount.toLocaleString()} exercises to learn from.</Reveal>
        <Reveal as="p" className="section-lede" delay={0.08}>
          Every exercise comes with an animated demo, the muscles it works and step-by-step instructions. Pick one and train it in the studio. {health.rep_counting_supported ? `${health.rep_counting_supported} of them get counted reps.` : ''}
        </Reveal>
      </div>

      <div className="library-controls">
        <label className="search">
          <span className="visually-hidden">Search exercises</span>
          <input type="search" placeholder="Search: squat, curl, plank…" value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        <Select label="Body area" value={filters.category} onChange={(v) => setFilter('category', v)} options={options.categories} />
        <Select label="Equipment" value={filters.equipment} onChange={(v) => setFilter('equipment', v)} options={options.equipment_types} />
        <Select label="Muscle" value={filters.target} onChange={(v) => setFilter('target', v)} options={options.targets} />
        <label className="switch">
          <input type="checkbox" checked={filters.rep_counting} onChange={(e) => setFilter('rep_counting', e.target.checked)} />
          <span className="switch-track" aria-hidden="true" />
          Counted reps only
        </label>
      </div>

      <p className="library-count" aria-live="polite">
        {status === 'error' ? 'The exercise list could not be loaded. Is the server running?' : `${total.toLocaleString()} found`}
      </p>

      <motion.ul className="library-grid" layout>
        <AnimatePresence mode="popLayout">
          {items.map((ex, i) => (
            <motion.li
              key={ex.id}
              layout
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0, transition: { delay: (i % PER_PAGE) * 0.025 } }}
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <button type="button" className="ex-card" onClick={() => setOpenId(ex.id)}>
                <span className="ex-thumb">
                  <img src={mediaUrl(ex.image)} alt="" width="180" height="180" loading="lazy" />
                </span>
                <span className="ex-name">{ex.name}</span>
                <span className="ex-meta">
                  {ex.target}
                  {ex.equipment ? ` · ${ex.equipment}` : ''}
                </span>
                {ex.rep_action && <span className="ex-counted">Counts reps</span>}
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
      </motion.ul>

      {status === 'ready' && items.length === 0 && (
        <p className="library-empty">Nothing matches that. Try a shorter search or clear a filter.</p>
      )}

      <div className="library-foot">
        {hasMore && (
          <button type="button" className="btn btn-ink" disabled={status === 'loading'} onClick={() => load(page + 1, true)}>
            {status === 'loading' ? 'Loading…' : `Show ${Math.min(PER_PAGE, total - items.length)} more`}
          </button>
        )}
        <p className="fine-print">Exercise images and animations {CREDIT}, used with permission.</p>
      </div>

      <AnimatePresence>
        {openId && <ExerciseDrawer id={openId} onClose={() => setOpenId(null)} />}
      </AnimatePresence>
    </section>
  );
}

function Select({ label: text, value, onChange, options }) {
  return (
    <label className="select">
      <span className="visually-hidden">{text}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Any {text.toLowerCase()}</option>
        {Object.entries(options || {}).map(([name, count]) => (
          <option key={name} value={name}>{name} ({count})</option>
        ))}
      </select>
    </label>
  );
}

function ExerciseDrawer({ id, onClose }) {
  const { setSelectedExercise, chooseTarget, setStudioMode, requestCamera } = useCoach();
  const [exercise, setExercise] = useState(null);
  const [failed, setFailed] = useState(false);
  const closeRef = useRef(null);

  useEffect(() => {
    api.exercise(id).then((data) => setExercise(data.exercise)).catch(() => setFailed(true));
  }, [id]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    window.__lenis?.stop();
    return () => { window.removeEventListener('keydown', onKey); window.__lenis?.start(); };
  }, [onClose]);

  const train = (mode) => {
    setSelectedExercise(exercise);
    chooseTarget(exercise.rep_action || null);
    setStudioMode(mode);
    onClose();
    scrollToId('studio');
    if (mode === 'live') requestCamera();
  };

  const instructions = exercise?.instructions?.en || (typeof exercise?.instructions === 'string' ? exercise.instructions : '');
  const steps = instructions ? instructions.split(/(?<=\.)\s+/).filter(Boolean) : [];

  return (
    <>
      <motion.div className="drawer-scrim" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
      <motion.aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={exercise?.name || 'Exercise details'}
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', stiffness: 260, damping: 32 }}
        data-lenis-prevent
      >
        <button ref={closeRef} type="button" className="drawer-close" onClick={onClose} aria-label="Close">×</button>
        {failed && <p>That exercise could not be loaded.</p>}
        {!exercise && !failed && <p className="drawer-loading">Loading…</p>}
        {exercise && (
          <>
            <div className="drawer-media">
              <img src={mediaUrl(exercise.gif_url || exercise.image)} alt={`${exercise.name} demonstration`} width="180" height="180" />
              <span className="fine-print">{CREDIT}</span>
            </div>
            <h3 className="drawer-title">{exercise.name}</h3>
            <dl className="drawer-facts">
              <div><dt>Works</dt><dd>{exercise.target}</dd></div>
              <div><dt>Also</dt><dd>{(exercise.secondary_muscles || []).join(', ') || '—'}</dd></div>
              <div><dt>Equipment</dt><dd>{exercise.equipment}</dd></div>
              <div><dt>Reps</dt><dd>{exercise.rep_action ? `Counted as ${label(exercise.rep_action).toLowerCase()}` : 'Not counted yet'}</dd></div>
            </dl>
            {steps.length > 0 && (
              <ol className="drawer-steps">
                {steps.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
            )}
            <div className="drawer-actions">
              <button type="button" className="btn btn-accent" onClick={() => train('live')}>Train with camera</button>
              <button type="button" className="btn btn-ghost" onClick={() => train('video')}>Analyse a video</button>
            </div>
          </>
        )}
      </motion.aside>
    </>
  );
}

function useDebounced(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}
