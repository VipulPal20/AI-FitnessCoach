import { animate, motion, useInView, useMotionValue, useReducedMotion, useSpring } from 'motion/react';
import { Fragment, useEffect, useRef, useState } from 'react';

const EASE = [0.22, 1, 0.36, 1];

/** Fade-and-rise entrance, triggered once when scrolled into view. */
export function Reveal({ as = 'div', delay = 0, y = 24, children, className, ...rest }) {
  const reduce = useReducedMotion();
  const Tag = motion[as];
  return (
    <Tag
      className={className}
      initial={{ opacity: 0, y: reduce ? 0 : y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '0px 0px -12% 0px' }}
      transition={{ duration: 0.7, ease: EASE, delay }}
      {...rest}
    >
      {children}
    </Tag>
  );
}

/** Staggered children: wrap items in <Stagger.Item>. */
export function Stagger({ as = 'div', gap = 0.075, children, className, ...rest }) {
  const Tag = motion[as];
  return (
    <Tag
      className={className}
      initial="hidden"
      whileInView="shown"
      viewport={{ once: true, margin: '0px 0px -10% 0px' }}
      variants={{ shown: { transition: { staggerChildren: gap } } }}
      {...rest}
    >
      {children}
    </Tag>
  );
}

Stagger.Item = function StaggerItem({ as = 'div', y = 24, children, className, ...rest }) {
  const reduce = useReducedMotion();
  const Tag = motion[as];
  return (
    <Tag
      className={className}
      variants={{
        hidden: { opacity: 0, y: reduce ? 0 : y },
        shown: { opacity: 1, y: 0, transition: { duration: 0.65, ease: EASE } },
      }}
      {...rest}
    >
      {children}
    </Tag>
  );
};

/** Headline that rises in word by word (in the spirit of React Bits' SplitText). */
export function SplitHeadline({ text, as = 'h1', className, delay = 0 }) {
  const reduce = useReducedMotion();
  const Tag = motion[as];
  const lines = text.split('\n');
  let index = 0;
  return (
    <Tag className={className} aria-label={text.replace(/\n/g, ' ')}>
      {lines.map((line, li) => (
        <span className="split-line" key={li} aria-hidden="true">
          {line.split(' ').map((word, wi, words) => {
            const i = index++;
            // The space sits between masks, not inside one, so it isn't
            // clipped away and the line can still wrap between words.
            return (
              <Fragment key={wi}>
                <span className="split-mask">
                  <motion.span
                    className="split-word"
                    initial={{ y: reduce ? 0 : '105%', opacity: reduce ? 0 : 1 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.9, ease: EASE, delay: delay + i * 0.07 }}
                  >
                    {word}
                  </motion.span>
                </span>
                {wi < words.length - 1 ? ' ' : null}
              </Fragment>
            );
          })}
        </span>
      ))}
    </Tag>
  );
}

/** Number that counts up when it first scrolls into view. */
export function CountUp({ to, decimals = 0, duration = 1.6, suffix = '', className }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '0px 0px -15% 0px' });
  const reduce = useReducedMotion();
  const [value, setValue] = useState(reduce ? to : 0);
  useEffect(() => {
    if (!inView || reduce) { setValue(to); return undefined; }
    const controls = animate(0, to, { duration, ease: EASE, onUpdate: setValue });
    return () => controls.stop();
  }, [inView, to, duration, reduce]);
  return (
    <span ref={ref} className={className}>
      {value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
      {suffix}
    </span>
  );
}

/** Live number that springs to each new value (used for rep counts). */
export function SpringNumber({ value, className }) {
  const mv = useMotionValue(value);
  const spring = useSpring(mv, { stiffness: 220, damping: 22 });
  const [shown, setShown] = useState(value);
  useEffect(() => { mv.set(value); }, [value, mv]);
  useEffect(() => spring.on('change', (v) => setShown(Math.round(v))), [spring]);
  return <span className={className}>{shown}</span>;
}

/** Button that leans toward the cursor (React Bits "Magnet" style). */
export function Magnetic({ children, strength = 0.28, className, ...rest }) {
  const ref = useRef(null);
  const reduce = useReducedMotion();
  const x = useSpring(0, { stiffness: 260, damping: 18 });
  const y = useSpring(0, { stiffness: 260, damping: 18 });
  const onMove = (event) => {
    if (reduce || !ref.current) return;
    const box = ref.current.getBoundingClientRect();
    x.set((event.clientX - (box.left + box.width / 2)) * strength);
    y.set((event.clientY - (box.top + box.height / 2)) * strength);
  };
  const onLeave = () => { x.set(0); y.set(0); };
  return (
    <motion.span ref={ref} className={className} style={{ x, y, display: 'inline-block' }} onPointerMove={onMove} onPointerLeave={onLeave} {...rest}>
      {children}
    </motion.span>
  );
}
