import { scrollToId } from '../lib/coach.jsx';
import { Magnetic, Reveal } from '../ui/motion.jsx';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-close">
        <Reveal as="p" className="footer-line">Clear some floor. Your first set is one click away.</Reveal>
        <Magnetic>
          <button type="button" className="btn btn-accent btn-lg" onClick={() => scrollToId('studio')}>
            Open the studio <span className="btn-arrow" aria-hidden="true">→</span>
          </button>
        </Magnetic>
      </div>

      <div className="footer-wordmark" aria-hidden="true">FitnessCoach</div>

      <div className="footer-meta">
        <p>
          A BTech final-year minor project. Pose detection by Google MediaPipe. Movement classifier and rep counter trained and built for this project.
        </p>
        <p>
          Training footage from MuscleWiki, via the JIM_DATA29 dataset. Exercise library media © Gym visual (gymvisual.com), used with permission.
        </p>
        <nav aria-label="Footer">
          <a href="/docs">API documentation</a>
          <button type="button" className="text-link" onClick={() => scrollToId('top')}>Back to top</button>
        </nav>
      </div>
    </footer>
  );
}
