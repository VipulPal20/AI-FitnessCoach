import { CoachProvider } from './lib/coach.jsx';
import Navbar from './components/Navbar.jsx';
import Hero from './components/Hero.jsx';
import HowItWorks from './components/HowItWorks.jsx';
import Studio from './components/studio/Studio.jsx';
import Session from './components/Session.jsx';
import Library from './components/Library.jsx';
import ModelNotes from './components/ModelNotes.jsx';
import Footer from './components/Footer.jsx';

export default function App() {
  return (
    <CoachProvider>
      <Navbar />
      <main>
        <Hero />
        <HowItWorks />
        <Studio />
        <Session />
        <Library />
        <ModelNotes />
      </main>
      <Footer />
    </CoachProvider>
  );
}
