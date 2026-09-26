import React from 'react';
import { FlaskConical, History, BookOpen } from 'lucide-react';

/** Mobile tab bar: analyze, history, guide. */
const BottomNav = ({ active, onAnalyze, onHistory, onGuide, t }) => (
  <nav className="bottom-nav" aria-label={t.navAnalyze}>
    <button className={active === 'analyze' ? 'active' : ''} onClick={onAnalyze} aria-current={active === 'analyze' ? 'page' : undefined}>
      <FlaskConical size={24} aria-hidden="true" />{t.navAnalyze}
    </button>
    <button className={active === 'history' ? 'active' : ''} onClick={onHistory} aria-current={active === 'history' ? 'page' : undefined}>
      <History size={24} aria-hidden="true" />{t.navHistory}
    </button>
    <button onClick={onGuide}>
      <BookOpen size={24} aria-hidden="true" />{t.navGuide}
    </button>
  </nav>
);

export default BottomNav;
