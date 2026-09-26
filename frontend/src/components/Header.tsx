import React from 'react';
import { History, Sun, Moon, BookOpen } from 'lucide-react';
import { fmt } from '../constants/translations';

export const BrandMark = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 21V11" /><path d="M12 11c0-4 3-6 7-6 0 4-3 6-7 6z" /><path d="M12 14c0-3-2.5-5-6-5 0 3 2.5 5 6 5z" />
  </svg>
);

const Header = ({ lang, setLang, theme, setTheme, showHistory, setShowHistory, historyCount = 0,
  onShowHelp = () => {}, online = true, pendingCount = 0, t }) => (
  <header className="header" role="banner">
    <div className="header-content">
      <div className="header-icon" aria-hidden="true"><BrandMark /></div>
      <div className="header-title"><h1>{t.brand}</h1></div>

      {(!online || pendingCount > 0) && (
        <div className="status-pill" role="status">
          <span className="status-dot" aria-hidden="true" />
          <span>{!online ? t.offline : ''}{!online && pendingCount > 0 ? ' · ' : ''}
            {pendingCount > 0 && <span className="status-text">{fmt(t.pendingCount, { n: pendingCount })}</span>}
          </span>
        </div>
      )}

      <nav className="header-actions" aria-label="App">
        <button onClick={() => setShowHistory(!showHistory)} className={`header-link ${showHistory ? 'active' : ''}`} aria-expanded={showHistory}>
          <History size={20} aria-hidden="true" />{t.navHistory}
          {historyCount > 0 && <span className="history-badge">{historyCount > 99 ? '99+' : historyCount}</span>}
        </button>
        <button onClick={onShowHelp} className="header-link">
          <BookOpen size={20} aria-hidden="true" />{t.navGuide}
        </button>
        <button onClick={() => setTheme(p => p === 'dark' ? 'light' : 'dark')} className="icon-btn"
          aria-label={theme === 'dark' ? t.themeLight : t.themeDark} title={theme === 'dark' ? t.themeLight : t.themeDark}>
          {theme === 'dark' ? <Sun size={22} /> : <Moon size={22} />}
        </button>
        <button onClick={() => setLang(p => p === 'en' ? 'fr' : 'en')} className="lang-toggle"
          aria-label={lang === 'en' ? 'Passer en français' : 'Switch to English'}>
          <span>{lang === 'en' ? 'FR' : 'EN'}</span>
        </button>
      </nav>
    </div>
  </header>
);

export default Header;
