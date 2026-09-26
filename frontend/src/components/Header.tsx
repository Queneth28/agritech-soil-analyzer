import React from 'react';
import { History, Sun, Moon, HelpCircle, Sprout, Languages } from 'lucide-react';

const Header = ({ lang, setLang, theme, setTheme, showHistory, setShowHistory, historyCount = 0, onShowHelp = () => {}, t }) => (
  <header className="header" role="banner">
    <div className="header-content">
      <div className="header-icon" aria-hidden="true"><Sprout size={22} /></div>
      <div className="header-title">
        <h1>{t.appTitle}</h1>
        <p className="header-subtitle">{t.appSubtitle}</p>
      </div>
      <nav className="header-actions" aria-label="App">
        <button onClick={onShowHelp} className="icon-btn" aria-label={t.helpTitle} title={t.helpTitle}>
          <HelpCircle size={20} />
        </button>
        <button onClick={() => setShowHistory(!showHistory)} className={`icon-btn ${showHistory ? 'active' : ''}`}
          aria-expanded={showHistory} aria-label={showHistory ? t.hideHistory : t.viewHistory}
          title={showHistory ? t.hideHistory : t.viewHistory}>
          <History size={20} />
          {historyCount > 0 && <span className="history-badge" aria-hidden="true">{historyCount > 99 ? '99+' : historyCount}</span>}
        </button>
        <button onClick={() => setTheme(p => p === 'dark' ? 'light' : 'dark')} className="icon-btn"
          aria-label={theme === 'dark' ? t.themeLight : t.themeDark} title={theme === 'dark' ? t.themeLight : t.themeDark}>
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
        </button>
        <button onClick={() => setLang(p => p === 'en' ? 'fr' : 'en')} className="lang-toggle"
          aria-label={lang === 'en' ? 'Passer en français' : 'Switch to English'}>
          <Languages size={16} aria-hidden="true" /><span>{lang === 'en' ? 'FR' : 'EN'}</span>
        </button>
      </nav>
    </div>
  </header>
);

export default Header;
