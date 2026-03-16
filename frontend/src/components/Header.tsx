import React from 'react';
import { Globe, History, Sun, Moon, HelpCircle } from 'lucide-react';

const Header = ({ lang, setLang, theme, setTheme, showHistory, setShowHistory, historyCount, onShowHelp, t }) => (
  <header className="header" role="banner">
    <div className="header-content">
      <div className="header-icon" aria-hidden="true">
        <img src="/app-icon.svg" alt="" style={{ width: '40px', height: '40px', borderRadius: '10px', display: 'block' }}
          onError={(e) => {
            const img = e.target as HTMLImageElement;
            img.src = '/app-icon.png';
          }} />
      </div>
      <div className="header-title">
        <h1>{t.appTitle}</h1>
        <p className="header-subtitle">{t.appSubtitle}</p>
      </div>
      <div className="header-actions">
        <button onClick={onShowHelp} className="header-btn" aria-label={t.helpTitle}>
          <HelpCircle size={18} /><span className="btn-text">{t.helpTitle}</span>
        </button>
        <button onClick={() => setTheme(p => p === 'dark' ? 'light' : 'dark')} className="header-btn"
          aria-label={theme === 'dark' ? t.themeLight : t.themeDark}>
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
        <button onClick={() => setShowHistory(!showHistory)} className={`header-btn ${showHistory ? 'active' : ''}`} aria-expanded={showHistory}>
          <History size={18} /><span className="btn-text">{showHistory ? t.hideHistory : t.viewHistory}</span>
          {historyCount > 0 && <span className="history-badge">{historyCount}</span>}
        </button>
        <button onClick={() => setLang(p => p === 'en' ? 'fr' : 'en')} className="header-btn accent">
          <Globe size={18} /><span className="btn-text">{lang === 'en' ? 'FR' : 'EN'}</span>
        </button>
      </div>
    </div>
  </header>
);

export default Header;
