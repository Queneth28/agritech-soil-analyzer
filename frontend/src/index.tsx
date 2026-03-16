import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// ============================================================================
// SERVICE WORKER REGISTRATION (PWA Support)
// ============================================================================

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/service-worker.js')
      .then((registration) => {
        console.log('✅ ServiceWorker registered successfully:', registration);
      })
      .catch((error) => {
        console.log('❌ ServiceWorker registration failed:', error);
      });
  });
}

// ============================================================================
// REACT APP RENDERING
// ============================================================================

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);