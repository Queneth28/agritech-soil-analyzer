import React from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

const ToastContainer = ({ toasts, removeToast }) => (
  <div className="toast-container" role="alert" aria-live="polite">
    {toasts.map(toast => (
      <div key={toast.id} className={`toast toast-${toast.type}`}>
        <div className="toast-content">
          {toast.type === 'success' && <CheckCircle2 size={18} />}
          {toast.type === 'error' && <AlertCircle size={18} />}
          {toast.type === 'info' && <Info size={18} />}
          <span>{toast.message}</span>
        </div>
        <button onClick={() => removeToast(toast.id)} className="toast-close" aria-label="Dismiss">
          <X size={14} />
        </button>
      </div>
    ))}
  </div>
);

export default ToastContainer;
