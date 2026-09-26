import type { AnalysisRequest } from './api';

// Analyses entered without network are kept on the phone and sent when the
// connection comes back.
const KEY = 'agritech_pending_analyses';

export interface PendingAnalysis extends AnalysisRequest {
  id: number;
  savedAt: string;
}

export function readQueue(): PendingAnalysis[] {
  try {
    return JSON.parse(window.localStorage.getItem(KEY) || '[]');
  } catch {
    return [];
  }
}

function writeQueue(queue: PendingAnalysis[]) {
  try { window.localStorage.setItem(KEY, JSON.stringify(queue)); } catch {}
}

export function enqueue(request: AnalysisRequest): PendingAnalysis[] {
  const queue = [...readQueue(), { ...request, id: Date.now(), savedAt: new Date().toISOString() }];
  writeQueue(queue);
  return queue;
}

export function removeFromQueue(id: number): PendingAnalysis[] {
  const queue = readQueue().filter(item => item.id !== id);
  writeQueue(queue);
  return queue;
}
