const API_BASE = process.env.REACT_APP_API_URL || `http://${window.location.hostname}:5000`;

export const API_ENDPOINTS = {
  predict: `${API_BASE}/api/predict`,
  crops: `${API_BASE}/api/crops`,
  health: `${API_BASE}/api/health`,
  modelInfo: `${API_BASE}/api/model/info`,
  compare: `${API_BASE}/api/compare`,
  seasonalCalendar: `${API_BASE}/api/seasonal-calendar`,
  soilHealthScore: `${API_BASE}/api/soil-health-score`,
  history: `${API_BASE}/api/history`,
};

export async function analyzeSoil(soilData) {
  const response = await fetch(API_ENDPOINTS.predict, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(soilData),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `API Error: ${response.status}`);
  }
  return response.json();
}

export async function fetchHistory(limit = 20) {
  const response = await fetch(`${API_ENDPOINTS.history}?limit=${limit}`);
  if (!response.ok) throw new Error('Failed to fetch history');
  return response.json();
}

export async function deleteHistoryItem(id) {
  const response = await fetch(`${API_ENDPOINTS.history}/${id}`, { method: 'DELETE' });
  if (!response.ok) throw new Error('Failed to delete history item');
  return response.json();
}
