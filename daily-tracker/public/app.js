// Daily Tracker - frontend
const API = '/api/items';
const TYPE_ICON = { work: '💼', project: '📁', schedule: '📅' };
const VIEW_TITLE = { today: 'Today', project: 'Projects', work: 'Work', schedule: 'Schedule', all: 'All items' };
const STATUSES = [['todo', 'To do'], ['doing', 'In progress'], ['done', 'Done']];

let items = [];
let view = 'today';
let current = null; // item open in the side page

const $ = id => document.getElementById(id);
const todayStr = () => new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD in local time

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.status === 204 ? null : res.json();
}

function visibleItems() {
  if (view === 'all') return items;
  if (view === 'today') return items.filter(i => i.date === todayStr() || (i.status === 'doing'));
  return items.filter(i => i.type === view);
}

function render() {
  $('view-title').textContent = VIEW_TITLE[view];
  const list = visibleItems().sort((a, b) => (a.date + a.time || '9').localeCompare(b.date + b.time || '9'));
  const cols = $('columns');
  cols.innerHTML = '';
  for (const [status, label] of STATUSES) {
    const group = list.filter(i => i.status === status);
    const col = document.createElement('div');
    col.className = 'column';
    col.innerHTML = `<h3>${label}<span class="count">${group.length}</span></h3>`;
    if (!group.length) col.insertAdjacentHTML('beforeend', '<div class="empty">Nothing here</div>');
    for (const item of group) {
      const card = document.createElement('div');
      card.className = 'card' + (item.status === 'done' ? ' done' : '');
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = `${TYPE_ICON[item.type] || ''} ${item.title}`;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = [item.date, item.time, item.notes ? '📝' : ''].filter(Boolean).join('  ·  ');
      card.append(title, meta);
      card.onclick = () => openPage(item);
      col.appendChild(card);
    }
    cols.appendChild(col);
  }
}

// ----- Side page (Notion-like editor) -----
function openPage(item) {
  current = item;
  $('page-title').value = item.title;
  $('page-type').value = item.type;
  $('page-status').value = item.status;
  $('page-date').value = item.date;
  $('page-time').value = item.time;
  $('page-notes').value = item.notes;
  $('save-state').textContent = '';
  $('page').classList.remove('hidden');
}

let saveTimer;
function scheduleSave() {
  if (!current) return;
  $('save-state').textContent = 'Saving...';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const changes = {
      title: $('page-title').value.trim() || 'Untitled',
      type: $('page-type').value,
      status: $('page-status').value,
      date: $('page-date').value,
      time: $('page-time').value,
      notes: $('page-notes').value,
    };
    const updated = await api('PUT', `${API}/${current.id}`, changes);
    Object.assign(current, updated);
    $('save-state').textContent = 'Saved ✓';
    render();
  }, 400);
}

['page-title', 'page-notes', 'page-date', 'page-time'].forEach(id => $(id).addEventListener('input', scheduleSave));
['page-type', 'page-status'].forEach(id => $(id).addEventListener('change', scheduleSave));

$('close-page').onclick = () => { $('page').classList.add('hidden'); current = null; };
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('close-page').click(); });

$('delete-item').onclick = async () => {
  if (!current || !confirm(`Delete "${current.title}"?`)) return;
  await api('DELETE', `${API}/${current.id}`);
  items = items.filter(i => i.id !== current.id);
  $('close-page').click();
  render();
};

// ----- New item -----
$('new-form').onsubmit = async e => {
  e.preventDefault();
  const item = await api('POST', API, {
    title: $('new-title').value.trim(),
    type: $('new-type').value,
    date: $('new-date').value || (view === 'today' ? todayStr() : ''),
    time: $('new-time').value,
  });
  items.push(item);
  $('new-title').value = '';
  $('new-time').value = '';
  render();
};

// ----- Navigation -----
$('nav').onclick = e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b === btn));
  view = btn.dataset.view;
  if (TYPE_ICON[view]) $('new-type').value = view;
  render();
};

$('today-date').textContent = new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });

(async () => { items = await api('GET', API); render(); })();
