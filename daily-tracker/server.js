// Daily Tracker - tiny Node.js backend (no dependencies)
const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = process.env.PORT || 3000;
const DATA_FILE = path.join(__dirname, 'data.json');
const PUBLIC_DIR = path.join(__dirname, 'public');

function load() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')); }
  catch { return []; }
}
function save(items) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(items, null, 2));
}

function send(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(body === undefined ? '' : JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', c => (data += c));
    req.on('end', () => { try { resolve(data ? JSON.parse(data) : {}); } catch (e) { reject(e); } });
  });
}

const FIELDS = ['title', 'type', 'status', 'date', 'time', 'notes'];
function pick(obj) {
  const out = {};
  for (const f of FIELDS) if (obj[f] !== undefined) out[f] = String(obj[f]);
  return out;
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const match = url.pathname.match(/^\/api\/items(?:\/([\w-]+))?$/);

  if (match) {
    const id = match[1];
    let items = load();
    try {
      if (req.method === 'GET' && !id) return send(res, 200, items);
      if (req.method === 'POST' && !id) {
        const body = pick(await readBody(req));
        if (!body.title) return send(res, 400, { error: 'title is required' });
        const item = { id: crypto.randomUUID(), type: 'work', status: 'todo', date: '', time: '', notes: '',
          ...body, createdAt: new Date().toISOString() };
        items.push(item); save(items);
        return send(res, 201, item);
      }
      const idx = items.findIndex(i => i.id === id);
      if (idx === -1) return send(res, 404, { error: 'not found' });
      if (req.method === 'PUT') {
        items[idx] = { ...items[idx], ...pick(await readBody(req)) };
        save(items);
        return send(res, 200, items[idx]);
      }
      if (req.method === 'DELETE') {
        items.splice(idx, 1); save(items);
        return send(res, 204);
      }
      return send(res, 405, { error: 'method not allowed' });
    } catch {
      return send(res, 400, { error: 'invalid JSON' });
    }
  }

  // Static files
  const file = path.normalize(path.join(PUBLIC_DIR, url.pathname === '/' ? 'index.html' : url.pathname));
  if (!file.startsWith(PUBLIC_DIR)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

server.listen(PORT, () => console.log(`Daily Tracker running at http://localhost:${PORT}`));
