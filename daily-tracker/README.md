# 📓 Daily Tracker

A really simple Notion-like tracker for your daily **projects**, **work** and **schedule**.

- **Frontend:** plain HTML + JavaScript (`public/`)
- **Backend:** Node.js with no dependencies (`server.js`)
- **Storage:** a local `data.json` file, created automatically

## Run it

You need [Node.js](https://nodejs.org) 18 or newer.

```bash
cd daily-tracker
npm start
```

Then open http://localhost:3000

## How to use

- Type a title in **+ New item**, pick a type (Work / Project / Schedule), optional date and time, then press Enter.
- The sidebar filters by **Today** (items dated today or in progress), Projects, Work, Schedule or All.
- Items are shown on a board: **To do / In progress / Done**.
- Click any card to open its page: edit the title, type, status, date, time and write notes. Changes save automatically.
- `Esc` closes the page.

## API

| Method | URL               | Description     |
|--------|-------------------|-----------------|
| GET    | `/api/items`      | List all items  |
| POST   | `/api/items`      | Create an item  |
| PUT    | `/api/items/:id`  | Update an item  |
| DELETE | `/api/items/:id`  | Delete an item  |
