#!/bin/sh
# Double-click on macOS (or run ./start.command on Linux) to start Daily Tracker
cd "$(dirname "$0")"
( sleep 1; open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null ) &
node server.js
