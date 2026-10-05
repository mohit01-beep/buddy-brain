import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const DATA_DIR = path.join(__dirname, 'data');
const TASKS_FILE = path.join(DATA_DIR, 'tasks.json');
const MEMORIES_FILE = path.join(DATA_DIR, 'memories.json');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Ensure storage files exist
const initFile = (filePath, defaultContent = []) => {
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, JSON.stringify(defaultContent, null, 2), 'utf-8');
  }
};
initFile(TASKS_FILE, []);
initFile(MEMORIES_FILE, []);

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Helpers for data reading and writing
const readData = (filePath) => {
  try {
    const raw = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(raw);
  } catch (err) {
    console.error(`Error reading ${filePath}:`, err);
    return [];
  }
};

const writeData = (filePath, data) => {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
};

// API: Tasks
app.get('/api/tasks', (req, res) => {
  res.json(readData(TASKS_FILE));
});

app.post('/api/tasks', (req, res) => {
  const tasks = req.body;
  writeData(TASKS_FILE, tasks);
  res.json({ success: true, count: tasks.length });
});

// API: Memories / Notes
app.get('/api/memories', (req, res) => {
  res.json(readData(MEMORIES_FILE));
});

app.post('/api/memories', (req, res) => {
  const memories = req.body;
  writeData(MEMORIES_FILE, memories);
  res.json({ success: true, count: memories.length });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'online',
    appName: 'BuddyBrain',
    version: '1.0.0',
    mode: '100% Private Local Open-Source AI',
    timestamp: new Date().toISOString()
  });
});

app.listen(PORT, () => {
  console.log(`\n======================================================`);
  console.log(`🚀 BuddyBrain Local Open-Source AI Companion is running!`);
  console.log(`👉 Open your browser at: http://localhost:${PORT}`);
  console.log(`🔒 All data & AI inference run strictly on your machine.`);
  console.log(`======================================================\n`);
});
