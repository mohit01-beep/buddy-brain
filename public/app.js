// BuddyBrain — Client Application with Transformers.js On-Device Open-Source AI
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure transformers.js environment for browser
env.allowLocalModels = false;
env.useBrowserCache = true;

// App State
let tasks = [];
let memories = [];
let sentimentClassifier = null;
let featureExtractor = null;
let isAiReady = false;

// DOM Elements
const aiStatusBadge = document.getElementById('aiStatusBadge');
const aiStatusDot = document.getElementById('aiStatusDot');
const aiStatusText = document.getElementById('aiStatusText');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const tabButtons = document.querySelectorAll('.tab-btn');
const quickFillDemoBtn = document.getElementById('quickFillDemoBtn');

// Deconstructor Elements
const brainDumpInput = document.getElementById('brainDumpInput');
const processDumpBtn = document.getElementById('processDumpBtn');
const clearDumpBtn = document.getElementById('clearDumpBtn');
const tasksList = document.getElementById('tasksList');
const taskCountBadge = document.getElementById('taskCountBadge');
const taskProgressSection = document.getElementById('taskProgressSection');
const progressPercent = document.getElementById('progressPercent');
const clearCompletedTasksBtn = document.getElementById('clearCompletedTasksBtn');
const sentimentCard = document.getElementById('sentimentCard');
const sentimentLabel = document.getElementById('sentimentLabel');
const sentimentScore = document.getElementById('sentimentScore');
const sentimentBar = document.getElementById('sentimentBar');
const sentimentAdvice = document.getElementById('sentimentAdvice');

// Memory Elements
const memoryTagInput = document.getElementById('memoryTagInput');
const memoryContentInput = document.getElementById('memoryContentInput');
const saveMemoryBtn = document.getElementById('saveMemoryBtn');
const searchMemoryQuery = document.getElementById('searchMemoryQuery');
const searchMemoryBtn = document.getElementById('searchMemoryBtn');
const semanticSearchResults = document.getElementById('semanticSearchResults');
const memoryCardsGrid = document.getElementById('memoryCardsGrid');
const totalMemoriesCount = document.getElementById('totalMemoriesCount');
const seedMemoriesBtn = document.getElementById('seedMemoriesBtn');

// Companion Elements
const buddyToneSelect = document.getElementById('buddyToneSelect');
const generateBuddyPepTalkBtn = document.getElementById('generateBuddyPepTalkBtn');
const buddyToneDisplay = document.getElementById('buddyToneDisplay');
const buddyMessageOutput = document.getElementById('buddyMessageOutput');

// Initialize Open-Source AI Pipelines (In-Browser WebAssembly/ONNX)
async function initAi() {
  try {
    updateAiStatus('loading', 'Loading open-source AI models...');
    
    // 1. Sentiment Pipeline (distilbert-base-uncased-finetuned-sst-2-english)
    sentimentClassifier = await pipeline(
      'sentiment-analysis',
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
      {
        progress_callback: (p) => {
          if (p.status === 'progress') {
            updateAiStatus('loading', `Loading AI models: ${Math.round(p.progress || 0)}%`);
          }
        }
      }
    );

    // 2. Feature Extractor / Embedding Pipeline (all-MiniLM-L6-v2) for Semantic Search
    featureExtractor = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
      { quantized: true }
    );

    isAiReady = true;
    updateAiStatus('ready', 'Open-Source AI Ready (100% On-Device)');
  } catch (err) {
    console.warn('AI pipeline fallback mode active:', err);
    updateAiStatus('fallback', 'AI Local Heuristics Active');
  }
}

function updateAiStatus(status, text) {
  if (!aiStatusBadge) return;
  aiStatusBadge.classList.remove('hidden');
  aiStatusText.textContent = text;
  
  aiStatusDot.className = 'w-2 h-2 rounded-full';
  if (status === 'ready') {
    aiStatusDot.classList.add('bg-emerald-500');
  } else if (status === 'loading') {
    aiStatusDot.classList.add('bg-amber-400', 'animate-ping');
  } else {
    aiStatusDot.classList.add('bg-teal-400');
  }
}

// Vector Math: Cosine Similarity for Semantic Search
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Extract Vector Embedding for Text
async function getEmbedding(text) {
  if (featureExtractor) {
    try {
      const output = await featureExtractor(text, { pooling: 'mean', normalize: true });
      return Array.from(output.data);
    } catch (e) {
      console.error('Embedding error:', e);
    }
  }
  // Fallback simple word-frequency pseudo-embedding
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  const hashVec = new Array(64).fill(0);
  words.forEach((w, idx) => {
    const code = w.charCodeAt(0) % 64;
    hashVec[code] += 1 / (idx + 1);
  });
  return hashVec;
}

// Task Deconstruction Engine (Natural Language Chunking & Micro-Step Heuristic)
function deconstructBrainDump(rawText) {
  if (!rawText.trim()) return [];

  // Split clauses and sentences into digestible thoughts
  const rawChunks = rawText
    .split(/(?:\. |\n+|; | and then | then | and I need to | also | but I have to |, so )/i)
    .map(c => c.trim().replace(/^[-*•\d.]+\s*/, ''))
    .filter(c => c.length > 4);

  const cleanTasks = [];

  rawChunks.forEach((chunk, index) => {
    let title = chunk;
    let category = 'focus';
    let estMinutes = 10;
    let urgency = 'medium';

    // Heuristics for ADHD-friendly micro-breakdown
    const lower = chunk.toLowerCase();
    if (lower.includes('email') || lower.includes('reply') || lower.includes('message') || lower.includes('sarah')) {
      category = 'communication';
      estMinutes = 5;
      urgency = 'high';
      title = `Draft quick reply: "${chunk.slice(0, 40)}..."`;
    } else if (lower.includes('code') || lower.includes('bug') || lower.includes('test') || lower.includes('refactor')) {
      category = 'deep-work';
      estMinutes = 15;
      urgency = 'high';
      title = `Isolate 1 failing test / error in: ${chunk.slice(0, 35)}`;
    } else if (lower.includes('clean') || lower.includes('desk') || lower.includes('room') || lower.includes('eat') || lower.includes('lunch') || lower.includes('water')) {
      category = 'wellness';
      estMinutes = 5;
      urgency = 'immediate';
      title = `Self-care reset: ${chunk}`;
    } else if (lower.includes('presentation') || lower.includes('study') || lower.includes('exam') || lower.includes('slides')) {
      category = 'prep';
      estMinutes = 20;
      urgency = 'medium';
      title = `Outline 3 key bullet points for: ${chunk.slice(0, 35)}`;
    } else {
      title = `Take the 1st step on: ${chunk}`;
      estMinutes = 10;
    }

    cleanTasks.push({
      id: Date.now() + index,
      title,
      originalThought: chunk,
      category,
      estMinutes,
      urgency,
      completed: false,
      createdAt: new Date().toISOString()
    });
  });

  return cleanTasks;
}

// Analyze Emotional Temperature with Sentiment Classifier
async function analyzeSentiment(text) {
  let sentiment = 'neutral';
  let score = 0.5;

  if (sentimentClassifier) {
    try {
      const result = await sentimentClassifier(text.slice(0, 512));
      if (result && result.length > 0) {
        sentiment = result[0].label.toLowerCase(); // 'positive' or 'negative'
        score = result[0].score;
      }
    } catch (e) {
      console.error('Sentiment inference error:', e);
    }
  } else {
    // Heuristic sentiment check
    const negWords = ['stressed', 'overwhelmed', 'failing', 'messy', 'tired', 'hate', 'exhausted', 'anxious', 'stuck', 'panic'];
    const posWords = ['great', 'excited', 'ready', 'happy', 'solved', 'love', 'easy', 'confident', 'done'];
    const lower = text.toLowerCase();
    const negCount = negWords.filter(w => lower.includes(w)).length;
    const posCount = posWords.filter(w => lower.includes(w)).length;
    
    if (negCount > posCount) {
      sentiment = 'negative';
      score = Math.min(0.95, 0.6 + negCount * 0.1);
    } else if (posCount > negCount) {
      sentiment = 'positive';
      score = Math.min(0.95, 0.6 + posCount * 0.1);
    }
  }

  return { sentiment, score };
}

// Render Tasks
function renderTasks() {
  if (tasks.length === 0) {
    tasksList.innerHTML = `
      <div class="flex flex-col items-center justify-center h-48 text-center p-6 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl">
        <i class="ph ph-sparkle text-3xl text-slate-300 dark:text-slate-600 mb-2"></i>
        <p class="text-sm font-medium text-slate-500 dark:text-slate-400">No active tasks yet</p>
        <p class="text-xs text-slate-400 mt-1 max-w-sm">Type a brain dump on the left or click "Load Sample Brain Dump" to see open-source AI turn chaos into simple micro-steps!</p>
      </div>
    `;
    taskCountBadge.textContent = `0 Tasks`;
    taskProgressSection.classList.add('hidden');
    return;
  }

  taskCountBadge.textContent = `${tasks.length} Tasks`;
  taskProgressSection.classList.remove('hidden');

  const completedCount = tasks.filter(t => t.completed).length;
  const pct = Math.round((completedCount / tasks.length) * 100);
  progressPercent.textContent = `${pct}% (${completedCount}/${tasks.length})`;

  tasksList.innerHTML = tasks.map(task => {
    const badgeColor = 
      task.category === 'wellness' ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border-emerald-300' :
      task.category === 'communication' ? 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300 border-blue-300' :
      task.category === 'deep-work' ? 'bg-purple-100 dark:bg-purple-950 text-purple-700 dark:text-purple-300 border-purple-300' :
      'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 border-amber-300';

    return `
      <div class="p-3.5 rounded-xl border transition flex items-start justify-between gap-3 ${task.completed ? 'bg-slate-100/60 dark:bg-slate-800/40 border-slate-200 dark:border-slate-800 opacity-60' : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 shadow-sm'}">
        <div class="flex items-start space-x-3 flex-1">
          <input type="checkbox" data-id="${task.id}" class="task-checkbox mt-1 w-4 h-4 rounded text-emerald-600 focus:ring-emerald-500 border-slate-300 cursor-pointer" ${task.completed ? 'checked' : ''} />
          <div class="space-y-1">
            <p class="text-sm font-medium ${task.completed ? 'line-through text-slate-400 dark:text-slate-500' : 'text-slate-800 dark:text-slate-100'}">${escapeHtml(task.title)}</p>
            <div class="flex items-center space-x-2 text-[11px] text-slate-400">
              <span class="px-2 py-0.5 rounded border text-[10px] font-semibold uppercase ${badgeColor}">${task.category}</span>
              <span><i class="ph ph-clock text-slate-400 mr-0.5"></i>~${task.estMinutes} mins</span>
            </div>
          </div>
        </div>
        <button data-id="${task.id}" class="delete-task-btn text-slate-400 hover:text-rose-500 transition p-1">
          <i class="ph ph-trash"></i>
        </button>
      </div>
    `;
  }).join('');

  // Attach event listeners for checkbox and delete
  document.querySelectorAll('.task-checkbox').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const id = Number(e.target.getAttribute('data-id'));
      const t = tasks.find(item => item.id === id);
      if (t) {
        t.completed = e.target.checked;
        saveTasksToStorage();
        renderTasks();
      }
    });
  });

  document.querySelectorAll('.delete-task-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = Number(e.currentTarget.getAttribute('data-id'));
      tasks = tasks.filter(t => t.id !== id);
      saveTasksToStorage();
      renderTasks();
    });
  });
}

// Storage Operations (Local backend + LocalStorage sync)
async function loadData() {
  try {
    const [tRes, mRes] = await Promise.all([
      fetch('/api/tasks'),
      fetch('/api/memories')
    ]);
    if (tRes.ok) tasks = await tRes.json();
    if (mRes.ok) memories = await mRes.json();
  } catch (err) {
    // Fallback to localStorage if server endpoint unavailable
    tasks = JSON.parse(localStorage.getItem('buddybrain_tasks') || '[]');
    memories = JSON.parse(localStorage.getItem('buddybrain_memories') || '[]');
  }
  renderTasks();
  renderMemories();
}

async function saveTasksToStorage() {
  localStorage.setItem('buddybrain_tasks', JSON.stringify(tasks));
  try {
    await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tasks)
    });
  } catch (e) {
    console.warn('Could not sync tasks to local file backend', e);
  }
}

async function saveMemoriesToStorage() {
  localStorage.setItem('buddybrain_memories', JSON.stringify(memories));
  try {
    await fetch('/api/memories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(memories)
    });
  } catch (e) {
    console.warn('Could not sync memories to local file backend', e);
  }
}

// Render Memories
function renderMemories() {
  totalMemoriesCount.textContent = memories.length;

  if (memories.length === 0) {
    memoryCardsGrid.innerHTML = `
      <div class="col-span-full text-center py-8 text-slate-400 text-xs border border-dashed border-slate-200 dark:border-slate-800 rounded-xl">
        No notes stored yet. Click "Insert Sample Notes" to load helpful memories.
      </div>
    `;
    return;
  }

  memoryCardsGrid.innerHTML = memories.map(mem => `
    <div class="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/80 shadow-sm flex flex-col justify-between space-y-2">
      <div class="space-y-1.5">
        <span class="inline-block px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950/80 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800">
          ${escapeHtml(mem.tag || '#note')}
        </span>
        <p class="text-xs text-slate-700 dark:text-slate-200 leading-relaxed">${escapeHtml(mem.content)}</p>
      </div>
      <div class="flex items-center justify-between text-[10px] text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-800">
        <span>${new Date(mem.createdAt).toLocaleDateString()}</span>
        <button data-id="${mem.id}" class="delete-memory-btn hover:text-rose-500 transition">
          <i class="ph ph-trash text-sm"></i>
        </button>
      </div>
    </div>
  `).join('');

  document.querySelectorAll('.delete-memory-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = Number(e.currentTarget.getAttribute('data-id'));
      memories = memories.filter(m => m.id !== id);
      saveMemoriesToStorage();
      renderMemories();
    });
  });
}

// Perform Semantic Search Across Local Memories
async function performSemanticSearch(query) {
  if (!query.trim()) {
    semanticSearchResults.innerHTML = `<div class="text-slate-400 text-center py-4 italic">Please enter a search phrase.</div>`;
    return;
  }

  if (memories.length === 0) {
    semanticSearchResults.innerHTML = `<div class="text-slate-400 text-center py-4">No notes saved in database to match against.</div>`;
    return;
  }

  semanticSearchResults.innerHTML = `<div class="text-indigo-500 text-center py-4 animate-pulse"><i class="ph ph-spinner animate-spin mr-1"></i>Computing on-device vector similarity...</div>`;

  const queryEmbedding = await getEmbedding(query);

  const scored = memories.map(mem => {
    let score = 0;
    if (mem.embedding && mem.embedding.length > 0) {
      score = cosineSimilarity(queryEmbedding, mem.embedding);
    } else {
      // Direct string matching if no embedding
      const match = mem.content.toLowerCase().includes(query.toLowerCase());
      score = match ? 0.8 : 0.1;
    }
    return { ...mem, score };
  });

  // Sort by highest similarity
  scored.sort((a, b) => b.score - a.score);
  const topMatches = scored.slice(0, 3);

  semanticSearchResults.innerHTML = topMatches.map(m => `
    <div class="p-2.5 rounded-lg border border-indigo-100 dark:border-indigo-900/50 bg-indigo-50/40 dark:bg-indigo-950/30 flex items-start justify-between gap-2">
      <div class="space-y-1">
        <div class="flex items-center space-x-1.5">
          <span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-200 dark:bg-indigo-900 text-indigo-800 dark:text-indigo-300 font-bold">${m.tag || '#note'}</span>
          <span class="text-[10px] text-emerald-600 dark:text-emerald-400 font-mono font-semibold">${Math.round(m.score * 100)}% match</span>
        </div>
        <p class="text-xs text-slate-700 dark:text-slate-200">${escapeHtml(m.content)}</p>
      </div>
    </div>
  `).join('');
}

// Process Brain Dump
processDumpBtn.addEventListener('click', async () => {
  const text = brainDumpInput.value.trim();
  if (!text) return;

  processDumpBtn.disabled = true;
  processDumpBtn.innerHTML = `<i class="ph ph-spinner animate-spin"></i><span>Deconstructing with Open AI...</span>`;

  try {
    // 1. Sentiment & Emotional Temperature
    const { sentiment, score } = await analyzeSentiment(text);
    sentimentCard.classList.remove('hidden');
    
    if (sentiment === 'negative' || sentiment === 'label_0') {
      sentimentLabel.textContent = 'High Stress / Overwhelmed';
      sentimentLabel.className = 'text-sm font-bold text-rose-600 dark:text-rose-400 capitalize';
      sentimentBar.className = 'bg-rose-500 h-full rounded-full transition-all duration-500';
      sentimentBar.style.width = `${Math.max(70, Math.round(score * 100))}%`;
      sentimentAdvice.textContent = 'Take a deep breath and drink some water. We transformed your pile of thoughts into 5-minute bite-sized steps below.';
    } else {
      sentimentLabel.textContent = 'Focused & Determined';
      sentimentLabel.className = 'text-sm font-bold text-emerald-600 dark:text-emerald-400 capitalize';
      sentimentBar.className = 'bg-emerald-500 h-full rounded-full transition-all duration-500';
      sentimentBar.style.width = `${Math.max(60, Math.round(score * 100))}%`;
      sentimentAdvice.textContent = 'You have clear energy! Tackle the highest impact micro-task first.';
    }
    sentimentScore.textContent = `Confidence: ${Math.round(score * 100)}%`;

    // 2. Micro-Task Breakdown
    const newTasks = deconstructBrainDump(text);
    tasks = [...newTasks, ...tasks];
    await saveTasksToStorage();
    renderTasks();
  } finally {
    processDumpBtn.disabled = false;
    processDumpBtn.innerHTML = `<i class="ph-bold ph-magic-wand"></i><span>Deconstruct & Calm</span>`;
  }
});

// Clear Brain Dump
clearDumpBtn.addEventListener('click', () => {
  brainDumpInput.value = '';
});

// Load Demo Sample
quickFillDemoBtn.addEventListener('click', () => {
  brainDumpInput.value = "I'm super anxious about my demo tomorrow morning. My backend has an unresolved CORS error, my slides are empty, I haven't had lunch yet, and Sarah messaged me asking for the project link.";
  processDumpBtn.click();
});

// Clear Completed Tasks
clearCompletedTasksBtn.addEventListener('click', async () => {
  tasks = tasks.filter(t => !t.completed);
  await saveTasksToStorage();
  renderTasks();
});

// Save Memory
saveMemoryBtn.addEventListener('click', async () => {
  const content = memoryContentInput.value.trim();
  const tag = memoryTagInput.value.trim() || '#note';

  if (!content) return;

  saveMemoryBtn.disabled = true;
  saveMemoryBtn.innerHTML = `<i class="ph ph-spinner animate-spin"></i><span>Vectorizing...</span>`;

  try {
    const embedding = await getEmbedding(content);
    const newMemory = {
      id: Date.now(),
      tag,
      content,
      embedding,
      createdAt: new Date().toISOString()
    };

    memories.unshift(newMemory);
    await saveMemoriesToStorage();
    renderMemories();

    memoryContentInput.value = '';
    memoryTagInput.value = '';
  } finally {
    saveMemoryBtn.disabled = false;
    saveMemoryBtn.innerHTML = `<i class="ph ph-floppy-disk"></i><span>Vectorize & Store Locally</span>`;
  }
});

// Semantic Search Listener
searchMemoryBtn.addEventListener('click', () => {
  performSemanticSearch(searchMemoryQuery.value);
});
searchMemoryQuery.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    performSemanticSearch(searchMemoryQuery.value);
  }
});

// Seed Memories
seedMemoriesBtn.addEventListener('click', async () => {
  const samples = [
    {
      tag: '#code-fix',
      content: 'When Express gives CORS errors on local fetch, ensure cors({ origin: "*" }) is added before any routes.'
    },
    {
      tag: '#mindset',
      content: 'A finished B+ project delivered to a friend is infinitely more valuable than a perfect A+ project never shipped.'
    },
    {
      tag: '#adhd-tip',
      content: 'When feeling task paralysis, set a timer for 3 minutes and just open the editor. Starting is 90% of the friction.'
    }
  ];

  for (const s of samples) {
    const embedding = await getEmbedding(s.content);
    memories.push({
      id: Date.now() + Math.floor(Math.random() * 1000),
      tag: s.tag,
      content: s.content,
      embedding,
      createdAt: new Date().toISOString()
    });
  }

  await saveMemoriesToStorage();
  renderMemories();
});

// AI Buddy Encouragement Generator
const pepTalks = {
  warm: [
    "Hey friend! Remember that you don't have to carry the whole mountain all at once. Just focus on moving one small stone. Pick the first 5-minute task, and know I'm cheering for you!",
    "Breathe out the tension in your shoulders. You've solved harder problems than this before, and you're making steady progress step by step.",
    "Give yourself credit for showing up today. Even opening your workspace is a victory. Let's do this together!"
  ],
  coach: [
    "Let's go! Focus is a muscle. Lock in on that 1st task for the next 15 minutes with zero tabs open. You're going to crush this milestone!",
    "Action creates clarity! Don't wait to feel ready — take the easiest step on your list right now and build that unstoppable momentum!",
    "High energy, clear target! Knock out the 5-minute quick win on your board and watch your confidence snowball!"
  ],
  calm: [
    "Notice the feeling of urgency and observe it without panic. One conscious breath, one intentional choice. What is the single most essential task right now?",
    "External pressure is noise; your inner steady rhythm is signal. Complete one thing calmly before looking at the rest.",
    "Nothing is as urgent as your calm presence of mind. Step slowly and intentionally into the next micro-step."
  ],
  tech: [
    "Senior Dev reminder: Break the feature branch down into tiny atomic commits. If a function is too large to test, split it. Keep your iteration loop tight and ship.",
    "Debug systematically: log your state, verify one hypothesis at a time, and remember that 99% of bugs are simple assumptions that didn't hold.",
    "Premature optimization is the enemy of shipping. Get the core flow green, verify locally, and deploy."
  ]
};

generateBuddyPepTalkBtn.addEventListener('click', () => {
  const tone = buddyToneSelect.value;
  const list = pepTalks[tone] || pepTalks.warm;
  const randomMsg = list[Math.floor(Math.random() * list.length)];
  
  buddyToneDisplay.textContent = buddyToneSelect.options[buddyToneSelect.selectedIndex].text;
  buddyMessageOutput.textContent = `"${randomMsg}"`;
});

// Tab Switcher
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    tabButtons.forEach(b => {
      b.classList.remove('active', 'text-emerald-600', 'dark:text-emerald-400', 'border-b-2', 'border-emerald-500');
      b.classList.add('text-slate-500', 'dark:text-slate-400');
    });
    btn.classList.add('active', 'text-emerald-600', 'dark:text-emerald-400', 'border-b-2', 'border-emerald-500');
    btn.classList.remove('text-slate-500', 'dark:text-slate-400');

    const targetTab = btn.getAttribute('data-tab');
    document.getElementById('tab-deconstructor').classList.toggle('hidden', targetTab !== 'deconstructor');
    document.getElementById('tab-memory').classList.toggle('hidden', targetTab !== 'memory');
    document.getElementById('tab-companion').classList.toggle('hidden', targetTab !== 'companion');
  });
});

// Theme Toggle
themeToggle.addEventListener('click', () => {
  const isDark = document.documentElement.classList.toggle('dark');
  themeIcon.className = isDark ? 'ph ph-sun text-xl text-amber-400' : 'ph ph-moon text-xl';
  localStorage.setItem('buddybrain_theme', isDark ? 'dark' : 'light');
});

// Restore Theme
if (localStorage.getItem('buddybrain_theme') === 'dark' || (!('buddybrain_theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
  document.documentElement.classList.add('dark');
  themeIcon.className = 'ph ph-sun text-xl text-amber-400';
}

// Utility: Escape HTML
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Initialization on load
window.addEventListener('DOMContentLoaded', () => {
  loadData();
  initAi();
});
