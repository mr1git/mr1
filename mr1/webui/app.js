const canvas = document.getElementById("timelineCanvas");
const ctx = canvas.getContext("2d");
const followButton = document.getElementById("followButton");
const statusPill = document.getElementById("statusPill");
const zoomValue = document.getElementById("zoomValue");
const runningCount = document.getElementById("runningCount");
const sessionMeta = document.getElementById("sessionMeta");
const latestEvent = document.getElementById("latestEvent");
const conversationMeta = document.getElementById("conversationMeta");
const conversationList = document.getElementById("conversationList");
const promptForm = document.getElementById("promptForm");
const promptInput = document.getElementById("promptInput");
const promptStatus = document.getElementById("promptStatus");
const sendButton = document.getElementById("sendButton");
const jobList = document.getElementById("jobList");
const jobMeta = document.getElementById("jobMeta");

const camera = {
  follow: true,
  panSeconds: 0,
  secondsPerPixel: 0.027
};

const state = {
  snapshot: null,
  dragging: false,
  lastX: 0,
  promptBusy: false
};

const ROYGBIV = ["#ff626c", "#ff9d4d", "#ffd85e", "#5ad878", "#61b6ff", "#6e70ff", "#cc6dff"];
const BG_GRID = "rgba(255,255,255,0.05)";
const AXIS = "rgba(255, 112, 118, 0.42)";

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const colorForPath = (path = []) => {
  if (path.length === 0) {
    return ROYGBIV[0];
  }
  const index = path.reduce((sum, part, depth) => sum + (part + 1) * (depth + 1), 0);
  return ROYGBIV[Math.min(index, ROYGBIV.length - 1)];
};

const dimColor = (hex) => {
  const clean = hex.replace("#", "");
  const channels = [0, 2, 4].map((offset) => parseInt(clean.slice(offset, offset + 2), 16));
  return `rgb(${channels.map((value) => Math.round(value * 0.48)).join(",")})`;
};

const shadeColor = (hex, ratio = 0.82) => {
  const clean = hex.replace("#", "");
  const channels = [0, 2, 4].map((offset) => parseInt(clean.slice(offset, offset + 2), 16));
  return `rgb(${channels.map((value) => Math.round(value * ratio)).join(",")})`;
};

const summarize = (text, length = 30) => {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean.length <= length ? clean : `${clean.slice(0, length - 1)}…`;
};

const headTimeMs = (task, nowMs) => {
  if (task.status === "running") {
    return nowMs;
  }
  return new Date(task.finished_at || task.updated_at || task.started_at || nowMs).getTime();
};

const worldToX = (worldMs, nowMs, width) => {
  const centerMs = nowMs + camera.panSeconds * 1000;
  return width / 2 + ((worldMs - centerMs) / 1000) / camera.secondsPerPixel;
};

const buildTaskLayout = (snapshot, axisY, height) => {
  const byParent = new Map();
  for (const task of snapshot.tasks) {
    const parentId = task.parent_task_id || "mr1";
    const siblings = byParent.get(parentId) || [];
    siblings.push(task);
    byParent.set(parentId, siblings);
  }

  const positions = new Map([["mr1", axisY]]);
  const leafCache = new Map();

  const laneChildren = (parentId, lane) => {
    return (byParent.get(parentId) || [])
      .filter((task) => (lane === "system" ? task.lane === "system" : task.lane !== "system"))
      .sort((left, right) => {
        return (left.started_at || "").localeCompare(right.started_at || "") || left.task_id.localeCompare(right.task_id);
      });
  };

  const leafCount = (taskId, lane) => {
    const cacheKey = `${lane}:${taskId}`;
    if (leafCache.has(cacheKey)) {
      return leafCache.get(cacheKey);
    }
    const children = laneChildren(taskId, lane);
    if (!children.length) {
      leafCache.set(cacheKey, 1);
      return 1;
    }
    const total = children.reduce((sum, child) => sum + leafCount(child.task_id, lane), 0);
    leafCache.set(cacheKey, total);
    return total;
  };

  const assignLane = (parentId, lane, startY, endY, depth = 0) => {
    const children = laneChildren(parentId, lane);
    if (!children.length) {
      return;
    }

    const span = endY - startY;
    const totalLeaves = children.reduce((sum, child) => sum + leafCount(child.task_id, lane), 0);
    let cursor = startY;
    const padding = Math.max(6, 16 - depth * 2);

    for (const child of children) {
      const portion = (span * leafCount(child.task_id, lane)) / totalLeaves;
      const childStart = cursor;
      const childEnd = cursor + portion;
      const childY = (childStart + childEnd) / 2;
      positions.set(child.task_id, childY);

      const nextStart = childStart + padding;
      const nextEnd = childEnd - padding;
      if (nextEnd - nextStart > 10) {
        assignLane(child.task_id, lane, nextStart, nextEnd, depth + 1);
      }
      cursor += portion;
    }
  };

  assignLane("mr1", "system", 54, Math.max(70, axisY - 84));
  assignLane("mr1", "conversation", Math.min(height - 70, axisY + 84), height - 54);
  return positions;
};

const drawText = (text, x, y, options = {}) => {
  ctx.font = options.font || "500 13px SF Pro Display, sans-serif";
  ctx.fillStyle = options.color || "#eef3ff";
  ctx.textAlign = options.align || "left";
  ctx.fillText(text, x, y);
};

const drawPixelSprite = (x, y, scale, color, options = {}) => {
  const shade = options.shade || shadeColor(color);
  const eye = options.eye || "#181313";
  const alive = options.alive !== false;
  const base = alive ? color : dimColor(color);
  const shadeBase = alive ? shade : dimColor(shade);
  const rows = [
    "..######..",
    ".########.",
    "##########",
    "##########",
    "##########",
    ".########."
  ];

  ctx.save();
  ctx.translate(x, y);
  ctx.imageSmoothingEnabled = false;

  rows.forEach((row, rowIndex) => {
    [...row].forEach((cell, colIndex) => {
      if (cell !== "#") {
        return;
      }
      const fill = colIndex >= 7 ? shadeBase : base;
      ctx.fillStyle = fill;
      ctx.fillRect((colIndex - row.length / 2) * scale, (rowIndex - rows.length / 2) * scale, scale, scale);
    });
  });

  ctx.fillStyle = eye;
  ctx.fillRect((-2.2 * scale), (-1.5 * scale), 0.95 * scale, 1.65 * scale);
  ctx.fillRect((1.25 * scale), (-1.5 * scale), 0.95 * scale, 1.65 * scale);
  ctx.restore();
};

const renderConversation = () => {
  const conversation = state.snapshot?.conversation || [];
  conversationMeta.textContent = `${conversation.length} messages`;
  conversationList.innerHTML = "";
  for (const entry of conversation.slice(-10)) {
    const item = document.createElement("article");
    item.className = `conversation-item ${entry.role}`;
    const meta = document.createElement("div");
    meta.className = "conversation-meta";
    const role = document.createElement("strong");
    role.textContent = entry.role;
    const timestamp = document.createElement("span");
    timestamp.textContent = new Date(entry.timestamp).toLocaleTimeString([], {hour12: false});
    meta.append(role, timestamp);
    const text = document.createElement("p");
    text.textContent = entry.text;
    item.append(meta, text);
    conversationList.appendChild(item);
  }
};

const renderJobs = () => {
  const tasks = state.snapshot?.tasks || [];
  const ranked = tasks.slice().sort((left, right) => {
    const liveDelta = Number(right.status === "running") - Number(left.status === "running");
    if (liveDelta !== 0) {
      return liveDelta;
    }
    return (right.updated_at || right.started_at || "").localeCompare(left.updated_at || left.started_at || "");
  });
  jobMeta.textContent = `${tasks.length} total`;
  jobList.innerHTML = "";

  if (!ranked.length) {
    const empty = document.createElement("p");
    empty.className = "job-empty";
    empty.textContent = "No jobs yet. Try /test spawn agents 3 in the prompt panel.";
    jobList.appendChild(empty);
    return;
  }

  for (const task of ranked.slice(0, 9)) {
    const item = document.createElement("article");
    item.className = `job-item ${task.status}`;
    const heading = document.createElement("div");
    heading.className = "job-heading";
    const name = document.createElement("strong");
    name.textContent = summarize(task.description || task.agent_type || task.task_id, 26);
    const status = document.createElement("span");
    status.textContent = task.status;
    heading.append(name, status);
    const meta = document.createElement("p");
    meta.textContent = `${task.task_id} • ${task.agent_type || "agent"}`;
    item.append(heading, meta);
    jobList.appendChild(item);
  }
};

const updateChrome = () => {
  if (!state.snapshot) {
    return;
  }
  const session = state.snapshot.session;
  statusPill.textContent = state.promptBusy ? "MR1 thinking…" : "Live";
  zoomValue.textContent = `${(camera.secondsPerPixel * 96).toFixed(1)}s / 96px`;
  runningCount.textContent = `${state.snapshot.summary.running_count} running`;
  sessionMeta.textContent = `${session.session_id || "no session"} • ${session.status}`;
  latestEvent.textContent = state.snapshot.events.at(-1)?.summary || "No events yet.";
  followButton.textContent = camera.follow ? "Following MR1" : "Relock on MR1";
};

const renderCanvas = () => {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = false;

  const nowMs = Date.now();
  const axisY = height / 2;
  const mr1X = camera.follow ? width / 2 : worldToX(nowMs, nowMs, width);

  ctx.strokeStyle = BG_GRID;
  ctx.lineWidth = 1;
  for (let y = 42; y < height; y += 72) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = AXIS;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, axisY);
  ctx.lineTo(width, axisY);
  ctx.stroke();

  drawText("SYSTEM JOBS", 24, 28, {color: "#93a0ff", font: "700 12px SF Pro Display, sans-serif"});
  drawText("CONVERSATION JOBS", 24, height - 20, {color: "#72e4ff", font: "700 12px SF Pro Display, sans-serif"});

  if (!state.snapshot) {
    drawText("Waiting for snapshot…", 28, axisY - 40, {color: "#9aa6bf", font: "500 18px SF Pro Display, sans-serif"});
    return;
  }

  const positions = buildTaskLayout(state.snapshot, axisY, height);
  const taskMap = new Map(state.snapshot.tasks.map((task) => [task.task_id, task]));

  for (const task of state.snapshot.tasks) {
    const alive = task.status === "running";
    const color = colorForPath(task.path || []);
    const y = positions.get(task.task_id) || axisY;
    const headX = worldToX(headTimeMs(task, nowMs), nowMs, width);
    const startX = worldToX(new Date(task.started_at || state.snapshot.generated_at).getTime(), nowMs, width);
    const parent = task.parent_task_id ? taskMap.get(task.parent_task_id) : null;
    const parentX = parent
      ? worldToX(headTimeMs(parent, nowMs), nowMs, width)
      : mr1X;
    const parentY = parent ? (positions.get(parent.task_id) || axisY) : axisY;

    ctx.strokeStyle = alive ? color : dimColor(color);
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (alive) {
      ctx.moveTo(parentX, parentY);
      ctx.lineTo(parentX, y);
      ctx.lineTo(headX, y);
    } else {
      ctx.moveTo(startX, y);
      ctx.lineTo(headX, y);
    }
    ctx.stroke();

    drawPixelSprite(headX, y, alive ? 4 : 3.5, color, {
      shade: shadeColor(color, 0.78),
      alive
    });
    drawText(
      summarize(task.description || task.agent_type || task.task_id, 24),
      headX + 22,
      y + (task.lane === "system" ? -12 : 22),
      {
        color: alive ? color : dimColor(color),
        font: "600 12px SF Pro Display, sans-serif"
      }
    );
  }

  drawPixelSprite(mr1X, axisY, 6.5, "#ff676d", {shade: "#cf4053", eye: "#171010", alive: true});
  drawText("MR1", mr1X, axisY + 52, {
    color: "#ffd6d8",
    align: "center",
    font: "700 13px SF Pro Display, sans-serif"
  });

  if (!state.snapshot.tasks.length) {
    drawText("No jobs yet. Use /test spawn agents <h> to generate a tree.", 28, axisY - 56, {
      color: "#9aa6bf",
      font: "500 15px SF Pro Display, sans-serif"
    });
  }
};

const fetchSnapshot = async () => {
  const response = await fetch("/api/snapshot", {cache: "no-store"});
  if (!response.ok) {
    throw new Error(`snapshot failed: ${response.status}`);
  }
  state.snapshot = await response.json();
  updateChrome();
  renderConversation();
  renderJobs();
};

const submitPrompt = async (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text || state.promptBusy) {
    return;
  }
  state.promptBusy = true;
  promptStatus.textContent = "Sending…";
  sendButton.disabled = true;
  updateChrome();
  try {
    const response = await fetch("/api/input", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "request failed");
    }
    promptInput.value = "";
    promptStatus.textContent = payload.kind === "command_result" ? "Command handled." : "MR1 replied.";
    await fetchSnapshot();
  } catch (error) {
    promptStatus.textContent = String(error);
  } finally {
    state.promptBusy = false;
    sendButton.disabled = false;
    updateChrome();
  }
};

canvas.addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.lastX = event.clientX;
  camera.follow = false;
  updateChrome();
});

window.addEventListener("pointerup", () => {
  state.dragging = false;
});

window.addEventListener("pointermove", (event) => {
  if (!state.dragging) {
    return;
  }
  const dx = event.clientX - state.lastX;
  state.lastX = event.clientX;
  camera.panSeconds -= dx * camera.secondsPerPixel;
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = event.deltaY > 0 ? 1.11 : 0.9;
  camera.secondsPerPixel = clamp(camera.secondsPerPixel * factor, 0.004, 0.33);
  updateChrome();
}, {passive: false});

window.addEventListener("keydown", (event) => {
  if (event.key === "f") {
    camera.follow = true;
    camera.panSeconds = 0;
    updateChrome();
  }
});

followButton.addEventListener("click", () => {
  camera.follow = true;
  camera.panSeconds = 0;
  updateChrome();
});

promptForm.addEventListener("submit", submitPrompt);

const tick = async () => {
  try {
    await fetchSnapshot();
  } catch (error) {
    statusPill.textContent = "Disconnected";
    latestEvent.textContent = String(error);
  }
};

const animate = () => {
  renderCanvas();
  requestAnimationFrame(animate);
};

await tick();
setInterval(tick, 1200);
animate();
