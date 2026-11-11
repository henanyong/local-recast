/* local-recast workbench frontend */
const $ = (id) => document.getElementById(id);

const state = {
  tasks: [],
  selectedId: null,
  checked: new Set(),
  running: false,
  stages: [],
  configs: [],
  scanRoot: "",
  overall: 0,
  runDone: 0,
  runTotal: 0,
  settingsDirty: false,
};

const STAGE_DEFAULT = [
  { id: "preprocess", label: "01 视频预处理" },
  { id: "subtitles", label: "02 提取字幕" },
  { id: "tracks", label: "03 人物跟踪与命名" },
  { id: "pose", label: "04 动作关键点" },
  { id: "compose", label: "05 本地替换合成" },
];

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

function applySnapshot(snap) {
  if (!snap) return;
  state.tasks = snap.tasks || [];
  state.running = !!snap.running;
  state.scanRoot = snap.scan_root || state.scanRoot;
  state.overall = snap.overall || 0;
  state.runDone = snap.run_done || 0;
  state.runTotal = snap.run_total || 0;
  if (snap.stages && snap.stages.length) state.stages = snap.stages;
  if (snap.configs && snap.configs.length) {
    state.configs = snap.configs;
    fillConfigs();
  }
  if (snap.scan_root && !$("scan-path").value) $("scan-path").value = snap.scan_root;
  const valid = new Set(state.tasks.map((t) => t.id));
  state.checked = new Set([...state.checked].filter((id) => valid.has(id)));
  if (state.selectedId && !valid.has(state.selectedId)) state.selectedId = null;
  render();
}

function render() {
  $("run-indicator").textContent = state.running ? "处理中" : "空闲";
  $("run-indicator").className = "chip " + (state.running ? "busy" : "idle");
  $("queue-count").textContent = `${state.tasks.length} 项`;
  $("overall-text").textContent = `${state.runDone} / ${state.runTotal}`;
  $("overall-bar").style.width = `${Math.round((state.overall || 0) * 100)}%`;
  $("btn-stop").disabled = !state.running;
  renderTable();
  renderSettings();
}

function renderTable() {
  const body = $("task-body");
  if (!state.tasks.length) {
    body.innerHTML = `<tr class="empty-row"><td colspan="7">选择一个目录并扫描，视频和分集文件夹会进入队列。</td></tr>`;
    $("check-all").checked = false;
    return;
  }
  body.innerHTML = state.tasks.map((t) => {
    const checked = state.checked.has(t.id) ? "checked" : "";
    const sel = state.selectedId === t.id ? "selected" : "";
    const openDisabled = t.output_dir ? "" : "disabled";
    const path = escapeHtml(t.path || "");
    const progress = escapeHtml(t.progress || "");
    return `<tr data-id="${t.id}" class="${sel}">
      <td class="col-check"><input type="checkbox" data-check="${t.id}" ${checked} /></td>
      <td><div class="name-cell"><strong>${escapeHtml(t.name)}</strong><small title="${path}">${path}</small></div></td>
      <td class="col-eps">${t.episode_count || 1}</td>
      <td class="col-status"><span class="status-pill status-${escapeHtml(t.status)}">${escapeHtml(t.status)}</span></td>
      <td class="progress-text">${progress}</td>
      <td class="col-out"><button class="linkish" data-open="${escapeHtml(t.output_dir || "")}" ${openDisabled}>打开</button></td>
      <td class="col-time">${escapeHtml(t.created_at || "")}</td>
    </tr>`;
  }).join("");
  const allChecked = state.tasks.every((t) => state.checked.has(t.id));
  $("check-all").checked = allChecked && state.tasks.length > 0;
}

function renderSettings() {
  const task = state.tasks.find((t) => t.id === state.selectedId);
  if (!task) {
    $("settings-form").hidden = true;
    $("settings-hint").hidden = false;
    return;
  }
  $("settings-hint").hidden = true;
  $("settings-form").hidden = false;
  $("sel-name").textContent = `${task.name}  ·  ${task.episode_count} 个视频`;
  if (state.settingsDirty) return;
  const s = task.settings || {};
  $("set-face").value = s.face || "";
  $("set-names").value = s.names || "";
  const cfgDefault = (state.configs[0] && state.configs[0].id) || "configs/4060ti-8g.yaml";
  $("set-config").value = s.config || cfgDefault;
  $("set-lang").value = s.lang || "auto";
  $("set-burn").checked = !!s.burn_subs;
  $("set-track").value = s.track_id || "";
}

function fillConfigs() {
  const sel = $("set-config");
  const task = state.tasks.find((t) => t.id === state.selectedId);
  const current = sel.value || (task && task.settings && task.settings.config) || "";
  sel.innerHTML = (state.configs.length ? state.configs : [
    { id: "configs/4060ti-8g.yaml", label: "4060 Ti 8GB（推荐）" },
    { id: "configs/4060ti-16g.yaml", label: "4060 Ti 16GB" },
  ]).map((c) => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`).join("");
  const options = [...sel.options].map((o) => o.value);
  if (current && options.includes(current)) sel.value = current;
}

function fillStages() {
  const stages = state.stages.length ? state.stages : STAGE_DEFAULT;
  $("stage-buttons").innerHTML = stages.map((s, i) => {
    const label = s.label || s.id;
    const num = label.slice(0, 2);
    const rest = label.replace(/^\d+\s*/, "");
    return `<button type="button" class="stage-btn" data-stage="${s.id}"><b>STAGE ${String(i + 1).padStart(2, "0")}</b>${escapeHtml(rest || label)}</button>`;
  }).join("");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

function checkedIds() {
  return state.tasks.map((t) => t.id).filter((id) => state.checked.has(id));
}

function collectSettings() {
  const track = $("set-track").value;
  return {
    face: $("set-face").value.trim() || null,
    names: $("set-names").value.trim() || null,
    config: $("set-config").value,
    lang: $("set-lang").value,
    burn_subs: $("set-burn").checked,
    track_id: track ? Number(track) : null,
  };
}

async function saveSettings(applyChecked) {
  if (!state.selectedId) return;
  const body = { settings: collectSettings() };
  if (applyChecked) body.apply_ids = checkedIds().length ? checkedIds() : [state.selectedId];
  const snap = await api(`/api/tasks/${state.selectedId}/settings`, { method: "POST", body: JSON.stringify(body) });
  state.settingsDirty = false;
  applySnapshot(snap);
}

async function start(stages, includeDone = false) {
  const ids = checkedIds();
  if (!ids.length) {
    alert("请先勾选要执行的任务。");
    return;
  }
  const forceRerun = !!($("set-force") && $("set-force").checked);
  try {
    const snap = await api("/api/start", {
      method: "POST",
      body: JSON.stringify({ ids, stages, include_done: includeDone, force_rerun: forceRerun }),
    });
    applySnapshot(snap);
  } catch (err) {
    alert(err.message || err);
  }
}

$("task-body").addEventListener("click", async (ev) => {
  const open = ev.target.closest("[data-open]");
  if (open) {
    ev.preventDefault();
    ev.stopPropagation();
    if (open.disabled || !open.dataset.open) return;
    try { await api("/api/open", { method: "POST", body: JSON.stringify({ path: open.dataset.open }) }); }
    catch (err) { alert(err.message || err); }
    return;
  }
  const check = ev.target.closest("[data-check]");
  if (check) {
    ev.stopPropagation();
    if (check.checked) state.checked.add(check.dataset.check);
    else state.checked.delete(check.dataset.check);
    const row = ev.target.closest("tr[data-id]");
    if (row && state.selectedId !== row.dataset.id) {
      state.selectedId = row.dataset.id;
      state.settingsDirty = false;
    }
    render();
    return;
  }
  const row = ev.target.closest("tr[data-id]");
  if (row) {
    if (state.selectedId !== row.dataset.id) state.settingsDirty = false;
    state.selectedId = row.dataset.id;
    render();
  }
});

$("check-all").addEventListener("change", (ev) => {
  if (ev.target.checked) state.tasks.forEach((t) => state.checked.add(t.id));
  else state.checked.clear();
  render();
});

$("btn-scan").addEventListener("click", async () => {
  const path = $("scan-path").value.trim();
  if (!path) { alert("请先填写或浏览一个本地目录。"); return; }
  try {
    const snap = await api("/api/scan", { method: "POST", body: JSON.stringify({ path, merge: true }) });
    applySnapshot(snap);
  } catch (err) { alert(err.message || err); }
});

$("btn-start").addEventListener("click", () => start(null, false));
$("btn-all-stages").addEventListener("click", () => start(null, false));
$("btn-stop").addEventListener("click", async () => {
  try { applySnapshot(await api("/api/stop", { method: "POST", body: "{}" })); }
  catch (err) { alert(err.message || err); }
});
$("btn-clear").addEventListener("click", async () => {
  try { applySnapshot(await api("/api/clear-completed", { method: "POST", body: "{}" })); }
  catch (err) { alert(err.message || err); }
});
$("btn-remove").addEventListener("click", async () => {
  const ids = checkedIds();
  if (!ids.length) { alert("请先勾选要移除的任务。"); return; }
  try { applySnapshot(await api("/api/remove", { method: "POST", body: JSON.stringify({ ids }) })); }
  catch (err) { alert(err.message || err); }
});
$("btn-save").addEventListener("click", () => saveSettings(false).catch((err) => alert(err.message || err)));
$("btn-apply").addEventListener("click", () => saveSettings(true).catch((err) => alert(err.message || err)));
$("stage-buttons").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-stage]");
  if (!btn) return;
  start([btn.dataset.stage], true);
});

$("set-face-file").addEventListener("change", async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!file || !state.selectedId) return;
  const body = new FormData();
  body.append("field", "face");
  body.append("file", file);
  const res = await fetch(`/api/tasks/${state.selectedId}/upload`, { method: "POST", body });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "上传失败"); return; }
  applySnapshot(data);
});
$("set-names-file").addEventListener("change", async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!file || !state.selectedId) return;
  const body = new FormData();
  body.append("field", "names");
  body.append("file", file);
  const res = await fetch(`/api/tasks/${state.selectedId}/upload`, { method: "POST", body });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "上传失败"); return; }
  applySnapshot(data);
});

const logView = $("log-view");
$("btn-log-clear").addEventListener("click", () => { logView.textContent = ""; });
function appendLog(line) {
  logView.textContent += (logView.textContent && !logView.textContent.endsWith("\n") ? "\n" : "") + line;
  if ($("log-follow").checked) logView.scrollTop = logView.scrollHeight;
}

function connectEvents() {
  const evs = new EventSource("/api/events");
  evs.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      if (data.type === "log" && data.line) appendLog(data.line);
      if (data.type === "state") applySnapshot(data.data);
    } catch { /* ignore malformed */ }
  };
  evs.onerror = () => {
    evs.close();
    setTimeout(connectEvents, 1500);
  };
}

/* Path browser modal */
const modal = {
  el: $("modal"),
  kind: "dir",
  current: "",
  parent: null,
  target: null,
};

function openModal(kind, target, title) {
  modal.kind = kind;
  modal.target = target;
  $("modal-title").textContent = title;
  $("modal-choose").style.display = kind === "dir" ? "" : "none";
  modal.el.hidden = false;
  const start = $("scan-path").value.trim() || "";
  loadBrowse(start, kind);
}

async function loadBrowse(path, kind) {
  const data = await api(`/api/browse?kind=${encodeURIComponent(kind)}&path=${encodeURIComponent(path || "")}`);
  modal.current = data.path || "";
  modal.parent = data.parent;
  $("modal-path").textContent = data.path || "此电脑 / 常用位置";
  $("modal-up").disabled = !data.parent && !!data.path;
  const entries = data.entries || [];
  $("modal-list").innerHTML = entries.length
    ? entries.map((e) => {
        const meta = e.is_dir ? (e.video_count ? `${e.video_count} 个视频` : "文件夹") : "文件";
        return `<button type="button" class="browser-item" data-path="${escapeHtml(e.path)}" data-dir="${e.is_dir ? 1 : 0}">
          <span>${escapeHtml(e.name)}</span><small>${meta}</small>
        </button>`;
      }).join("")
    : `<div class="muted" style="padding:16px">此目录为空，或没有可显示的项目。</div>`;
}

$("modal-list").addEventListener("click", (ev) => {
  const item = ev.target.closest(".browser-item");
  if (!item) return;
  const isDir = item.dataset.dir === "1";
  if (isDir) {
    loadBrowse(item.dataset.path, modal.kind).catch((err) => alert(err.message || err));
    return;
  }
  if (modal.kind !== "dir" && modal.target) {
    modal.target.value = item.dataset.path;
    if (modal.target.id === "set-face" || modal.target.id === "set-names") state.settingsDirty = true;
    modal.el.hidden = true;
  }
});
$("modal-up").addEventListener("click", () => {
  loadBrowse(modal.parent || "", modal.kind).catch((err) => alert(err.message || err));
});
$("modal-choose").addEventListener("click", () => {
  if (modal.target) modal.target.value = modal.current;
  else $("scan-path").value = modal.current;
  modal.el.hidden = true;
});
$("modal-close").addEventListener("click", () => { modal.el.hidden = true; });
$("btn-browse").addEventListener("click", () => openModal("dir", $("scan-path"), "选择扫描目录"));
$("btn-face-browse").addEventListener("click", () => openModal("image", $("set-face"), "选择参考脸图片"));
$("btn-names-browse").addEventListener("click", () => openModal("names", $("set-names"), "选择人名花名册"));

["set-face", "set-names", "set-config", "set-lang", "set-burn", "set-track"].forEach((id) => {
  const el = $(id);
  if (!el) return;
  el.addEventListener("input", () => { state.settingsDirty = true; });
  el.addEventListener("change", () => { state.settingsDirty = true; });
});

fillStages();
fillConfigs();
api("/api/state").then((snap) => {
  applySnapshot(snap);
  fillStages();
}).catch((err) => appendLog("无法连接到工作台: " + err.message));
connectEvents();
