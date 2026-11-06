"use strict";
// The Settings tab. State lives in S (the working copy) and S0 (what is saved); the bar at the bottom counts the difference.
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  n.append(...kids.flat().filter(k => k !== null && k !== undefined && k !== false));
  return n;
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const clone = o => JSON.parse(JSON.stringify(o));
const PURPOSES = [["General text (apps, tools)", "llm", "plain"], ["Roleplay: goes first, short replies", "llm", "rp"], ["ComfyUI images", "comfy", "plain"]];
const SHORT = ["General text", "Roleplay", "ComfyUI images"];
const purposeOf = r => r.kind === "comfy" ? 2 : r.profile === "rp" ? 1 : 0;
const SERVICE = new Set(["ui_port", "data_dir"]), ENGINE = new Set(["threads", "engine_dir", "engine_port"]);
const FEATURES = [["vision", "Vision"], ["reasoning", "Reasoning"], ["tools", "Tool use"]];

let S = null, S0 = "", MODELS = [], CLIENTS = [], found = null, downloads = [], pollTimer = null;
const openDoors = new Set();

const defaultPriority = routes => {
  const rank = r => r.kind === "llm" && r.profile === "rp" ? 0 : r.kind === "llm" ? 1 : 2;
  return [...routes].sort((a, b) => rank(a) - rank(b)).map(r => ({ door: r.port }));
};
const rules = () => S.priority ?? defaultPriority(S.routes);
const utc = ts => new Date(ts.replace(" ", "T") + "Z");
const ago = ts => {
  const s = Math.max(0, (Date.now() - utc(ts)) / 1000);
  return s < 90 ? "just now" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 129600 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " days ago";
};
const gb = n => n >= 2 ** 30 ? (n / 2 ** 30).toFixed(1) + " GB" : Math.round(n / 2 ** 20) + " MB";

function copyText(text, button) {
  const done = () => { const old = button.title; button.classList.add("done"); button.title = "Copied"; setTimeout(() => { button.classList.remove("done"); button.title = old; }, 1500); };
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text).then(done);
  const box = el("textarea", { value: text }); box.style.position = "fixed"; box.style.opacity = "0";  // plain http on a LAN has no clipboard API
  document.body.append(box); box.select(); try { document.execCommand("copy"); done(); } finally { box.remove(); }
}

const ICON = {
  chev: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M5 3l5 5-5 5"/></svg>',
  copy: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M3 10.5V4a1.5 1.5 0 0 1 1.5-1.5H11"/></svg>',
  trash: '<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/></svg>',
  vision: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z"/><circle cx="8" cy="8" r="2"/></svg>',
  tools: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M10.5 2.5a3 3 0 0 0-2.7 4L2.5 11.8a1.4 1.4 0 0 0 2 2L9.8 8.5a3 3 0 0 0 4-2.7L12 7.5 9.5 5z"/></svg>',
  reasoning: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="8" cy="8" r="6"/><path d="M8 2v12"/></svg>',
};
const withIcon = (svg, tag, props) => { const n = el(tag, props); n.insertAdjacentHTML("afterbegin", svg); return n; };

// ---------- dirty tracking and the save bar
function changes() {
  const o = JSON.parse(S0), out = { n: 0, service: false, engine: false };
  for (const k of Object.keys(S)) {
    if (k === "routes") {
      for (let i = 0; i < Math.max(S.routes.length, o.routes.length); i++) {
        const a = S.routes[i], b = o.routes[i];
        if (same(a, b)) continue;
        out.n++;
        if (!a || !b || a.port !== b.port || a.kind !== b.kind || a.profile !== b.profile) out.service = true;  // a new purpose is only picked up by a restart
      }
    } else if (!same(S[k], o[k])) {
      out.n++;
      if (SERVICE.has(k)) out.service = true;
      if (ENGINE.has(k)) out.engine = true;
    }
  }
  return out;
}

function updateBar() {
  const c = changes(), bar = document.getElementById("savebar");
  bar.hidden = c.n === 0;
  document.getElementById("save-count").textContent = c.n;
  document.getElementById("save-note").textContent = (c.n === 1 ? " unsaved change" : " unsaved changes") +
    (c.service ? " (restarts the service)" : c.engine ? " (restarts the engine)" : "");
}

// ---------- doors
function doorRow(r, i) {
  const comfy = r.kind === "comfy", open = openDoors.has(i);
  const label = el("b"), port = el("span", { className: "dim" }), chip = el("span", { className: "warnchip", textContent: "not the usual 8188", hidden: true });
  const url = el("b", { className: "p" });
  const refresh = () => {
    const c = r.kind === "comfy";
    label.textContent = SHORT[purposeOf(r)]; port.textContent = "port " + r.port; url.textContent = r.port;
    chip.hidden = !(c && r.port !== 8188);
  };
  const chev = withIcon(ICON.chev, "button", { type: "button", className: "chev", ariaLabel: "Show settings for this door", onclick: () => { open ? openDoors.delete(i) : openDoors.add(i); drawDoors(); } });
  chev.setAttribute("aria-expanded", String(open));
  const copy = withIcon(ICON.copy, "button", { type: "button", className: "copy", title: "Copy address", ariaLabel: "Copy address" });
  copy.onclick = () => copyText(`http://${location.hostname}:${r.port}`, copy);
  const main = el("tr", { className: "door" + (open ? " open" : "") },
    el("td", {}, chev), el("td", {}, label, " ", port, " ", chip), el("td", { className: "st-ok", textContent: "listening" }),
    el("td", {}, el("span", { className: "addr" }, el("code", {}, `http://${location.hostname}:`, url), copy)),
    el("td", { className: "act", id: "act-" + r.port }));
  refresh();
  if (!open) return [main];

  const portIn = el("input", { type: "number", value: r.port, ariaLabel: "Port" });
  let listed = r.port;  // the port the priority lines were last written with
  portIn.oninput = () => { r.port = +portIn.value; refresh(); updateBar(); };
  portIn.onchange = () => {  // only when the edit is finished, and never onto another door's port, so lines cannot merge
    if (S.routes.some((x, n) => n !== i && x.port === r.port)) return;
    if (S.priority) for (const line of S.priority) if (line.door === listed) line.door = r.port;
    listed = r.port; drawPriority(); updateBar();
  };
  const purpose = el("select", { ariaLabel: "What it is for" }, PURPOSES.map((p, n) => new Option(p[0], n, false, n === purposeOf(r))));
  purpose.onchange = () => {
    const [, kind, profile] = PURPOSES[+purpose.value];
    r.kind = kind; r.profile = profile;
    if (kind === "llm") { r.max_tokens = profile === "rp" ? 1024 : 8192; r.allow = r.allow || { vision: true, reasoning: true, tools: true }; }
    drawDoors(); drawPriority(); updateBar();
  };
  const words = el("small", { className: "dim" });
  const reply = el("input", { type: "number", value: r.max_tokens ?? (r.profile === "rp" ? S.max_tokens_cap : S.max_tokens_cap_plain), ariaLabel: "Longest reply in tokens" });
  const showWords = () => words.textContent = "≈ " + Math.round(reply.value * 0.75).toLocaleString() + " words";
  reply.oninput = () => { r.max_tokens = +reply.value; showWords(); updateBar(); };
  showWords();
  const allow = r.allow || { vision: true, reasoning: true, tools: true };
  const feats = el("fieldset", { className: "textonly" }, el("legend", { textContent: "Allowed on this door" }),
    FEATURES.map(([key, name]) => {
      const box = el("input", { type: "checkbox", checked: allow[key] !== false });
      box.onchange = () => { r.allow = { ...(r.allow || allow), [key]: box.checked }; updateBar(); };
      return el("label", {}, box, " " + name);
    }));
  const remove = el("button", { type: "button", className: "small danger-text", textContent: "Remove door" });
  remove.onclick = () => {
    S.routes.splice(i, 1);
    if (S.priority) S.priority = S.priority.filter(l => l.door !== r.port);
    openDoors.clear(); drawDoors(); drawPriority(); updateBar();
  };
  const panel = el("div", { className: "panel" },
    el("label", {}, el("span", { textContent: "Port" }), portIn),
    el("label", {}, el("span", { textContent: "What it is for" }), purpose),
    comfy ? null : el("label", {}, el("span", { textContent: "Longest reply" }), el("span", { className: "inl" }, reply, " tokens ", words)),
    comfy ? null : feats,
    comfy ? el("p", { className: "note comfynote", textContent: "Apps expect ComfyUI on 8188. The real ComfyUI must use another port (now " + S.comfy_url.replace(/^.*:(\d+).*$/, "$1") + ", under Advanced)." }) : null,
    remove);
  return [main, el("tr", { className: "detail" }, el("td"), el("td", { colSpan: 4 }, panel))];
}

function drawDoors() {
  const body = document.getElementById("doors-body"); body.replaceChildren();
  S.routes.forEach((r, i) => body.append(...doorRow(r, i)));
  if (window.lastDoors) drawActivity(window.lastDoors);
}

function drawActivity(doors) {
  window.lastDoors = doors;
  for (const [port, d] of Object.entries(doors)) {
    const cell = document.getElementById("act-" + port); if (!cell) continue;
    const text = d.busy ? "working now" : d.last_seen ? `${d.today} today · last ${ago(d.last_seen)}` : "no requests yet";
    cell.replaceChildren(el("i", { className: "dot" + (d.busy ? " busy" : ""), title: d.busy ? "working now" : "idle" }), text);
  }
}

function addDoor() {
  const used = S.routes.map(r => r.port); let port = 11436; while (used.includes(port)) port++;
  S.routes.push({ port, kind: "llm", profile: "plain", max_tokens: null, allow: { vision: true, reasoning: true, tools: true } });
  if (S.priority) S.priority.push({ door: port });
  openDoors.add(S.routes.length - 1); drawDoors(); drawPriority(); updateBar();
}

// ---------- priority list
const lineName = line => line.label || (line.door !== undefined
  ? (() => { const r = S.routes.find(x => x.port === line.door); return (r ? SHORT[purposeOf(r)] : "Door") + " door"; })()
  : (line.agent ? line.agent.split(/[\s\/]/)[0] + " · " : "") + line.client);
const lineDetail = line => line.door !== undefined ? "port " + line.door : (line.agent ? line.client + " · " + line.agent.slice(0, 40) : "any program");
let dragFrom = null;

function setRules(list) { S.priority = list; drawPriority(); updateBar(); }

function drawPriority() {
  const list = document.getElementById("plist"); list.replaceChildren();
  const all = rules();
  all.forEach((line, i) => {
    const grip = el("button", { type: "button", className: "grip", textContent: "⠿", ariaLabel: "Move line (Alt+Up or Alt+Down)", title: "Drag to move" });
    const drop = el("input", { type: "checkbox", checked: !!line.drop });
    drop.onchange = () => {
      const next = clone(all); next[i].drop = drop.checked;
      setRules([...next.filter(l => l.drop), ...next.filter(l => !l.drop)]);  // drop-everything lines always sit on top
    };
    const rm = el("button", { type: "button", className: "rm", textContent: "×", ariaLabel: "Remove line", onclick: () => setRules(clone(all).filter((_, n) => n !== i)) });
    const li = el("li", { className: "pr" + (line.drop ? " danger" : ""), draggable: false },
      grip, el("span", { className: "what" }, el("b", { textContent: lineName(line) }), " ", el("span", { className: "dim", textContent: lineDetail(line) })),
      el("label", { className: "dng" }, drop, " Drop everything"), rm);
    grip.onmousedown = () => { li.draggable = true; };
    li.ondragstart = e => { dragFrom = i; li.classList.add("drag"); e.dataTransfer.effectAllowed = "move"; };
    li.ondragend = () => { li.draggable = false; li.classList.remove("drag"); dragFrom = null; };
    li.ondragover = e => { if (dragFrom !== null) e.preventDefault(); };
    li.ondrop = e => {
      e.preventDefault(); if (dragFrom === null || dragFrom === i) return;
      const next = clone(all), [moved] = next.splice(dragFrom, 1); next.splice(i, 0, moved);
      setRules([...next.filter(l => l.drop), ...next.filter(l => !l.drop)]);
    };
    grip.onkeydown = e => {
      if (!e.altKey || !["ArrowUp", "ArrowDown"].includes(e.key)) return;
      const to = i + (e.key === "ArrowUp" ? -1 : 1); if (to < 0 || to >= all.length) return;
      e.preventDefault(); const next = clone(all), [moved] = next.splice(i, 1); next.splice(to, 0, moved);
      setRules([...next.filter(l => l.drop), ...next.filter(l => !l.drop)]);
      list.children[Math.min(to, list.children.length - 1)]?.querySelector(".grip")?.focus();
    };
    list.append(li);
  });
  list.append(el("li", { className: "else" }, el("span", { className: "what" }, el("b", { textContent: "Everything else" }), " ", el("span", { className: "dim", textContent: "waits in turn, first come first served" }))));
}

async function suggestions() {
  const q = document.getElementById("padd").value.trim().toLowerCase(), have = rules();
  const out = [];
  try { CLIENTS = await (await fetch("/api/clients")).json(); } catch (e) { /* the list is only a convenience */ }
  for (const c of CLIENTS) {
    if (have.some(l => l.client === c.client && (l.agent || "") === c.agent.split(/[\s\/]/)[0])) continue;
    const program = c.agent.split(/[\s\/]/)[0];  // the program name, not the whole version-stamped string, so a line outlives updates
    const line = { client: c.client, ...(program ? { agent: program } : {}) };
    const text = `${lineName(line)} · ${c.client}`;
    if ((text + c.agent).toLowerCase().includes(q)) out.push([text, `${c.count} request${c.count === 1 ? "" : "s"} · ${ago(c.last_seen)}`, line]);
  }
  for (const r of S.routes) {
    if (have.some(l => l.door === r.port)) continue;
    const line = { door: r.port }, text = lineName(line) + " · port " + r.port;
    if (text.toLowerCase().includes(q)) out.push([text, "a door", line]);
  }
  return out;
}

async function showSuggestions() {
  const box = document.getElementById("psug"), items = await suggestions();
  box.replaceChildren(...items.map(([text, note, line]) => el("li", { onmousedown: e => { e.preventDefault(); setRules([...rules(), line]); document.getElementById("padd").value = ""; box.hidden = true; } },
    el("span", { textContent: text }), el("small", { textContent: note }))));
  box.hidden = !items.length;
}

// ---------- models
const modelNames = m => Object.entries(S.aliases).filter(([, v]) => [m.id, m.path.toLowerCase()].includes(v.toLowerCase())).map(([n]) => n);
const allNames = () => new Set(Object.keys(S.aliases));

function suggestNames(m) {
  const family = m.id.replace(/([-_.](q\d[\w]*|f16|bf16|fp16|f32|iq\d[\w]*))+$/i, "");
  const parts = m.path.split(/[\\/]/), publisher = parts.length > 3 ? parts[parts.length - 3].toLowerCase() : "";
  const family2 = family.replace(/-gguf$/i, "");
  return [...new Set([family2, m.id, publisher && publisher + "/" + family2])].filter(n => n && !allNames().has(n));
}

function capPill(kind, name) {
  const p = withIcon(ICON[kind], "span", { className: "cap " + kind });
  p.append(name);
  return p;
}

function drawModels() {
  const body = document.getElementById("models-body"); body.replaceChildren();
  if (!MODELS.length) body.append(el("tr", {}, el("td", { colSpan: 5, className: "dim", textContent: "No models found in the model folders yet. Add one below, or put .gguf files in a model folder." })));
  for (const m of MODELS) {
    const pills = FEATURES.filter(([k]) => m.capabilities[k]).map(([k, name]) => capPill(k, name));
    const chips = modelNames(m).map(n => el("span", { className: "chip" }, n, el("button", { type: "button", className: "x", textContent: "×", ariaLabel: "Remove name " + n, onclick: () => { delete S.aliases[n]; drawModels(); updateBar(); } })));
    const add = el("button", { type: "button", className: "chip add", textContent: "+ name", onclick: () => nameDialog(m) });
    const radio = el("input", { type: "radio", name: "default-model", checked: S.default_model.toLowerCase() === m.id, ariaLabel: "Default model " + m.file });
    radio.onchange = () => { S.default_model = m.id; updateBar(); };
    const trash = withIcon(ICON.trash, "button", { type: "button", className: "trash", title: "Delete from disk", ariaLabel: "Delete " + m.file, onclick: () => deleteDialog(m) });
    body.append(el("tr", { className: "model" }, el("td", {}, radio),
      el("td", {}, el("b", { textContent: m.file.replace(/\.gguf$/i, "") }), el("div", { className: "caps" }, pills.length ? pills : el("span", { className: "dim", textContent: "text only" }))),
      el("td", { className: "nowrap", textContent: gb(m.size) }), el("td", {}, chips, add), el("td", {}, trash)));
  }
}

function nameDialog(m) {
  const dlg = document.getElementById("namedlg"), box = document.getElementById("presets"), custom = document.getElementById("custom"), err = document.getElementById("dlgerr");
  document.getElementById("dlgmodel").textContent = m.file; custom.value = ""; err.textContent = "";
  const add = name => {
    name = name.trim();
    if (!name) { err.textContent = "Type a name first."; return false; }
    if (allNames().has(name) || MODELS.some(x => x.id === name.toLowerCase() && x !== m)) { err.textContent = "That name is already in use."; return false; }
    S.aliases[name] = m.id; drawModels(); updateBar(); dlg.close(); return true;
  };
  const names = suggestNames(m);
  box.replaceChildren(...(names.length ? names.map(n => el("button", { type: "button", className: "chip pre", textContent: n, onclick: () => add(n) })) : [el("span", { className: "dim", textContent: "Every suggestion is already in use." })]));
  document.getElementById("addname").onclick = e => { e.preventDefault(); add(custom.value); };
  dlg.showModal(); custom.focus();
}

function deleteDialog(m) {
  const dlg = document.getElementById("deldlg"), err = document.getElementById("delerr"); err.textContent = "";
  document.getElementById("delmodel").textContent = m.file;
  document.getElementById("delok").onclick = async e => {
    e.preventDefault();
    const res = await fetch("/api/models/" + encodeURIComponent(m.id), { method: "DELETE" });
    if (!res.ok) { err.textContent = (await res.json()).error; return; }
    for (const [n, v] of Object.entries(S.aliases)) if ([m.id, m.path.toLowerCase()].includes(v.toLowerCase())) delete S.aliases[n];
    if (S.default_model.toLowerCase() === m.id) S.default_model = "";
    const base = JSON.parse(S0);  // the server already saved this; keep the baseline in step so it is not counted as unsaved
    for (const [n, v] of Object.entries(base.aliases)) if ([m.id, m.path.toLowerCase()].includes(v.toLowerCase())) delete base.aliases[n];
    if (base.default_model.toLowerCase() === m.id) base.default_model = "";
    S0 = JSON.stringify(base); dlg.close(); await loadModels(); updateBar();
  };
  dlg.showModal();
}

async function loadModels() { MODELS = await (await fetch("/api/models")).json(); drawModels(); }

async function findFiles() {
  const out = document.getElementById("find-out"), url = document.getElementById("hf-url").value.trim();
  out.replaceChildren(el("span", { className: "dim", textContent: "Looking…" }));
  const res = await fetch("/api/models/find", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
  const data = await res.json();
  if (!res.ok) { out.replaceChildren(el("span", { className: "err", textContent: data.error })); return; }
  found = data;
  if (!data.files.length) { out.replaceChildren(el("span", { className: "dim", textContent: "No .gguf files in " + data.repo })); return; }
  const vision = data.vision[0], withVision = el("input", { type: "checkbox", checked: true });
  const rows = data.files.map((f, i) => el("label", { className: "pick" }, el("input", { type: "radio", name: "pick-file", value: f.file, checked: i === 0 }), ` ${f.file} `, el("span", { className: "dim", textContent: gb(f.size) })));
  const go = el("button", { type: "button", className: "act", textContent: "Download" });
  go.onclick = async () => {
    const file = out.querySelector("input[name=pick-file]:checked").value;
    const res2 = await fetch("/api/models/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ repo: data.repo, file, ...(vision && withVision.checked ? { vision: vision.file } : {}) }) });
    if (!res2.ok) { document.getElementById("dl-box").replaceChildren(el("span", { className: "err", textContent: (await res2.json()).error })); return; }
    out.replaceChildren(); pollDownloads();
  };
  out.replaceChildren(el("div", { className: "dim", textContent: data.repo }), ...rows, vision ? el("label", { className: "pick" }, withVision, ` Also download its vision file (${vision.file}, ${gb(vision.size)})`) : null, go);
}

async function pollDownloads() {
  clearTimeout(pollTimer);
  downloads = await (await fetch("/api/models/downloads")).json();
  document.getElementById("dl-box").replaceChildren(...downloads.slice(-3).map(d => el("div", { className: "dl" },
    `${d.repo}/${d.file}: `, d.status === "running" ? Math.round(100 * d.done / Math.max(1, d.total)) + "% of " + gb(d.total)
      : d.status === "done" ? "done" : el("span", { className: "err", textContent: "failed (" + d.error + "), choose it again to resume" }))));
  if (downloads.some(d => d.status === "running")) pollTimer = setTimeout(pollDownloads, 1000);
  else if (downloads.length) loadModels();
}

// ---------- the rest
function numField(label, key, unit, hint) {
  const input = el("input", { type: "number", value: S[key], ariaLabel: label });
  input.oninput = () => { S[key] = +input.value; updateBar(); };
  return el("div", { className: "frow" }, el("div", { className: "what" }, el("b", { textContent: label }), hint ? el("span", { className: "dim", textContent: hint }) : null), el("span", {}, input, unit ? " " + unit : ""));
}
function textField(label, key, hint) {
  const input = el("input", { type: "text", value: S[key], ariaLabel: label });
  input.oninput = () => { S[key] = input.value; updateBar(); };
  return el("div", { className: "frow" }, el("div", { className: "what" }, el("b", { textContent: label }), hint ? el("span", { className: "dim", textContent: hint }) : null), input);
}

function drawFields() {
  document.getElementById("gpu-fields").replaceChildren(
    numField("Unload the model when idle for", "idle_minutes", "minutes", "Frees the GPU for images."),
    numField("Tell an image app \"GPU busy\" after waiting", "image_wait_seconds", "seconds"));
  document.getElementById("gpu-adv").replaceChildren(
    numField("CPU threads", "threads", "", "Fewer is often faster when the GPU does the work."),
    numField("Room kept free for a reply", "reply_reserve", "tokens"),
    numField("Most one picture may cost", "image_max_tokens", "tokens"));
  document.getElementById("hist-fields").replaceChildren(numField("Keep prompts and replies for the newest", "keep_content", "requests", "Older ones keep their numbers but lose their text."));
  document.getElementById("adv-fields").replaceChildren(
    textField("ComfyUI address", "comfy_url"), numField("Engine port", "engine_port"), textField("Engine folder", "engine_dir"),
    numField("Settings page port", "ui_port"), textField("Data folder", "data_dir"));
  const folders = document.getElementById("folder-list"); folders.replaceChildren();
  S.model_folders.forEach((f, i) => {
    const path = el("input", { type: "text", value: f.path, ariaLabel: "Model folder" });
    path.oninput = () => { f.path = path.value; updateBar(); };
    folders.append(el("div", { className: "frow" }, path, el("button", { type: "button", className: "rm", textContent: "×", ariaLabel: "Remove folder", onclick: () => { S.model_folders.splice(i, 1); drawFields(); updateBar(); } })));
  });
  document.getElementById("folder-count").textContent = S.model_folders.length;
  document.getElementById("raw").value = JSON.stringify(S, null, 2);
}

function applyRaw() {
  const err = document.getElementById("raw-err"); err.textContent = "";
  try { S = JSON.parse(document.getElementById("raw").value); } catch (e) { err.textContent = "That is not valid JSON."; return; }
  drawAll();
}

async function saveAll() {
  const msg = document.getElementById("save-msg"); msg.className = ""; msg.textContent = "";
  const res = await fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(S) });
  if (!res.ok) { msg.className = "err"; msg.textContent = (await res.json()).error; return; }
  S = await res.json(); S0 = JSON.stringify(S); drawAll(); msg.className = "ok"; msg.textContent = "Saved"; setTimeout(() => { msg.textContent = ""; }, 3000);
}

function discard() { S = JSON.parse(S0); openDoors.clear(); drawAll(); }

function drawAll() { drawDoors(); drawPriority(); drawModels(); drawFields(); updateBar(); }

async function loadSettings() {
  S = await (await fetch("/api/settings")).json(); S0 = JSON.stringify(S);
  await loadModels(); drawAll(); pollDownloads();
  if (window.tocSync) tocSync();  // the sections just got their real heights
}

async function clearText() {
  const res = await fetch("/api/history/clear-text", { method: "POST" });
  document.getElementById("clear-msg").textContent = res.ok ? "Cleared." : "Could not clear.";
}

function initSettings() {
  const add = document.getElementById("padd"), box = document.getElementById("psug");
  add.onfocus = add.onclick = add.oninput = showSuggestions;
  add.onblur = () => { box.hidden = true; };
  add.onkeydown = e => {
    if (e.key === "Escape") box.hidden = true;
    if (e.key === "Enter" && add.value.trim()) {
      e.preventDefault();
      if (!/^[0-9a-fA-F:.]+(\/\d+)?$/.test(add.value.trim())) { box.replaceChildren(el("li", { textContent: "Type an IP address (or range such as 192.168.1.0/24), or pick a suggestion." })); box.hidden = false; return; }
      setRules([...rules(), { client: add.value.trim() }]); add.value = ""; box.hidden = true;
    }
  };
  document.getElementById("add-door").onclick = addDoor;
  document.getElementById("find-btn").onclick = findFiles;
  document.getElementById("add-folder").onclick = () => { S.model_folders.push({ path: "", label: "" }); drawFields(); updateBar(); };
  document.getElementById("raw-apply").onclick = applyRaw;
  document.getElementById("save-btn").onclick = saveAll;
  document.getElementById("discard-btn").onclick = discard;
  document.getElementById("clear-btn").onclick = clearText;
  document.querySelectorAll("#namedlg button[value=cancel], #deldlg button[value=cancel]").forEach(b => b.onclick = () => b.closest("dialog").close());
  document.querySelectorAll("dialog").forEach(d => d.addEventListener("click", e => { if (e.target === d) d.close(); }));  // a click on the backdrop lands on the dialog element itself
  initToc();
}

// ---------- side navigation: smooth jump, and a highlight that slides to the section in view
function initToc() {
  const toc = document.querySelector(".toc"), links = [...toc.querySelectorAll("a")];
  const calm = matchMedia("(prefers-reduced-motion: reduce)");
  const pill = el("i", { className: "pill", ariaHidden: "true" }); toc.prepend(pill);
  const sections = links.map(a => document.querySelector(a.getAttribute("href")));
  let current = null;
  const mark = a => {
    if (a === current) return; current = a;
    links.forEach(l => l.classList.toggle("on", l === a));
    pill.style.transform = `translateY(${a.offsetTop}px)`; pill.style.height = a.offsetHeight + "px"; pill.style.opacity = 1;
  };
  const spy = () => {
    if (!toc.offsetParent) return;  // the Settings tab is not showing, so nothing has a position yet
    // the last section whose top has passed the upper third of the window
    let k = 0; sections.forEach((s, n) => { if (s.getBoundingClientRect().top < innerHeight / 3) k = n; });
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) k = sections.length - 1;
    mark(links[k]);
  };
  links.forEach((a, n) => a.onclick = e => {
    e.preventDefault(); sections[n].scrollIntoView({ behavior: calm.matches ? "auto" : "smooth", block: "start" }); mark(a);
  });
  addEventListener("scroll", spy, { passive: true }); addEventListener("resize", () => { current = null; spy(); });
  window.tocSync = () => { current = null; spy(); };
  new ResizeObserver(() => spy()).observe(document.body);  // sections fill in after the tab opens and the data loads, which moves them without a scroll
  spy();
}
