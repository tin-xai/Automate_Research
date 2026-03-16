const searchInput = document.getElementById("searchInput");
const countPill = document.getElementById("countPill");
const resultList = document.getElementById("resultList");
const sourceFilters = document.getElementById("sourceFilters");
const clearFiltersBtn = document.getElementById("clearFiltersBtn");
const summaryText = document.getElementById("summaryText");
const emptyState = document.getElementById("emptyState");
const detailCard = document.getElementById("detailCard");
const detailHeading = document.getElementById("detailHeading");
const detailMeta = document.getElementById("detailMeta");
const detailNote = document.getElementById("detailNote");
const detailBody = document.getElementById("detailBody");
const onlySourceBtn = document.getElementById("onlySourceBtn");
const copyBtn = document.getElementById("copyBtn");

const initialParams = new URLSearchParams(window.location.search);
const initialQuery = initialParams.get("q") || "";
const initialSources = new Set(initialParams.getAll("source"));

let sources = [];
let allTricks = [];
let tricks = [];
let selectedId = null;
let selectedSources = new Set();
let searchTimer = null;
let currentQuery = "";
let dataMode = "api";

function escapeHtml(input) {
  return String(input)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function isEditableTarget(target) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target?.isContentEditable
  );
}

function setSummary() {
  const totalFiles = sources.length;
  const totalTricks = allTricks.length;
  const mode = dataMode === "static" ? " • static mode" : "";
  summaryText.textContent = `${totalTricks} tricks across ${totalFiles} files${mode}`;
}

function setCount(total) {
  const suffix = total === 1 ? "result" : "results";
  countPill.textContent = `${total} ${suffix}`;
}

function updateUrlState() {
  const params = new URLSearchParams();
  if (currentQuery) {
    params.set("q", currentQuery);
  }
  for (const source of selectedSources) {
    params.append("source", source);
  }
  const url = params.toString() ? `${window.location.pathname}?${params.toString()}` : window.location.pathname;
  window.history.replaceState({}, "", url);
}

function normalizeText(entry) {
  return `${entry.heading || ""} ${entry.note || ""} ${entry.code || ""}`.toLowerCase();
}

function rankTricks(entries, query) {
  const q = query.trim().toLowerCase();
  if (!q) {
    return [...entries].sort((a, b) => a.id - b.id);
  }

  const tokens = q.split(/\s+/).filter(Boolean);
  const scored = [];
  for (const entry of entries) {
    const text = normalizeText(entry);
    let score = 0;
    for (const token of tokens) {
      if (text.includes(token)) {
        score += 2;
      }
    }
    if (text.includes(q)) {
      score += 3;
    }
    if ((entry.heading || "").toLowerCase().startsWith(q)) {
      score += 2;
    }
    if (score > 0) {
      scored.push({ score, entry });
    }
  }
  scored.sort((a, b) => b.score - a.score || a.entry.id - b.entry.id);
  return scored.map((item) => item.entry);
}

function buildSourcesFromTricks(items) {
  const counts = new Map();
  for (const item of items) {
    const key = item.source_rel;
    const existing = counts.get(key);
    if (existing) {
      existing.count += 1;
      continue;
    }
    counts.set(key, {
      key,
      rel_path: item.source_rel,
      name: item.source_name || item.source_rel,
      display: item.source_display || item.source_name || item.source_rel,
      group: item.source_group || "root",
      count: 1,
    });
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || a.rel_path.localeCompare(b.rel_path));
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${url} (${response.status})`);
  }
  return response.json();
}

function sourceSort(a, b) {
  if (a.group !== b.group) {
    return a.group.localeCompare(b.group);
  }
  return (a.display || a.name || "").localeCompare(b.display || b.name || "");
}

function groupedSources(list) {
  const groups = new Map();
  for (const source of [...list].sort(sourceSort)) {
    const group = source.group || "root";
    if (!groups.has(group)) {
      groups.set(group, []);
    }
    groups.get(group).push(source);
  }
  return [...groups.entries()];
}

function renderSourceFilters() {
  if (!sources.length) {
    sourceFilters.innerHTML = '<p class="no-result">No files found.</p>';
    return;
  }

  const total = sources.reduce((sum, item) => sum + (item.count || 0), 0);
  const allChecked = selectedSources.size === 0;
  const groups = groupedSources(sources);

  let html = `
    <div class="source-all-row">
      <button type="button" class="source-all-btn ${allChecked ? "active" : ""}" data-all="1">
        <span>All Files</span>
        <span class="source-count">${total}</span>
      </button>
    </div>
  `;

  for (const [group, items] of groups) {
    html += `<section class="source-group"><h4>${escapeHtml(group)}</h4>`;
    for (const source of items) {
      const active = selectedSources.has(source.key);
      const label = escapeHtml(source.display || source.name || source.rel_path);
      html += `
        <button type="button" class="source-item ${active ? "active" : ""}" data-key="${escapeHtml(source.key)}">
          <span class="source-label">${label}</span>
          <span class="source-count">${source.count}</span>
        </button>
      `;
    }
    html += "</section>";
  }

  sourceFilters.innerHTML = html;
  clearFiltersBtn.classList.toggle("hidden", selectedSources.size === 0);
}

function renderList(items) {
  if (!items.length) {
    resultList.innerHTML = '<p class="no-result">No tricks match this search/filter.</p>';
    setCount(0);
    showDetail(null);
    return;
  }

  setCount(items.length);
  const groupOrder = [];
  const grouped = new Map();
  for (const item of items) {
    if (!grouped.has(item.source_rel)) {
      grouped.set(item.source_rel, []);
      groupOrder.push(item.source_rel);
    }
    grouped.get(item.source_rel).push(item);
  }

  const parts = [];
  for (const sourceKey of groupOrder) {
    const sourceItems = grouped.get(sourceKey) || [];
    if (!sourceItems.length) {
      continue;
    }
    const first = sourceItems[0];
    const sourceTitle = escapeHtml(first.source_display || first.source_name || first.source_rel);
    const sourcePath = escapeHtml(first.source_rel);
    parts.push(`
      <div class="source-group-header">
        <div>
          <strong>${sourceTitle}</strong>
          <span class="source-path">${sourcePath}</span>
        </div>
        <span class="source-group-count">${sourceItems.length}</span>
      </div>
    `);

    for (const item of sourceItems) {
      const activeClass = selectedId === item.id ? "result-card active" : "result-card";
      const preview = escapeHtml(item.preview || "");
      const heading = escapeHtml(item.heading || "(untitled)");
      parts.push(`
        <button class="${activeClass}" data-id="${item.id}" type="button">
          <span class="result-id">#${item.id}</span>
          <span class="result-heading">${heading}</span>
          <code class="result-preview">${preview}</code>
        </button>
      `);
    }
  }

  resultList.innerHTML = parts.join("");
}

function normalizePath(path) {
  const segments = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") {
      continue;
    }
    if (part === "..") {
      segments.pop();
      continue;
    }
    segments.push(part);
  }
  return segments.join("/");
}

function toFileHref(relPath) {
  const normalized = normalizePath(relPath);
  if (!normalized) {
    return "";
  }
  return `files/${normalized.split("/").map((segment) => encodeURIComponent(segment)).join("/")}`;
}

function resolveAssetUrl(sourceRel, rawTarget) {
  const target = String(rawTarget || "").trim();
  if (!target) {
    return "";
  }
  const lower = target.toLowerCase();
  if (lower.startsWith("http://") || lower.startsWith("https://") || lower.startsWith("data:")) {
    return target;
  }
  if (target.startsWith("/")) {
    return target;
  }

  const sourceDir = sourceRel.includes("/") ? sourceRel.slice(0, sourceRel.lastIndexOf("/")) : "";
  const merged = sourceDir ? `${sourceDir}/${target}` : target;
  return toFileHref(merged);
}

function parseInlineMarkdown(text, sourceRel) {
  let html = escapeHtml(text);
  html = html.replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    (_, alt, target) =>
      `<img class="md-inline-image" alt="${escapeHtml(alt)}" src="${escapeHtml(resolveAssetUrl(sourceRel, target))}" loading="lazy">`,
  );
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    (_, label, target) =>
      `<a href="${escapeHtml(resolveAssetUrl(sourceRel, target))}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`,
  );
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  return html;
}

function renderMarkdownToHtml(markdown, sourceRel) {
  const lines = markdown.replaceAll("\r\n", "\n").split("\n");
  const out = [];
  let inFence = false;
  let fenceBuffer = [];
  let inIndentCode = false;
  let indentBuffer = [];
  let inList = false;
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) {
      return;
    }
    out.push(`<p>${parseInlineMarkdown(paragraph.join(" "), sourceRel)}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!inList) {
      return;
    }
    out.push("</ul>");
    inList = false;
  }

  function flushIndentCode() {
    if (!inIndentCode) {
      return;
    }
    out.push(`<pre>${escapeHtml(indentBuffer.join("\n"))}</pre>`);
    indentBuffer = [];
    inIndentCode = false;
  }

  for (const line of lines) {
    const trimmed = line.trim();

    if (inFence) {
      if (trimmed.startsWith("```")) {
        out.push(`<pre>${escapeHtml(fenceBuffer.join("\n"))}</pre>`);
        fenceBuffer = [];
        inFence = false;
      } else {
        fenceBuffer.push(line);
      }
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      flushList();
      flushIndentCode();
      inFence = true;
      fenceBuffer = [];
      continue;
    }

    const isIndentCode = line.startsWith("    ") || line.startsWith("\t");
    if (isIndentCode) {
      flushParagraph();
      flushList();
      inIndentCode = true;
      indentBuffer.push(line.replace(/^\t/, "    ").replace(/^ {4}/, ""));
      continue;
    }
    flushIndentCode();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      flushParagraph();
      flushList();
      out.push("<hr>");
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.*)$/.exec(trimmed);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = Math.min(6, headingMatch[1].length);
      out.push(`<h${level}>${parseInlineMarkdown(headingMatch[2], sourceRel)}</h${level}>`);
      continue;
    }

    const imageOnlyMatch = /^!\[([^\]]*)\]\(([^)]+)\)$/.exec(trimmed);
    if (imageOnlyMatch) {
      flushParagraph();
      flushList();
      out.push(
        `<figure><img alt="${escapeHtml(imageOnlyMatch[1])}" src="${escapeHtml(resolveAssetUrl(sourceRel, imageOnlyMatch[2]))}" loading="lazy"></figure>`,
      );
      continue;
    }

    const listMatch = /^-\s+(.*)$/.exec(trimmed);
    if (listMatch) {
      flushParagraph();
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${parseInlineMarkdown(listMatch[1], sourceRel)}</li>`);
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  flushIndentCode();

  if (inFence) {
    out.push(`<pre>${escapeHtml(fenceBuffer.join("\n"))}</pre>`);
  }

  return out.join("");
}

function renderBody(trick) {
  if ((trick.source_name || "").toLowerCase().endsWith(".md")) {
    detailBody.classList.add("markdown");
    detailBody.innerHTML = renderMarkdownToHtml(trick.code || "", trick.source_rel || "");
    return;
  }
  detailBody.classList.remove("markdown");
  detailBody.innerHTML = `<pre>${escapeHtml(trick.code || "")}</pre>`;
}

function showDetail(trick) {
  if (!trick) {
    detailCard.classList.add("hidden");
    emptyState.classList.remove("hidden");
    detailHeading.textContent = "";
    detailMeta.textContent = "";
    detailNote.textContent = "";
    detailBody.innerHTML = "";
    onlySourceBtn.dataset.source = "";
    return;
  }

  emptyState.classList.add("hidden");
  detailCard.classList.remove("hidden");
  detailHeading.textContent = trick.heading || "(untitled)";
  detailMeta.textContent = `${trick.source_rel}:${trick.line} • Trick #${trick.id}`;
  detailNote.textContent = trick.note || "";
  renderBody(trick);
  onlySourceBtn.dataset.source = trick.source_rel;
  onlySourceBtn.disabled = selectedSources.size === 1 && selectedSources.has(trick.source_rel);
}

function selectTrickById(id, shouldScroll = false) {
  selectedId = id;
  const selected = tricks.find((item) => item.id === selectedId) || null;
  renderList(tricks);
  showDetail(selected);
  if (shouldScroll && selected) {
    const card = resultList.querySelector(`button[data-id="${selected.id}"]`);
    card?.scrollIntoView({ block: "nearest" });
  }
}

function navigateSelection(delta) {
  if (!tricks.length) {
    return;
  }
  const currentIndex = tricks.findIndex((item) => item.id === selectedId);
  const start = currentIndex < 0 ? (delta > 0 ? -1 : 0) : currentIndex;
  const nextIndex = (start + delta + tricks.length) % tricks.length;
  selectTrickById(tricks[nextIndex].id, true);
}

function applyFilters(query = "") {
  currentQuery = query.trim();
  let subset = allTricks;
  if (selectedSources.size > 0) {
    subset = subset.filter((item) => selectedSources.has(item.source_rel));
  }
  tricks = rankTricks(subset, currentQuery);

  const selectedStillExists = tricks.some((item) => item.id === selectedId);
  if (!selectedStillExists) {
    selectedId = null;
  }
  if (selectedId === null && tricks.length > 0 && (currentQuery || selectedSources.size > 0)) {
    selectedId = tricks[0].id;
  }

  renderList(tricks);
  const selected = tricks.find((item) => item.id === selectedId) || null;
  showDetail(selected);
  updateUrlState();
}

function runSearchDebounced() {
  const query = searchInput.value.trim();
  if (searchTimer) {
    clearTimeout(searchTimer);
  }
  searchTimer = setTimeout(() => {
    applyFilters(query);
  }, 120);
}

async function loadData() {
  try {
    const [sourcesData, tricksData] = await Promise.all([
      fetchJson("api/sources"),
      fetchJson("api/tricks"),
    ]);
    dataMode = "api";
    sources = sourcesData.items || [];
    allTricks = tricksData.items || [];
  } catch {
    const [sourcesData, tricksData] = await Promise.all([
      fetchJson("sources.json"),
      fetchJson("tricks.json"),
    ]);
    dataMode = "static";
    sources = sourcesData.items || [];
    allTricks = tricksData.items || [];
  }

  if (!sources.length) {
    sources = buildSourcesFromTricks(allTricks);
  }
  allTricks.sort((a, b) => a.id - b.id);
  const available = new Set(sources.map((item) => item.key));
  selectedSources = new Set([...initialSources].filter((item) => available.has(item)));
}

sourceFilters.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) {
    return;
  }
  if (button.dataset.all === "1") {
    selectedSources.clear();
  } else if (button.dataset.key) {
    const key = button.dataset.key;
    if (selectedSources.has(key)) {
      selectedSources.delete(key);
    } else {
      selectedSources.add(key);
    }
  }
  renderSourceFilters();
  applyFilters(searchInput.value.trim());
});

clearFiltersBtn.addEventListener("click", () => {
  selectedSources.clear();
  renderSourceFilters();
  applyFilters(searchInput.value.trim());
});

resultList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) {
    return;
  }
  selectTrickById(Number(button.dataset.id));
});

onlySourceBtn.addEventListener("click", () => {
  const sourceKey = onlySourceBtn.dataset.source;
  if (!sourceKey) {
    return;
  }
  selectedSources = new Set([sourceKey]);
  renderSourceFilters();
  applyFilters(searchInput.value.trim());
});

copyBtn.addEventListener("click", async () => {
  const selected = tricks.find((item) => item.id === selectedId);
  if (!selected) {
    return;
  }
  try {
    await navigator.clipboard.writeText(selected.code || "");
    copyBtn.textContent = "Copied";
    setTimeout(() => {
      copyBtn.textContent = "Copy Command";
    }, 1000);
  } catch {
    copyBtn.textContent = "Copy Failed";
    setTimeout(() => {
      copyBtn.textContent = "Copy Command";
    }, 1000);
  }
});

searchInput.addEventListener("input", runSearchDebounced);

document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !isEditableTarget(event.target)) {
    event.preventDefault();
    searchInput.focus();
    searchInput.select();
    return;
  }

  if (event.key === "Escape" && document.activeElement === searchInput) {
    searchInput.value = "";
    applyFilters("");
    return;
  }

  if (isEditableTarget(event.target)) {
    return;
  }

  if (event.key === "j") {
    event.preventDefault();
    navigateSelection(1);
    return;
  }
  if (event.key === "k") {
    event.preventDefault();
    navigateSelection(-1);
    return;
  }
  if (event.key === "Enter" && selectedId === null && tricks.length > 0) {
    event.preventDefault();
    selectTrickById(tricks[0].id, true);
  }
});

async function init() {
  searchInput.value = initialQuery;
  await loadData();
  renderSourceFilters();
  setSummary();
  applyFilters(initialQuery);
}

init().catch((error) => {
  resultList.innerHTML = `<p class="no-result">${escapeHtml(error.message)}</p>`;
  summaryText.textContent = "Could not load tricks.";
});
