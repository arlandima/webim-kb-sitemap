// Webim AI Knowledge — интерфейс без сборки и внешних зависимостей. Весь пользовательский текст экранируется.
const $ = (s, r = document) => r.querySelector(s);
const app = $("#app");
const state = { history: [], suggestions: null, tree: null, current: null, abort: null, lastPayload: null };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ICON_EXT = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2h6v6M10 2 3 9" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
const ICON_SEND = '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
async function suggestions() {
  if (!state.suggestions) {
    try { state.suggestions = await api("/api/suggestions"); } catch { state.suggestions = { featured: [], demo: [], topics: [] }; }
  }
  return state.suggestions;
}

/* ---------- маршрутизация ---------- */
function go(url, replace = false) {
  history[replace ? "replaceState" : "pushState"]({}, "", url);
  route();
}
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-link]");
  if (a && !e.metaKey && !e.ctrlKey && !e.shiftKey && a.target !== "_blank") {
    e.preventDefault();
    go(a.getAttribute("href"));
    window.scrollTo(0, 0);
  }
});
window.addEventListener("popstate", route);
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
    const t = $("textarea.q-input");
    if (t) { e.preventDefault(); t.focus(); }
  }
  if (e.key === "Escape") closeDrawer();
});

function setNav(name) {
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
}

async function route() {
  closeDrawer();
  if (state.abort) { state.abort.abort(); state.abort = null; }
  const u = new URL(location.href);
  const p = u.pathname.replace(/\/+$/, "") || "/";
  if (p === "/browse") { setNav("browse"); return viewBrowse(u); }
  if (p === "/demo") { setNav("demo"); return viewDemo(); }
  if (p === "/admin") { setNav(""); return viewAdmin(); }
  setNav("home");
  const q = u.searchParams.get("q");
  if (q) return viewResults(q);
  return viewHome();
}

/* ---------- поле вопроса ---------- */
function askBox({ id = "q", value = "", placeholder = "Задайте вопрос своими словами — например, как поставить диалог на паузу?", btn = "Найти ответ", rows = 1 } = {}) {
  return `<form class="ask" data-ask role="search" aria-label="Задать вопрос">
    <label class="sr" for="${id}" style="position:absolute;left:-999px">Ваш вопрос</label>
    <textarea id="${id}" class="q-input" rows="${rows}" maxlength="600" placeholder="${esc(placeholder)}" autocomplete="off" spellcheck="true">${esc(value)}</textarea>
    <button class="btn" type="submit">${esc(btn)} ${ICON_SEND}</button></form>`;
}
function wireAsk(root, onSubmit) {
  root.querySelectorAll("form[data-ask]").forEach((f) => {
    const ta = f.querySelector("textarea");
    const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; };
    ta.addEventListener("input", fit); fit();
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); f.requestSubmit(); } });
    f.addEventListener("submit", (e) => { e.preventDefault(); const v = ta.value.trim(); if (v) onSubmit(v, f); });
  });
}

/* ---------- главная ---------- */
async function viewHome() {
  document.title = "База знаний Webim — вопросы своими словами";
  const sg = await suggestions();
  const topics = (sg.topics || []).slice(0, 8);
  app.innerHTML = `
  <section class="hero"><div class="wrap">
    <h1>Спросите что угодно о Webim</h1>
    <p class="lead">Задайте вопрос своими словами — ответ соберётся из официальной Базы знаний, а каждый шаг будет подтверждён ссылкой на исходный раздел.</p>
    ${askBox({ rows: 2 })}
    ${topics.length ? `<div class="chips" role="list" aria-label="Разделы Базы знаний"><span class="chips-label">Разделы документации</span>${topics.map((t) => `<a role="listitem" class="chip" href="/browse?url=${encodeURIComponent(t.url)}" data-link>${esc(t.title)}</a>`).join("")}</div>` : ""}
  </div></section>
  <div class="wrap">
    ${(sg.featured || []).length ? `<section class="section"><h2>Попробуйте спросить</h2><p class="sub">Реальные вопросы — с ответами из текущей Базы знаний.</p>
      <div class="q-grid">${sg.featured.map((f) => `<a class="q-card" href="/?q=${encodeURIComponent(f.q)}" data-link><span class="q">${esc(f.q)}</span><span class="tag">${esc(f.tag || "")}</span></a>`).join("")}</div></section>` : ""}
    <section class="section" style="padding-bottom:56px"><h2>Как это работает</h2><p class="sub">Поиск не заменяет документацию — он быстрее приводит к нужному месту в ней.</p>
      <div class="how">
        <div class="item"><div class="num">1</div><h3>Понимает вопрос</h3><p>Ищет по смыслу и по точным терминам одновременно — даже если вы называете вещи иначе, чем в статьях.</p></div>
        <div class="item"><div class="num">2</div><h3>Отвечает только по документации</h3><p>Если в Базе знаний нет надёжной информации, система так и скажет — без догадок.</p></div>
        <div class="item"><div class="num">3</div><h3>Показывает источник</h3><p>Каждый ответ содержит цитаты и ссылки на исходную статью и конкретный раздел на webim.ru.</p></div>
      </div></section>
  </div>`;
  wireAsk(app, (q) => go(`/?q=${encodeURIComponent(q)}`));
}

/* ---------- результаты ---------- */
function relLevel(score) { return score >= 0.5 ? 3 : score >= 0.2 ? 2 : 1; }
const REL_TXT = { 3: "Высокая", 2: "Средняя", 1: "Возможно" };
function crumbsHtml(parts) { return parts.filter(Boolean).map((p) => esc(p)).join('<span class="sep">›</span>'); }
function hl(text, ranges) {
  if (!ranges || !ranges.length) return esc(text);
  let out = "", pos = 0;
  for (const [a, b] of [...ranges].sort((x, y) => x[0] - y[0])) {
    if (a < pos || b > text.length) continue;
    out += esc(text.slice(pos, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>";
    pos = b;
  }
  return out + esc(text.slice(pos));
}
function sectionPath(h) { return h.heading_path.length > 1 ? h.heading_path.slice(1) : []; }

// Мини-разметка ответа: **жирный**, `код`, списки, ссылки-цитаты [n]. Сначала экранируем, потом размечаем.
function renderAnswer(raw) {
  const lines = String(raw).replace(/\r/g, "").split("\n");
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d{1,2})\]/g, (m, n) => `<button type="button" class="cite" data-cite="${n}" aria-label="Источник ${n}">${n}</button>`);
  let html = "", list = null, para = [];
  const flushP = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const flushL = () => { if (list) { html += `<${list.t}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.t}>`; list = null; } };
  for (const ln of lines) {
    const ol = ln.match(/^\s*\d+[.)]\s+(.*)/), ul = ln.match(/^\s*[-•*]\s+(.*)/);
    if (ol || ul) { flushP(); const t = ol ? "ol" : "ul"; if (!list || list.t !== t) { flushL(); list = { t, items: [] }; } list.items.push((ol || ul)[1]); }
    else if (!ln.trim()) { flushP(); flushL(); }
    else { flushL(); para.push(ln.trim()); }
  }
  flushP(); flushL();
  return html;
}

function sourceCard(h) {
  const ex = hl(h.excerpt, h.highlights);
  return `<li class="card src" id="src-${h.n}" data-n="${h.n}"><div class="n">${h.n}</div><div>
    <div class="t">${esc(h.title)}</div>
    <div class="crumbs">${crumbsHtml([...h.breadcrumbs, ...sectionPath(h)])}</div>
    <p class="excerpt">${ex}</p>
    <div class="actions"><a href="${esc(h.source_url)}" target="_blank" rel="noopener">Открыть оригинал ${ICON_EXT}</a>
      <button class="link-btn" type="button" data-inspect="${h.n}" style="font-size:14px">Показать фрагмент</button></div></div></li>`;
}
function articleCard(a) {
  const b = a.best, lvl = relLevel(a.score);
  const sec = sectionPath(b).join(" › ");
  return `<a class="card art" href="${esc(b.source_url)}" target="_blank" rel="noopener">
    <div class="t"><span>${esc(a.title)}</span><span class="ext">${ICON_EXT}</span></div>
    <div class="crumbs">${crumbsHtml(a.breadcrumbs)}</div>
    ${sec ? `<div class="sec">Раздел: ${esc(sec)}</div>` : ""}
    <p class="excerpt">${hl(b.excerpt, b.highlights)}</p>
    <div class="meta-line"><span><span class="rel" data-l="${lvl}" aria-hidden="true"><i></i><i></i><i></i></span>Релевантность: ${REL_TXT[lvl]}</span><span>${a.updated ? "обновлено " + esc(a.updated.split("-").reverse().join(".")) : ""}</span></div></a>`;
}

async function viewResults(q) {
  document.title = `${q} — База знаний Webim`;
  const sg = await suggestions();
  const scen = (sg.demo || []).find((d) => d.q.trim().toLowerCase() === q.trim().toLowerCase());
  const debug = new URL(location.href).searchParams.has("debug");
  app.innerHTML = `<div class="qbar"><div class="wrap">${askBox({ value: q, btn: "Спросить" })}</div></div>
    <div class="wrap"><div class="res-grid"><div class="main" id="main-col"></div><aside class="side" id="side-col" aria-label="Релевантные статьи"></aside></div></div>`;
  wireAsk(app, (v) => { state.history = []; go(`/?q=${encodeURIComponent(v)}`); });
  const main = $("#main-col"), side = $("#side-col");
  main.innerHTML = `<div class="card answer" aria-live="polite" aria-busy="true"><div class="status"><span class="spin"></span><span id="st">Ищу по Базе знаний…</span></div><div class="skel"><i></i><i></i><i></i><i></i></div></div>`;
  side.innerHTML = `<h2>Релевантные статьи</h2><div class="card art skel" style="padding:18px"><i></i><i></i><i></i></div>`;
  await streamAnswer(q, { main, side, scen, debug, append: false });
}

async function streamAnswer(q, { main, side, scen, debug, append }) {
  const ctrl = new AbortController(); state.abort = ctrl;
  let payload = null, buf = "", final = null, answerEl = null, statusEl = null;
  const holder = append ? document.createElement("div") : main;
  if (append) { holder.innerHTML = `<div class="card answer" aria-live="polite"><div class="status"><span class="spin"></span><span>Ищу по Базе знаний…</span></div><div class="skel"><i></i><i></i><i></i></div></div>`; main.appendChild(holder); holder.scrollIntoView({ behavior: "smooth", block: "start" }); }
  const answerCard = () => holder.querySelector(".answer");
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, signal: ctrl.signal,
      body: JSON.stringify({ q, history: state.history.slice(-6), k: 5 }) });
    if (!r.ok || !r.body) throw new Error("HTTP " + r.status);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let pending = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += dec.decode(value, { stream: true });
      let i;
      while ((i = pending.indexOf("\n\n")) >= 0) {
        const block = pending.slice(0, i); pending = pending.slice(i + 2);
        const ev = /^event: (.+)$/m.exec(block)?.[1], data = /^data: (.+)$/m.exec(block)?.[1];
        if (!ev || !data) continue;
        const d = JSON.parse(data);
        if (ev === "retrieval") {
          payload = d; state.lastPayload = d;
          if (!append) renderSide(side, d);
          const c = answerCard();
          if (d.confident) {
            c.innerHTML = `<div class="answer-head"><span class="badge">Ответ</span><span class="sub">по материалам Базы знаний Webim</span></div>
              <div class="status" id="st2"><span class="spin"></span><span>Нашёл ${d.evidence.length} ${plural(d.evidence.length, "раздел", "раздела", "разделов")} документации, формирую ответ…</span></div>
              <div class="answer-body" id="ab"></div>`;
          }
        } else if (ev === "token") {
          buf += d.text;
          const ab = holder.querySelector("#ab"); if (ab) { holder.querySelector("#st2")?.remove(); ab.innerHTML = renderAnswer(buf) + '<span class="caret"></span>'; }
        } else if (ev === "final") { final = d; }
        else if (ev === "error") { throw new Error(d.message); }
      }
    }
    if (!payload) throw new Error("empty");
    finishAnswer(holder, main, payload, final, { scen, debug, q, append });
    if (append) { /* боковая колонка обновляется по последнему вопросу */ renderSide(side, payload); }
    if (final) state.history.push({ role: "user", content: q }, { role: "assistant", content: final.mode === "insufficient" ? "В Базе знаний недостаточно информации." : final.text });
  } catch (e) {
    if (e.name === "AbortError") return;
    const c = answerCard() || holder;
    c.outerHTML = `<div class="card errbox" role="alert"><h2>Не удалось получить ответ</h2><p>Возникла временная проблема при обращении к сервису. Ваш вопрос не потерян.</p><button class="btn small" type="button" data-retry>Повторить</button></div>`;
    holder.querySelector("[data-retry]")?.addEventListener("click", () => { holder.remove(); go(location.pathname + location.search, true); });
    if (!append) main.querySelector("[data-retry]")?.addEventListener("click", () => route());
  } finally { if (state.abort === ctrl) state.abort = null; }
}
function plural(n, a, b, c) { const m = n % 10, k = n % 100; return m === 1 && k !== 11 ? a : m >= 2 && m <= 4 && (k < 10 || k >= 20) ? b : c; }

function renderSide(side, d) {
  const arts = d.articles || [];
  side.innerHTML = `<h2>Релевантные статьи</h2>${arts.length ? arts.slice(0, 6).map(articleCard).join("") : `<div class="card art"><p class="excerpt">Подходящих статей не найдено.</p></div>`}`;
}

function finishAnswer(holder, main, d, final, { scen, debug, q, append }) {
  const card = holder.querySelector(".answer");
  const ev = d.evidence || [];
  const src = ev.length ? `<h2 class="h-sec">Источники <small>${ev.length} ${plural(ev.length, "раздел", "раздела", "разделов")} документации</small></h2><ol class="sources">${ev.map(sourceCard).join("")}</ol>` : "";
  let html = "";
  if (!d.confident || !final || final.mode === "insufficient") {
    const rel = (d.articles || []).slice(0, 3);
    card.outerHTML = `<div class="card insufficient"><h2>В Базе знаний Webim недостаточно информации для надёжного ответа</h2>
      <p>${(d.unknown_terms || []).length ? `В Базе знаний нет упоминаний: <strong>${d.unknown_terms.map(esc).join(", ")}</strong>. ` : ""}Мы не нашли в документации подтверждения, на которое можно опереться, поэтому не станем гадать. Ниже — материалы, которые могут быть близки по теме.</p>
      <div class="row"><a class="btn small" href="https://webim.ru/kb/" target="_blank" rel="noopener">Открыть Базу знаний ${ICON_EXT}</a>
      <button class="btn small secondary" type="button" data-similar>Посмотреть похожие материалы</button></div></div>`;
    if (rel.length) holder.insertAdjacentHTML("beforeend", `<h2 class="h-sec">Возможно, пригодится</h2><ol class="sources">${rel.map((a, i) => sourceCard({ ...a.best, n: i + 1, title: a.title, breadcrumbs: a.breadcrumbs })).join("")}</ol>`);
    holder.insertAdjacentHTML("beforeend", `<p class="crumbs" style="margin-top:18px">Если вопрос касается вашей настройки Webim, можно <a href="https://webim.ru/" target="_blank" rel="noopener">обратиться в поддержку</a>.</p>`);
  } else {
    const mode = final.mode === "llm" ? "Ответ подготовлен ИИ строго по найденным фрагментам" : "Выдержки из документации, наиболее подходящие к вопросу";
    card.setAttribute("aria-busy", "false");
    card.innerHTML = `<div class="answer-head"><span class="badge">Ответ</span><span class="sub">${esc(mode)}</span></div>
      <div class="answer-body">${renderAnswer(final.text)}</div>
      <div class="answer-foot"><button class="pill" type="button" data-inspect-all>Использовано ${ev.length} ${plural(ev.length, "источник", "источника", "источников")} — посмотреть<span aria-hidden="true">›</span></button>
      ${debug ? `<span class="foot-note">${esc(JSON.stringify(d.timings))}</span>` : ""}</div>`;
    if (scen && scen.why) card.insertAdjacentHTML("afterend", `<div class="why"><b>Почему это нашлось</b>${esc(scen.why)}</div>`);
    holder.insertAdjacentHTML("beforeend", src);
    if (!append || true) holder.insertAdjacentHTML("beforeend", `<form class="follow" data-ask role="search" aria-label="Уточняющий вопрос"><div class="ask"><label style="position:absolute;left:-999px" for="fq">Уточняющий вопрос</label><textarea id="fq" class="q-input" rows="1" maxlength="600" placeholder="Уточнить или задать следующий вопрос…"></textarea><button class="btn" type="submit">Спросить</button></div></form>`);
  }
  holder.querySelectorAll("form.follow").forEach((f) => {
    const ta = f.querySelector("textarea");
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); f.requestSubmit(); } });
    f.addEventListener("submit", (e) => { e.preventDefault(); const v = ta.value.trim(); if (!v) return; f.remove(); streamAnswer(v, { main: holder.parentElement || main, side: $("#side-col"), append: true, debug }); });
  });
  holder.addEventListener("click", (e) => {
    const c = e.target.closest("[data-cite]");
    if (c) { const el = holder.querySelector(`#src-${c.dataset.cite}`); if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1600); } }
    const one = e.target.closest("[data-inspect]");
    if (one) openInspector(d, Number(one.dataset.inspect));
    if (e.target.closest("[data-inspect-all]")) openInspector(d, null);
    if (e.target.closest("[data-similar]")) holder.querySelector(".h-sec")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

/* ---------- инспектор источников ---------- */
function openInspector(d, focusN) {
  const root = $("#drawer-root");
  const ev = d.evidence || [];
  root.innerHTML = `<div class="overlay" data-close></div><aside class="drawer" role="dialog" aria-modal="true" aria-labelledby="dr-t">
    <div class="drawer-head"><div><h2 id="dr-t">Использовано ${ev.length} ${plural(ev.length, "источник", "источника", "источников")}</h2><p>Именно эти фрагменты Базы знаний Webim легли в основу ответа.</p></div><button class="x" type="button" data-close aria-label="Закрыть">×</button></div>
    <div class="drawer-body">${ev.map((h) => `<div class="ev" id="ev-${h.n}"><div class="head"><div class="n" style="width:30px;height:30px;border-radius:9px;background:var(--tint);color:var(--indigo);font-weight:700;display:grid;place-items:center;flex:none">${h.n}</div><div>
      <div class="t" style="font-weight:700;color:var(--navy)">${esc(h.title)}</div><div class="crumbs">${crumbsHtml([...h.breadcrumbs, ...sectionPath(h)])}</div></div></div>
      <div class="quote">${hl(h.text.length > 1400 ? h.text.slice(0, 1400) + "…" : h.text, [])}</div>
      <div class="actions"><a class="btn small" href="${esc(h.source_url)}" target="_blank" rel="noopener">Открыть оригинал ${ICON_EXT}</a></div>
      <div class="url">${esc(h.source_url)}</div></div>`).join("")}</div></aside>`;
  requestAnimationFrame(() => { $(".overlay", root).classList.add("open"); $(".drawer", root).classList.add("open"); });
  root.onclick = (e) => { if (e.target.closest("[data-close]")) closeDrawer(); };
  $(".x", root).focus();
  if (focusN) { const el = $(`#ev-${focusN}`, root); if (el) setTimeout(() => { el.scrollIntoView({ block: "start" }); el.classList.add("flash"); }, 260); }
}
function closeDrawer() { const r = $("#drawer-root"); if (r) r.innerHTML = ""; }

/* ---------- обзор документации ---------- */
async function loadTree() {
  if (!state.tree) state.tree = await api("/api/tree");
  return state.tree;
}
function treeHtml(nodes, cur, depth = 0) {
  return `<ul>${nodes.map((n) => {
    const kids = n.children || [];
    const has = kids.length > 0;
    const href = `/browse?url=${encodeURIComponent(n.url)}`;
    const active = containsUrl(n, cur);
    if (!has) return `<li><a href="${href}" data-link class="${n.url === cur ? "cur" : ""}" data-t="${esc(n.title.toLowerCase())}">${esc(n.title)}</a></li>`;
    return `<li><details ${active || depth === 0 && !cur ? "open" : ""}><summary data-t="${esc(n.title.toLowerCase())}">${esc(n.title)}<span class="cnt">${n.count}</span></summary>
      <ul><li><a href="${href}" data-link class="${n.url === cur ? "cur" : ""}">О разделе</a></li>${treeHtml(kids, cur, depth + 1).slice(4, -5)}</ul></details></li>`;
  }).join("")}</ul>`;
}
function containsUrl(n, u) { return n.url === u || (n.children || []).some((c) => containsUrl(c, u)); }

async function viewBrowse(u) {
  document.title = "Документация — База знаний Webim";
  const url = u.searchParams.get("url");
  const tree = await loadTree();
  const root = tree.length === 1 && tree[0].children.length ? tree[0].children : tree;
  app.innerHTML = `<button class="btn small secondary mobile-tree-btn" type="button" id="tt" aria-expanded="false">Оглавление</button>
    <div class="browse"><nav class="tree-pane" id="tp" aria-label="Оглавление Базы знаний"><input class="tree-search" id="ts" type="search" placeholder="Найти раздел…" aria-label="Фильтр оглавления"><div class="tree" id="tr">${treeHtml(root, url)}</div></nav>
    <section class="reader-wrap" id="rd"></section></div>`;
  $("#tt").onclick = () => { const o = $("#tp").classList.toggle("open"); $("#tt").setAttribute("aria-expanded", o); };
  $("#ts").addEventListener("input", (e) => {
    const v = e.target.value.trim().toLowerCase();
    document.querySelectorAll("#tr li").forEach((li) => {
      const t = li.querySelector(":scope > a, :scope > details > summary");
      const own = !v || (t && (t.dataset.t || t.textContent.toLowerCase()).includes(v)) || li.textContent.toLowerCase().includes(v);
      li.style.display = own ? "" : "none";
      const det = li.querySelector(":scope > details"); if (det && v) det.open = true;
    });
  });
  const cur = $("#tr a.cur"); if (cur) { const pane = $("#tp"); pane.scrollTop = Math.max(0, cur.offsetTop - pane.clientHeight / 2); }
  const rd = $("#rd");
  if (!url) {
    const sg = await suggestions();
    rd.innerHTML = `<div class="reader"><h1>Документация Webim</h1><p style="color:var(--muted)">Выберите раздел в оглавлении слева или задайте вопрос — так обычно быстрее.</p>
      <p><a class="btn small" href="/" data-link>Спросить в Базе знаний</a></p></div>`;
    return;
  }
  rd.innerHTML = `<div class="reader"><div class="skel"><i></i><i></i><i></i></div></div>`;
  try {
    const a = await api(`/api/article?url=${encodeURIComponent(url)}`);
    document.title = `${a.title} — База знаний Webim`;
    rd.innerHTML = `<article class="reader"><div class="crumbs">${crumbsHtml(a.breadcrumbs.map((b) => b.title))}</div><h1>${esc(a.title)}</h1>
      <div class="top"><a class="btn small" href="${esc(a.url)}" target="_blank" rel="noopener">Открыть оригинал ${ICON_EXT}</a>${a.updated ? `<span class="crumbs">Обновлено ${esc(a.updated.split("-").reverse().join("."))}</span>` : ""}</div>
      ${a.sections.map(sectionHtml).join("")}
      <div class="reader-foot">Текст показан из индекса прототипа без изображений. Актуальная версия — на <a href="${esc(a.url)}" target="_blank" rel="noopener">webim.ru</a>.</div></article>`;
    const hash = decodeURIComponent(location.hash.slice(1));
    if (hash) { const el = document.getElementById(hash); if (el) { el.scrollIntoView(); el.classList.add("target"); } }
  } catch {
    rd.innerHTML = `<div class="empty"><h2>Статья не найдена в индексе</h2><p>Возможно, она была удалена или ещё не проиндексирована.</p><a class="btn small" href="/browse" data-link>К оглавлению</a></div>`;
  }
}
function sectionHtml(s, i) {
  const lvl = Math.min(Math.max(s.level, 2), 4);
  const head = i === 0 ? "" : `<h${lvl} id="${esc(s.anchor)}">${esc(s.heading)}</h${lvl}>`;
  return head + s.blocks.map(blockHtml).join("");
}
function blockHtml(b) {
  switch (b.type) {
    case "p": return `<p>${esc(b.text)}</p>`;
    case "list": { const t = b.ordered ? "ol" : "ul"; return `<${t}>${b.items.map((i) => `<li style="margin-left:${(i.depth || 0) * 18}px">${esc(i.text)}</li>`).join("")}</${t}>`; }
    case "code": return `<pre><code>${esc(b.text)}</code></pre>`;
    case "table": return `<table>${b.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table>`;
    case "note": return `<div class="note">${b.title ? `<strong>${esc(b.title)}.</strong> ` : ""}${esc(b.text)}</div>`;
    case "figure": return `<p class="figure">Иллюстрация: ${esc(b.text)}</p>`;
    case "quote": return `<blockquote>${esc(b.text)}</blockquote>`;
    default: return "";
  }
}

/* ---------- демо ---------- */
async function viewDemo() {
  document.title = "Примеры вопросов — База знаний Webim";
  const sg = await suggestions();
  const demo = sg.demo || [];
  app.innerHTML = `<div class="wrap"><div class="page-title"><h1>Примеры вопросов</h1>
    <p>Подобранные и проверенные вручную сценарии. Каждый ответ собирается из текущей Базы знаний; в результатах можно открыть точный фрагмент и перейти на оригинальный раздел на webim.ru.</p></div>
    <div class="scen">${demo.map((d, i) => `<div class="card sc"><div class="no">${i + 1}</div><div>
      <div class="qq">${esc(d.q)}${d.tech ? "<small>для разработчиков</small>" : ""}</div>
      ${d.why ? `<p class="why2">${esc(d.why)}</p>` : ""}
      ${d.expect ? `<div class="exp">Ожидаемый источник: <a href="${esc(d.expect.url + (d.expect.anchor ? "#" + d.expect.anchor : ""))}" target="_blank" rel="noopener">${esc(d.expect.title)}${d.expect.section ? " › " + esc(d.expect.section) : ""} ${ICON_EXT}</a></div>` : ""}</div>
      <a class="btn small" href="/?q=${encodeURIComponent(d.q)}" data-link>Показать</a></div>`).join("")}</div></div>`;
}

/* ---------- статус ---------- */
async function viewAdmin() {
  document.title = "Статус индекса — База знаний Webim";
  app.innerHTML = `<div class="wrap"><div class="page-title"><h1>Статус индекса</h1><p>Служебная страница: состояние поискового индекса и подключённых моделей.</p></div><div id="ad" class="skel" style="padding:24px 0"><i></i><i></i><i></i></div></div>`;
  let s;
  try { s = await api("/api/status"); } catch { $("#ad").outerHTML = `<div class="card errbox"><h2>Статус недоступен</h2></div>`; return; }
  const ls = s.last_sync, fmt = (t) => t ? new Date(t * 1000).toLocaleString("ru-RU") : "—";
  const llm = s.llm.configured ? esc(s.llm.label) : "не подключён — режим «поиск + выдержки из документации»";
  $("#ad").outerHTML = `<div class="stat-grid">
      <div class="card stat"><div class="v">${s.counts.articles}</div><div class="l">статей в индексе</div></div>
      <div class="card stat"><div class="v">${s.counts.chunks}</div><div class="l">разделов (чанков)</div></div>
      <div class="card stat"><div class="v">${s.counts.embedded_chunks}</div><div class="l">уникальных эмбеддингов</div></div>
      <div class="card stat"><div class="v">${s.recent_latency_ms.avg ?? "—"}${s.recent_latency_ms.avg ? " мс" : ""}</div><div class="l">средняя задержка поиска (${s.recent_latency_ms.n} последних)</div></div></div>
    ${(s.warnings || []).map((w) => `<div class="warnbox">⚠ ${esc(w)}</div>`).join("")}
    <div class="card" style="padding:6px 10px;margin-bottom:22px"><table class="kv">
      <tr><th>Состояние</th><td><span class="dot ${s.healthy && s.semantic_ready ? "" : "warn"}"></span>${s.healthy ? (s.semantic_ready ? "индекс готов, семантический поиск активен" : "индекс готов, семантический поиск недоступен (только лексический)") : "индекс пуст — выполните синхронизацию"}</td></tr>
      <tr><th>Модель эмбеддингов</th><td>${esc(s.embedding_model || "не загружена")}</td></tr>
      <tr><th>Реранкер</th><td>${esc(s.reranker || "не загружен")}</td></tr>
      <tr><th>Языковая модель (LLM)</th><td>${llm}</td></tr>
      <tr><th>Последняя синхронизация</th><td>${ls ? fmt(ls.finished) : "—"}</td></tr>
      ${ls ? `<tr><th>Итоги последней синхронизации</th><td>найдено ${ls.discovered}, без изменений ${ls.unchanged}, обновлено ${ls.updated}, добавлено ${ls.new}, удалено ${ls.removed}, ошибок ${ls.failed}; чанков обновлено ${ls.chunks_updated}; за ${ls.duration_s} с</td></tr>` : ""}
    </table></div>
    <h2 class="h-sec">История синхронизаций</h2>
    <div class="card" style="padding:6px 10px;margin-bottom:30px"><table class="kv"><tr><th style="width:auto">Когда</th><th style="width:auto">Найдено</th><th style="width:auto">Новых</th><th style="width:auto">Обновлено</th><th style="width:auto">Удалено</th><th style="width:auto">Длительность</th></tr>
      ${s.sync_history.map((r) => `<tr><td>${fmt(r.finished)}</td><td>${r.discovered}</td><td>${r.new}</td><td>${r.updated}</td><td>${r.removed}</td><td>${r.duration_s} с</td></tr>`).join("") || '<tr><td colspan="6">пока нет записей</td></tr>'}</table></div>
    ${s.recent.length ? `<h2 class="h-sec">Последние запросы <small>диагностика, мс</small></h2><div class="card" style="padding:6px 10px;margin-bottom:56px"><table class="kv"><tr><th style="width:auto">Запрос</th><th style="width:auto">Лексика</th><th style="width:auto">Семантика</th><th style="width:auto">Реранк</th><th style="width:auto">Итого</th></tr>
      ${s.recent.slice().reverse().map((r) => `<tr><td>${esc(r.q)}</td><td>${r.lexical_ms ?? "—"}</td><td>${r.semantic_ms ?? "—"}</td><td>${r.rerank_ms ?? "—"}</td><td>${r.total_ms ?? "—"}</td></tr>`).join("")}</table></div>` : ""}`;
}

route();
