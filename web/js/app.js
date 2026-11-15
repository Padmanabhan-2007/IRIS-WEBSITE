import { Stage } from './stage.js';
import { initField } from './field.js';

// Species inks (for type, bars, dots) and the room each verdict paints behind the flower.
// Rooms are roughly complementary to the bloom: apricot/blue, pistachio/violet, saffron/indigo.
const CLR = ['var(--c0)', 'var(--c1)', 'var(--c2)'];
const RGB = [[59, 120, 196], [138, 63, 174], [56, 55, 154]];
const ROOM = [[243, 202, 160], [222, 227, 150], [244, 187, 78]];

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const ease = t => 1 - Math.pow(1 - t, 3.2);
const api = (p, body) => fetch('/api' + p, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

function tween(from, to, ms, fn, done) {
  if (reduced) { fn(to); done && done(); return; }
  const t0 = performance.now();
  const step = now => { const p = clamp((now - t0) / ms, 0, 1); fn(from + (to - from) * ease(p)); p < 1 ? requestAnimationFrame(step) : done && done(); };
  requestAnimationFrame(step);
}

// Light: clean paper. Dark: ink. The verdict only breathes a little of its colour into either.
const THEMES = {
  light: { base: [249, 248, 246], f: 0.09, ink: '29,27,34', rgb: [[59, 120, 196], [138, 63, 174], [56, 55, 154]] },
  dark: { base: [18, 17, 22], f: 0.1, ink: '236,233,225', rgb: [[111, 166, 238], [194, 127, 230], [143, 148, 255]] },
};
let theme = 'light', lastW = [0, 1, 0];
function tint(w) {
  lastW = w;
  const T = THEMES[theme], s = (w[0] + w[1] + w[2]) || 1;
  const c = [0, 1, 2].map(ch => Math.round(T.base[ch] * (1 - T.f) + T.f * (w[0] * ROOM[0][ch] + w[1] * ROOM[1][ch] + w[2] * ROOM[2][ch]) / s));
  document.documentElement.style.setProperty('--bg', `rgb(${c})`);
}
function setTheme(t, save) {
  theme = t; const T = THEMES[t];
  document.documentElement.dataset.theme = t;
  RGB.forEach((c, k) => c.splice(0, 3, ...T.rgb[k]));
  window.__ink = T.ink; window.__bgc = T.base.join(',');
  const btn = document.getElementById('themeBtn'); if (btn) $('span', btn).textContent = t === 'light' ? 'Dark' : 'Light';
  if (save) { try { localStorage.setItem('iris-theme', t); } catch (e) {} }
  tint(lastW);
  dispatchEvent(new Event('themechange'));
}
(function () { let t = 'light'; try { t = localStorage.getItem('iris-theme') || 'light'; } catch (e) {} setTheme(t); })();
document.getElementById('themeBtn').addEventListener('click', () => setTheme(theme === 'light' ? 'dark' : 'light', true));

/* ===================================================== scroll chrome + reveal */
(function chrome() {
  const fill = $('#railFill');
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    fill.style.transform = `scaleY(${max > 0 ? scrollY / max : 0})`;
    $('.nav').classList.toggle('scrolled', scrollY > 40);
  };
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
  document.querySelectorAll('.reveal, .step, .trial').forEach(n => io.observe(n));
  const links = [...document.querySelectorAll('.nav nav a')];
  const so = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('.plate').forEach(s => so.observe(s));
})();

/* ================================================================== boot */
const S = { x: [5.8, 3.0, 4.35, 1.3], subs: [], set(x, from) { this.x = x.slice(); this.subs.forEach(f => f(from)); } };
let stage;
try { stage = new Stage($('#stage')); } catch (e) { console.error(e); document.body.classList.add('nogl'); }

async function boot() {
  const status = $('#status'), st = $('#statusText');
  let meta, ev, data;
  try { [meta, ev, data] = await Promise.all([api('/meta'), api('/evaluation'), api('/dataset')]); }
  catch (e) { status.classList.add('err'); st.textContent = 'api offline — start the server'; return; }
  status.classList.add('live');
  st.textContent = `model live · ${meta.n_train} train / ${meta.n_test} test`;
  $('#hN').textContent = meta.n_total;
  $('#hFit').textContent = meta.fit_ms.toFixed(1);

  const labApi = lab(meta, ev);
  hero(meta, labApi);
  initField({ meta, data, S, CLR, RGB, api });
  machinery(meta);
  trial(meta, ev);
}

/* ==================================================================== hero */
function hero(meta, labApi) {
  const means = meta.model.map(m => m.mean), names = meta.classes;
  const pickHost = $('#speciesPick');
  let hk = 1, hovering = false;
  const btns = names.map((n, k) => {
    const b = el('button', '', `<i></i>${n}<span>petal ${means[k][2].toFixed(1)} cm</span>`);
    b.style.setProperty('--cc', CLR[k]);
    b.addEventListener('pointerenter', () => { hovering = true; hk = k; apply(); });
    b.addEventListener('pointerleave', () => { hovering = false; });
    b.addEventListener('focus', () => { hk = k; apply(); });
    pickHost.appendChild(b); return b;
  });
  const apply = (snap) => {
    btns.forEach((b, i) => b.classList.toggle('on', i === hk));
    if (labApi.active() !== 'hero' && !snap) return;
    const m = means[hk], onehot = [0, 1, 2].map(i => +(i === hk));
    stage.setTraits({ sl: m[0], sw: m[1], pl: m[2], pw: m[3] }, snap);
    stage.setMix(onehot, snap); tint(onehot); stage.pulse();
  };
  labApi.onHero(apply);
  apply(true);
  setTimeout(() => { $('#hero').classList.add('go'); stage.bloom(1, reduced ? 0.01 : 3.2, 0.1); }, 250);
  setInterval(() => { if (document.hidden || hovering || labApi.active() !== 'hero') return; hk = (hk + 1) % 3; apply(); }, 5200);
  addEventListener('pointermove', e => stage.setPointer(e.clientX / innerWidth * 2 - 1, e.clientY / innerHeight * 2 - 1), { passive: true });
}

/* ===================================================================== lab */
function lab(meta, ev) {
  const names = meta.classes;
  let active = '', lastPost = null, heroApply = () => {};
  stage.cur.bloom = 0;

  // sliders
  const wrap = $('#sliders'), inputs = [], vals = [];
  const blurb = ['shapes the falls', 'widens the falls', 'shapes the standards', 'widens the standards'];
  const ranges = meta.ranges.map(r => { const p = 0.15 * (r.max - r.min); return [Math.max(0.1, +(r.min - p).toFixed(1)), +(r.max + p).toFixed(1)]; });
  meta.features.forEach((nm, i) => {
    const [lo, hi] = ranges[i];
    const row = el('div', 'sl');
    row.innerHTML = `<div class="sl-top"><div class="sl-name">${nm[0].toUpperCase() + nm.slice(1)}<small>${blurb[i]}</small></div><div class="sl-val"><span>${S.x[i].toFixed(1)}</span><i>cm</i></div></div>
      <div class="rng"><input type="range" min="${lo}" max="${hi}" step="0.1" value="${S.x[i]}" aria-label="${nm}"><div class="bands"></div></div>`;
    const bands = $('.bands', row);
    meta.model.forEach(m => {
      const sd = Math.sqrt(m.covariance[i][i]);
      const b = el('b'); b.style.left = ((m.mean[i] - sd - lo) / (hi - lo) * 100) + '%'; b.style.width = (2 * sd / (hi - lo) * 100) + '%';
      b.title = `${m.name}: ${m.mean[i].toFixed(2)} ± ${sd.toFixed(2)}`; bands.appendChild(b);
    });
    const inp = $('input', row);
    inp.addEventListener('input', () => { const x = S.x.slice(); x[i] = +inp.value; S.set(x, 'slider'); });
    inputs.push(inp); vals.push($('.sl-val span', row)); wrap.appendChild(row);
  });

  let glideId = 0;
  const glide = (target, ms = 900) => {
    const id = ++glideId, from = S.x.slice();
    tween(0, 1, ms, p => { if (id === glideId) S.set(from.map((v, i) => v + (target[i] - v) * p), 'glide'); });
    stage.pulse();
  };

  const pr = $('#presets');
  const add = (label, fn) => { const b = el('button', 'chip', label); b.addEventListener('click', fn); pr.appendChild(b); };
  meta.model.forEach(m => add('Typical ' + m.name, () => glide(m.mean)));
  const hardest = ev.test.slice().sort((a, b) => a.confidence - b.confidence)[0];
  const wrong = ev.test.find(t => t.label !== t.pred);
  add('Hardest case', () => glide(hardest.x));
  if (wrong) add('The misread one', () => glide(wrong.x));
  add('Surprise me', () => glide(ev.test[Math.floor(Math.random() * ev.test.length)].x));
  window.__loadSpecimen = x => { glide(x); document.getElementById('lab').scrollIntoView({ behavior: 'smooth' }); };

  // drag to turn the bloom
  const surf = $('#labStage');
  let d0 = null;
  surf.addEventListener('pointerdown', e => { surf.setPointerCapture(e.pointerId); d0 = { x: e.clientX, y: e.clientY, yaw: stage.cur.yaw, pitch: stage.cur.pitch }; });
  surf.addEventListener('pointermove', e => { if (!d0) return; stage.drag = { yaw: d0.yaw + (e.clientX - d0.x) * 0.012, pitch: clamp(d0.pitch + (e.clientY - d0.y) * 0.006, -0.5, 0.6) }; });
  const end = () => { d0 = null; stage.drag = null; };
  surf.addEventListener('pointerup', end); surf.addEventListener('pointercancel', end);

  // readout
  const posts = $('#posts');
  const rows = names.map((n, k) => { const r = el('div', 'post', `<span class="nm">${n}</span><span class="bar"><i style="background:${CLR[k]}"></i></span><span class="pc">0.0%</span>`); posts.appendChild(r); return r; });
  const hood = $('#hood tbody');
  const hrows = names.map(n => { const r = el('tr', '', `<td>${n}</td><td></td><td></td><td></td><td></td>`); hood.appendChild(r); return r; });
  const vName = $('#verdict'), conf = $('#confidence');
  let curK = -1, shownPct = [0, 0, 0];

  const labApply = (snap) => {
    stage.setTraits({ sl: S.x[0], sw: S.x[1], pl: S.x[2], pw: S.x[3] }, snap);
    if (lastPost) { stage.setMix(lastPost, snap); tint(lastPost); }
  };

  const render = res => {
    const post = res.classes.map(c => c.posterior);
    lastPost = post;
    if (active !== 'hero') { stage.setMix(post); tint(post); }
    const k = res.prediction_index;
    if (k !== curK) {
      vName.classList.add('swap'); void vName.offsetWidth;
      vName.textContent = res.prediction; vName.style.color = CLR[k]; vName.classList.remove('swap'); curK = k;
    }
    const top = post[k];
    conf.innerHTML = top > 0.9995 ? `<b>≥ 99.9%</b> posterior · decisive` : `<b>${(top * 100).toFixed(1)}%</b> posterior · margin ${res.margin.toFixed(3)} over the runner-up`;
    res.classes.forEach((c, i) => {
      rows[i].classList.toggle('win', i === k);
      $('.bar i', rows[i]).style.transform = `scaleX(${clamp(c.posterior, 0, 1)})`;
      const pc = $('.pc', rows[i]), from = shownPct[i], to = c.posterior * 100;
      tween(from, to, 450, v => { pc.textContent = v.toFixed(1) + '%'; }); shownPct[i] = to;
      const tds = hrows[i].children; hrows[i].classList.toggle('win', i === k);
      tds[1].textContent = c.prior.toFixed(3); tds[2].textContent = c.log_likelihood.toFixed(2);
      tds[3].textContent = c.mahalanobis_sq.toFixed(2); tds[4].textContent = c.discriminant.toFixed(2);
    });
  };

  let inflight = false, dirty = false;
  const predict = async () => {
    if (inflight) { dirty = true; return; }
    inflight = true;
    try {
      do { dirty = false; const [a, b, c, d] = S.x; render(await api('/predict', { sepal_length: a, sepal_width: b, petal_length: c, petal_width: d })); } while (dirty);
    } catch (e) { $('#status').classList.replace('live', 'err'); }
    inflight = false;
  };

  S.subs.push(from => {
    S.x.forEach((v, i) => { vals[i].textContent = v.toFixed(1); if (from !== 'slider') inputs[i].value = v; });
    if (active !== 'hero') stage.setTraits({ sl: S.x[0], sw: S.x[1], pl: S.x[2], pw: S.x[3] });
    $('#capRight').textContent = 'x = [' + S.x.map(v => v.toFixed(1)).join(', ') + ']';
    predict();
  });
  $('#capRight').textContent = 'x = [' + S.x.map(v => v.toFixed(1)).join(', ') + ']';
  predict();

  // the single flower serves every anchor; swap what it shows by where it is
  stage.onActive = name => { active = name; name === 'hero' ? heroApply() : labApply(); };

  // anatomical callouts, anchored to projected vertices of the 3D mesh
  const co = $('#callouts');
  const place = (id, pt, dx, dy, side, l1, l2) => {
    const g = $(id), s = side === 'l' ? -1 : 1;
    $('path', g).setAttribute('d', `M${pt[0]},${pt[1]}L${pt[0] + dx},${pt[1] + dy}L${pt[0] + dx + s * 26},${pt[1] + dy}`);
    const c = $('circle', g); c.setAttribute('cx', pt[0]); c.setAttribute('cy', pt[1]);
    for (const [cls, txt, oy] of [['.t1', l1, -7], ['.t2', l2, 9]]) {
      const t = $(cls, g); t.textContent = txt; t.setAttribute('x', pt[0] + dx + s * 34); t.setAttribute('y', pt[1] + dy + oy); t.setAttribute('text-anchor', s < 0 ? 'end' : 'start');
    }
  };
  stage.onFrame = f => {
    const on = f.name === 'lab' && f.settled && f.bloom > 0.85;
    co.classList.toggle('on', on);
    if (!on) return;
    const c = stage.cur;
    place('#coFall', f.fallEdge, -60, 50, 'l', 'FALL · SEPAL', `${c.sl.toFixed(1)} × ${c.sw.toFixed(1)} cm`);
    place('#coStd', f.stdEdge, 70, -46, 'r', 'STANDARD · PETAL', `${c.pl.toFixed(1)} × ${c.pw.toFixed(1)} cm`);
  };

  return {
    active: () => active,
    onHero: fn => { heroApply = fn; },
  };
}

/* ============================================================== machinery */
function machinery(meta) {
  const tex = () => document.querySelectorAll('.eq').forEach(n => window.katex && katex.render(n.dataset.tex, n, { displayMode: true, throwOnError: false }));
  window.katex ? tex() : addEventListener('load', tex);
  const pri = $('#priors');
  meta.model.forEach((m, k) => {
    const nk = Math.round(m.prior * meta.n_train);
    pri.appendChild(el('div', 'prior-row', `<span class="nm">${m.name}</span><span class="bar"><i style="background:${CLR[k]};--w:${clamp(m.prior * 2.6, 0, 1)}"></i></span><span>N<sub>k</sub> = ${nk} / ${meta.n_train} → ${m.prior.toFixed(4)}</span>`));
  });
  const means = $('#means');
  meta.model.forEach((m, k) => {
    const card = el('div', 'mean-card'); card.style.setProperty('--cc', CLR[k]);
    card.innerHTML = `<h4>${m.name}</h4>` + meta.features.map((f, i) => {
      const r = meta.ranges[i], pos = (m.mean[i] - r.min) / (r.max - r.min) * 100;
      return `<div class="mean-row"><span>${f}</span><span class="t"><i style="left:${pos}%"></i></span><b>${m.mean[i].toFixed(2)}</b></div>`;
    }).join('');
    means.appendChild(card);
  });
  const covs = $('#covs'), short = ['SL', 'SW', 'PL', 'PW'];
  meta.model.forEach((m, k) => {
    const card = el('div', 'cov-card'); card.style.setProperty('--cc', CLR[k]);
    let cells = '';
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const c = m.covariance[i][j], r = c / Math.sqrt(m.covariance[i][i] * m.covariance[j][j]);
      const col = r >= 0 ? RGB[k] : [204, 76, 54], a = 0.1 + 0.8 * Math.abs(r);
      cells += `<div style="--i:${i * 4 + j};background:rgba(${col},${a.toFixed(2)});color:${a > 0.55 ? '#fff' : 'var(--ink)'}" title="${short[i]}×${short[j]} · corr ${r.toFixed(2)}">${c.toFixed(2)}</div>`;
    }
    card.innerHTML = `<h4>${m.name}</h4><div class="cov-grid">${cells}</div>`;
    covs.appendChild(card);
  });
}

/* ==================================================================== trial */
function trial(meta, ev) {
  const names = meta.classes, accNum = $('#accNum');
  let counted = false;
  new IntersectionObserver(es => { if (es[0].isIntersecting && !counted) { counted = true; tween(0, ev.accuracy * 100, 1800, v => { accNum.textContent = v.toFixed(1); }); } }, { threshold: 0.3 }).observe(accNum);
  const ok = ev.test.filter(t => t.label === t.pred).length;
  $('#accSub').innerHTML = `${ok} of ${ev.test.length} held-out flowers classified correctly.<br>Training accuracy ${(ev.train_accuracy * 100).toFixed(1)}% — the gap is the honest cost of generalising.`;

  const cm = $('#cm');
  cm.appendChild(el('div', 'h', ''));
  names.forEach(n => cm.appendChild(el('div', 'h', n.slice(0, 5) + '.')));
  ev.confusion_matrix.forEach((row, r) => {
    cm.appendChild(el('div', 'rl', names[r]));
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    row.forEach((v, c) => {
      const cell = el('div', 'cell', v); cell.style.setProperty('--i', r * 3 + c);
      if (r === c) { cell.style.background = `rgba(${RGB[r]},${(0.2 + 0.75 * v / tot).toFixed(2)})`; cell.style.color = v / tot > 0.45 ? '#fff' : 'var(--ink)'; }
      else if (v > 0) { cell.style.background = 'var(--ink)'; cell.style.color = 'var(--bg)'; }
      else { cell.style.background = 'var(--soft)'; cell.style.color = 'var(--ink-3)'; }
      cm.appendChild(cell);
    });
  });

  const prf = $('#prf');
  names.forEach((n, k) => {
    const r = ev.report[n];
    const m = (lab, v) => `<div class="prf-m">${lab}<b>${v.toFixed(2)}</b><div class="t"><i style="--w:${v}"></i></div></div>`;
    const row = el('div', 'prf-row', `<span class="nm" style="color:${CLR[k]}">${n}</span>${m('precision', r.precision)}${m('recall', r.recall)}${m('f1', r['f1-score'])}`);
    row.style.setProperty('--cc', CLR[k]); prf.appendChild(row);
  });

  const grid = $('#grid45'), tip = $('#gridTip');
  const order = ev.test.map((t, i) => ({ ...t, i })).sort((a, b) => a.label - b.label || a.i - b.i);
  const glyph = (x, k) => {
    const [sl, sw, pl, pw] = x, c = CLR[k]; let s = '';
    for (let a = 0; a < 3; a++) s += `<ellipse cx="0" cy="${sl * 3.3}" rx="${sw * 3.1}" ry="${sl * 3.6}" transform="rotate(${a * 120 + 90} 0 0)" fill="${c}" fill-opacity=".8" stroke="currentColor" stroke-opacity=".45" stroke-width="1"/>`;
    for (let a = 0; a < 3; a++) s += `<ellipse cx="0" cy="${-pl * 3}" rx="${pw * 6.5}" ry="${pl * 3.3}" transform="rotate(${a * 120 + 60} 0 0)" fill="${c}" stroke="currentColor" stroke-opacity=".55" stroke-width="1"/>`;
    return `<svg viewBox="-50 -50 100 100">${s}<circle r="3.4" fill="currentColor"/></svg>`;
  };
  order.forEach((t, n) => {
    const bad = t.label !== t.pred;
    const d = el('button', 'sp' + (bad ? ' bad' : ''), glyph(t.x, t.label));
    d.style.setProperty('--i', n);
    d.setAttribute('aria-label', `${names[t.label]} specimen, predicted ${names[t.pred]}`);
    d.addEventListener('pointerenter', () => {
      const r = d.getBoundingClientRect(), p = grid.parentElement.getBoundingClientRect();
      tip.innerHTML = `<b>${names[t.label]}</b>${bad ? ` → read as <b>${names[t.pred]}</b>` : ' ✓'}<br>confidence ${(t.confidence * 100).toFixed(1)}% · [${t.x.map(v => v.toFixed(1)).join(', ')}]`;
      tip.style.left = (r.left - p.left + r.width / 2) + 'px'; tip.style.top = (r.top - p.top) + 'px'; tip.classList.add('on');
    });
    d.addEventListener('pointerleave', () => tip.classList.remove('on'));
    d.addEventListener('click', () => window.__loadSpecimen(t.x));
    grid.appendChild(d);
  });
  grid.parentElement.style.position = 'relative';
}

boot();
