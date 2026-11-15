// Plate II: a 2-D slice of the model — posterior territory, Gaussian contours and every specimen.

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const ease = t => 1 - Math.pow(1 - t, 3.4);

export function initField({ meta, data, S, CLR, RGB, api }) {
  const cv = document.getElementById('fieldCv'), ctx = cv.getContext('2d');
  const tipEl = document.getElementById('fieldTip');
  const names = meta.classes, feats = meta.features;
  const st = { fx: 2, fy: 3, layers: { regions: true, rings: true, dots: true }, grid: null, intro: 0, hover: -1, t: 0 };
  const cache = {};
  let W = 0, H = 0, dpr = 1, raf = 0, visible = false, introStarted = false;
  const M = { l: 64, r: 24, t: 24, b: 56 };

  // ---- controls
  const mk = (host, which) => {
    const h = document.getElementById(host); h.innerHTML = '';
    feats.forEach((f, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (st[which] === i ? ' on' : ''); b.textContent = f;
      b.onclick = () => {
        if (st[which] === i) return;
        const other = which === 'fx' ? 'fy' : 'fx';
        if (st[other] === i) st[other] = st[which];
        st[which] = i; mk('axisX', 'fx'); mk('axisY', 'fy'); load();
      };
      h.appendChild(b);
    });
  };
  mk('axisX', 'fx'); mk('axisY', 'fy');
  document.querySelectorAll('#layers .chip').forEach(b => b.onclick = () => { b.classList.toggle('on'); st.layers[b.dataset.layer] = b.classList.contains('on'); draw(); });
  const legend = document.getElementById('legend');
  names.forEach((n, k) => {
    const nk = data.samples.filter(s => s.label === k).length;
    const li = document.createElement('li'); li.innerHTML = `<i style="background:${CLR[k]}"></i>${n}<span>${nk} specimens</span>`; legend.appendChild(li);
  });
  document.getElementById('fieldHint').textContent = 'Bright ring = held-out test flower. Black halo = a flower the model reads wrongly.';

  // ---- data fetch
  async function load() {
    const key = st.fx + '-' + st.fy;
    if (!cache[key]) cache[key] = await api(`/field?fx=${st.fx}&fy=${st.fy}`);
    st.grid = cache[key]; buildWash(); draw();
  }

  let wash = document.createElement('canvas');
  function buildWash() {
    const g = st.grid; if (!g) return;
    wash.width = g.nx; wash.height = g.ny;
    const c = wash.getContext('2d'), img = c.createImageData(g.nx, g.ny);
    for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
      const p = g.posterior[g.ny - 1 - y][x];
      let r = 0, gg = 0, b = 0, mx = 0;
      for (let k = 0; k < 3; k++) { r += RGB[k][0] * p[k]; gg += RGB[k][1] * p[k]; b += RGB[k][2] * p[k]; mx = Math.max(mx, p[k]); }
      const i = (y * g.nx + x) * 4;
      img.data[i] = r; img.data[i + 1] = gg; img.data[i + 2] = b;
      img.data[i + 3] = 255 * (0.05 + 0.2 * Math.pow(mx, 2));
    }
    c.putImageData(img, 0, 0);
  }

  // ---- geometry
  const bounds = () => st.grid ? st.grid.bounds : [0, 1, 0, 1];
  const px = v => { const [x0, x1] = bounds(); return M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r); };
  const py = v => { const [, , y0, y1] = bounds(); return H - M.b - (v - y0) / (y1 - y0) * (H - M.t - M.b); };
  const ux = p => { const [x0, x1] = bounds(); return x0 + (p - M.l) / (W - M.l - M.r) * (x1 - x0); };
  const uy = p => { const [, , y0, y1] = bounds(); return y0 + (H - M.b - p) / (H - M.t - M.b) * (y1 - y0); };

  function ticks(lo, hi) {
    const span = hi - lo, raw = span / 7, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(+v.toFixed(6)); return out;
  }

  function ellipse(k) {
    const m = meta.model[k], i = st.fx, j = st.fy;
    const a = m.covariance[i][i], b = m.covariance[i][j], c = m.covariance[j][j];
    const half = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    return { mx: m.mean[i], my: m.mean[j], l1: Math.sqrt(half + d), l2: Math.sqrt(Math.max(1e-9, half - d)), ang: 0.5 * Math.atan2(2 * b, a - c) };
  }

  // ---- draw
  function draw() {
    if (!W || !st.grid) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const g = st.grid, [x0, x1, y0, y1] = g.bounds;
    const pw = W - M.l - M.r, ph = H - M.t - M.b;
    const I = ease(st.intro);

    // wash
    if (st.layers.regions) {
      ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, pw, ph); ctx.clip();
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.globalAlpha = I; ctx.drawImage(wash, M.l, M.t, pw, ph); ctx.restore();
    }
    // grid + ticks
    ctx.font = '10px "DM Mono", monospace'; ctx.fillStyle = 'rgba(' + window.__ink + ',.62)'; ctx.strokeStyle = 'rgba(' + window.__ink + ',.1)'; ctx.lineWidth = 1;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ticks(x0, x1).forEach(v => { const x = Math.round(px(v)) + 0.5; ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.fillText(v.toFixed(v % 1 ? 1 : 0), x, H - M.b + 8); });
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ticks(y0, y1).forEach(v => { const y = Math.round(py(v)) + 0.5; ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.fillText(v.toFixed(v % 1 ? 1 : 0), M.l - 10, y); });
    ctx.strokeStyle = 'rgba(' + window.__ink + ',.9)'; ctx.beginPath(); ctx.moveTo(M.l + 0.5, M.t); ctx.lineTo(M.l + 0.5, H - M.b + 0.5); ctx.lineTo(W - M.r, H - M.b + 0.5); ctx.stroke();
    ctx.fillStyle = 'rgba(' + window.__ink + ',.85)'; ctx.font = '500 10px "DM Mono", monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
    ctx.fillText((feats[st.fx] + ' · cm').toUpperCase(), W - M.r, H - 12);
    ctx.save(); ctx.translate(16, M.t + 4); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.fillText((feats[st.fy] + ' · cm').toUpperCase(), 0, 0); ctx.restore();

    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, pw, ph); ctx.clip();

    // Gaussian contours, drawn progressively
    if (st.layers.rings) {
      for (let k = 0; k < 3; k++) {
        const e = ellipse(k);
        [1, 2, 3].forEach((s, si) => {
          const prog = clamp(I * 1.4 - si * 0.18, 0, 1);
          if (prog <= 0) return;
          ctx.beginPath();
          const n = 120, end = Math.PI * 2 * prog;
          for (let q = 0; q <= n; q++) {
            const t = end * q / n, ex = e.l1 * s * Math.cos(t), ey = e.l2 * s * Math.sin(t);
            const dx = e.mx + ex * Math.cos(e.ang) - ey * Math.sin(e.ang), dy = e.my + ex * Math.sin(e.ang) + ey * Math.cos(e.ang);
            q ? ctx.lineTo(px(dx), py(dy)) : ctx.moveTo(px(dx), py(dy));
          }
          ctx.strokeStyle = `rgba(${RGB[k]},${[1, 0.65, 0.4][si]})`; ctx.lineWidth = si === 0 ? 2 : 1.2;
          ctx.setLineDash(si === 2 ? [2, 5] : []); ctx.stroke(); ctx.setLineDash([]);
        });
        // mean marker
        ctx.fillStyle = `rgb(${RGB[k]})`; ctx.globalAlpha = I;
        const cx = px(e.mx), cy = py(e.my);
        ctx.fillRect(cx - 5, cy - 1, 10, 2); ctx.fillRect(cx - 1, cy - 5, 2, 10); ctx.globalAlpha = 1;
      }
    }
    // specimens
    if (st.layers.dots) {
      data.samples.forEach((s, n) => {
        const a = clamp(I * 2.2 - (n / data.samples.length) * 1.1, 0, 1); if (a <= 0) return;
        const x = px(s.x[st.fx]), y = py(s.x[st.fy]), hv = st.hover === n;
        ctx.globalAlpha = a;
        if (s.label !== s.pred) { ctx.strokeStyle = 'rgb(' + window.__ink + ')'; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.3); ctx.stroke(); }
        ctx.fillStyle = `rgb(${RGB[s.label]})`;
        ctx.beginPath(); ctx.arc(x, y, (s.test ? 4.2 : 3) * (hv ? 1.6 : 1), 0, 6.3); ctx.fill();
        if (s.test || hv) { ctx.strokeStyle = 'rgb(' + window.__ink + ')'; ctx.lineWidth = 1.3; ctx.stroke(); }
        ctx.globalAlpha = 1;
      });
    }
    // query
    {
      const qx = px(S.x[st.fx]), qy = py(S.x[st.fy]);
      ctx.setLineDash([3, 4]); ctx.strokeStyle = 'rgba(' + window.__ink + ',.75)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(qx, M.t); ctx.lineTo(qx, H - M.b); ctx.moveTo(M.l, qy); ctx.lineTo(W - M.r, qy); ctx.stroke(); ctx.setLineDash([]);
      const pulse = (Math.sin(st.t * 3) + 1) / 2;
      ctx.strokeStyle = 'rgba(' + window.__ink + ',' + (0.6 - pulse * 0.4) + ')'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(qx, qy, 10 + pulse * 9, 0, 6.3); ctx.stroke();
      ctx.fillStyle = 'rgb(' + window.__ink + ')'; ctx.beginPath(); ctx.arc(qx, qy, 5, 0, 6.3); ctx.fill();
      ctx.strokeStyle = 'rgb(' + window.__bgc + ')'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(qx, qy, 5, 0, 6.3); ctx.stroke();
      ctx.restore();
      ctx.font = '500 10px "DM Mono", monospace'; ctx.fillStyle = 'rgb(' + window.__ink + ')'; ctx.textBaseline = 'alphabetic';
      ctx.textAlign = 'center'; ctx.fillText(S.x[st.fx].toFixed(1), clamp(qx, M.l + 14, W - M.r - 14), M.t - 8 < 10 ? M.t + 14 : M.t - 8);
      ctx.textAlign = 'left'; ctx.fillText('x', qx + 12, qy - 10);
    }
  }

  // ---- loop (pulse of the query marker only while visible)
  function loop(ts) {
    if (!visible) { raf = 0; return; }
    st.t = ts / 1000;
    if (st.intro < 1 && introStarted) st.intro = clamp(st.intro + 0.0125, 0, 1);
    draw();
    raf = requestAnimationFrame(loop);
  }
  const kick = () => { if (!raf && visible) raf = requestAnimationFrame(loop); };
  new IntersectionObserver(es => {
    visible = es[0].isIntersecting; if (visible) { if (!introStarted && st.grid) introStarted = true; kick(); }
  }, { threshold: 0.2 }).observe(cv);

  function size() {
    dpr = Math.min(devicePixelRatio || 1, 2); W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; M.l = W < 520 ? 48 : 64; draw();
  }
  new ResizeObserver(size).observe(cv); size();

  // ---- interaction
  const nearest = (mx, my) => {
    let best = -1, bd = 14 * 14;
    data.samples.forEach((s, n) => { const dx = px(s.x[st.fx]) - mx, dy = py(s.x[st.fy]) - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = n; } });
    return best;
  };
  cv.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const n = nearest(mx, my); st.hover = n;
    if (n >= 0) {
      const s = data.samples[n], bad = s.label !== s.pred;
      tipEl.innerHTML = `<b>${names[s.label]}</b>${bad ? ` · read as <b>${names[s.pred]}</b>` : ''}<br>${s.x.map(v => v.toFixed(1)).join(' · ')} · ${s.test ? 'test' : 'train'}`;
      tipEl.style.left = px(s.x[st.fx]) + 'px'; tipEl.style.top = py(s.x[st.fy]) + 'px'; tipEl.classList.add('on');
    } else tipEl.classList.remove('on');
  });
  cv.addEventListener('pointerleave', () => { st.hover = -1; tipEl.classList.remove('on'); });
  cv.addEventListener('click', e => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (mx < M.l || mx > W - M.r || my < M.t || my > H - M.b) return;
    const n = nearest(mx, my), x = S.x.slice();
    if (n >= 0) { const s = data.samples[n]; x[st.fx] = s.x[st.fx]; x[st.fy] = s.x[st.fy]; }
    else { x[st.fx] = +ux(mx).toFixed(1); x[st.fy] = +uy(my).toFixed(1); }
    S.set(x, 'field');
  });
  S.subs.push(() => { if (!raf) draw(); });
  addEventListener('themechange', () => { buildWash(); draw(); });

  load().then(() => { if (visible) { introStarted = true; kick(); } });
}
