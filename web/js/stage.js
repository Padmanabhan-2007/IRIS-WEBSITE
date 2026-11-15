// One persistent WebGL iris that glides between page anchors.
// Petals are smooth parametric surfaces (spine integrated from a tangent angle, cupped + ruffled
// cross-section) shaded with a physical material, soft shadows and an image-based environment.
// Anatomy follows the four measurements; colour follows the posterior P(class | x).
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const D2R = Math.PI / 180;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const lighten = (c, t) => mix(c, [255, 252, 244], t);
const darken = (c, t) => mix(c, [12, 8, 30], t);
const lin = c => Math.pow(c / 255, 2.2);

export const PALETTE = [
  { body: [96, 156, 238], rim: [60, 108, 214], claw: [228, 240, 255], vein: [40, 72, 172], gold: [255, 238, 176] }, // setosa
  { body: [156, 74, 208], rim: [108, 40, 164], claw: [250, 238, 208], vein: [72, 26, 124], gold: [246, 192, 62] },  // versicolor
  { body: [56, 50, 166], rim: [28, 24, 114], claw: [246, 220, 152], vein: [20, 16, 84], gold: [246, 178, 42] },    // virginica
];

function profile(u, o) {
  const base = o.claw + (1 - o.claw) * smooth(0, o.peak, u);
  const t = clamp((u - o.tip) / (1 - o.tip), 0, 1);
  return base * Math.sqrt(Math.max(0, 1 - Math.pow(t, o.tipPow || 2.3)));
}

function fillMesh(o, pts) {
  const { nu, nv } = o;
  const ct = Math.cos(o.theta), st = Math.sin(o.theta), du = 1 / nu;
  let r = o.r0 || 0, h = 0;
  for (let i = 0; i <= nu; i++) {
    const u = i * du;
    const a = lerp(o.a0, o.a1, Math.pow(u, o.bend));
    const w = o.W * profile(u, o);
    const nr = -Math.sin(a), nz = Math.cos(a);
    for (let j = 0; j <= nv; j++) {
      const v = -1 + (2 * j) / nv;
      const x = v * w * 0.5;
      const cup = o.cup * v * v * w * 0.5;
      const ruf = o.ruffle * smooth(0.4, 1, u) * Math.pow(Math.abs(v), 1.4) * Math.sin(u * o.rf + v * 2.4 + o.seed) * w * 0.2;
      const off = cup + ruf;
      const R = r + nr * off, Z = h + nz * off;
      const k = (i * (nv + 1) + j) * 3;
      pts[k] = o.ox + R * ct - x * st;
      pts[k + 1] = o.oy + R * st + x * ct;
      pts[k + 2] = o.oz + Z;
    }
    r += Math.cos(a) * o.L * du;
    h += Math.sin(a) * o.L * du;
  }
}

function petalColor(type, u, v, pal) {
  const av = Math.abs(v);
  if (type === 'fall') {
    let col = mix(pal.claw, pal.body, smooth(0.04, 0.32, u));
    col = mix(col, pal.rim, smooth(0.62, 1, u) * 0.5 + Math.pow(av, 3.2) * 0.42);
    const patch = (1 - smooth(0.1, 0.5, u)) * (1 - smooth(0.06, 0.62, av));
    col = mix(col, pal.gold, patch * 0.9);
    const halo = smooth(0.18, 0.4, u) * (1 - smooth(0.4, 0.7, u)) * (1 - smooth(0.2, 0.8, av)) * 0.25;
    col = mix(col, pal.claw, halo);
    return mix(col, darken(col, 0.45), (1 - smooth(0, 0.26, u)) * 0.4);
  }
  if (type === 'std') {
    let col = mix(lighten(pal.body, 0.3), pal.body, smooth(0.1, 0.8, u));
    col = mix(col, pal.claw, (1 - smooth(0, 0.28, u)) * 0.6);
    return mix(col, pal.rim, Math.pow(av, 3) * 0.3 + smooth(0.85, 1, u) * 0.2);
  }
  if (type === 'style') return mix(lighten(pal.body, 0.5), pal.claw, 0.3 + 0.3 * (1 - u));
  let col = mix([30, 78, 46], [104, 158, 78], smooth(0, 0.8, u) * 0.7 + (1 - av) * 0.3);
  return mix(col, [168, 204, 120], Math.max(0, 1 - av * 8) * 0.5);
}

class Petal {
  constructor(type, nu, nv, material) {
    this.type = type; this.nu = nu; this.nv = nv;
    const n = (nu + 1) * (nv + 1);
    this.pos = new Float32Array(n * 3);
    this.col = new Float32Array(n * 3);
    const uv = new Float32Array(n * 2), idx = [];
    for (let i = 0; i <= nu; i++) for (let j = 0; j <= nv; j++) { const k = (i * (nv + 1) + j) * 2; uv[k] = j / nv; uv[k + 1] = i / nu; }
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
      const a = i * (nv + 1) + j, b = a + nv + 1, c = b + 1, d = a + 1;
      idx.push(a, b, d, b, c, d);
    }
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    this.geo.setIndex(idx);
    this.mesh = new THREE.Mesh(this.geo, material);
    this.mesh.castShadow = true; this.mesh.receiveShadow = true; this.mesh.frustumCulled = false;
  }
  update(o, pal) {
    fillMesh({ ...o, nu: this.nu, nv: this.nv }, this.pos);
    const { nu, nv } = this;
    for (let i = 0; i <= nu; i++) for (let j = 0; j <= nv; j++) {
      const c = petalColor(this.type, i / nu, -1 + (2 * j) / nv, pal), k = (i * (nv + 1) + j) * 3;
      this.col[k] = lin(c[0]); this.col[k + 1] = lin(c[1]); this.col[k + 2] = lin(c[2]);
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
    this.geo.computeVertexNormals();
  }
  at(i, j, out) { const k = (i * (this.nv + 1) + j) * 3; return out.set(this.pos[k], this.pos[k + 1], this.pos[k + 2]); }
}

function veinTexture() {
  const W = 512, H = 512, c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
  g.lineCap = 'round';
  for (let k = -9; k <= 9; k++) {
    const v0 = k / 24, v1 = k / 8.6, wob = (k * 1.7) % 1;
    g.strokeStyle = `rgba(0,0,0,${k % 3 === 0 ? 0.22 : 0.12})`; g.lineWidth = k % 3 === 0 ? 2.2 : 1.2;
    g.beginPath();
    for (let s = 0; s <= 40; s++) {
      const u = 0.1 + 0.88 * (s / 40), t = (u - 0.1) / 0.88;
      const v = lerp(v0, v1, Math.pow(t, 1.2)) + Math.sin(t * 9 + wob * 6) * 0.012;
      const x = (v + 1) / 2 * W, y = (1 - u) * H;
      s ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
  }
  for (let n = 0; n < 5000; n++) { g.fillStyle = `rgba(0,0,0,${Math.random() * 0.035})`; g.fillRect(Math.random() * W, Math.random() * H, 1.4, 5 + Math.random() * 12); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

export class Stage {
  constructor(canvas) {
    this.cv = canvas;
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.NeutralToneMapping; r.toneMappingExposure = 1.0;
    r.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(26, 1, 1, 400); this.camera.position.set(0, 0, 80);
    const pm = new THREE.PMREMGenerator(r);
    this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;

    // hierarchy: pivot (screen position + scale) > tilt > spin > flower (z-up -> y-up)
    this.pivot = new THREE.Group(); this.tilt = new THREE.Group(); this.spin = new THREE.Group(); this.flower = new THREE.Group();
    this.flower.rotation.x = -Math.PI / 2; this.flower.position.y = -2.3;
    this.pivot.add(this.tilt); this.tilt.add(this.spin); this.spin.add(this.flower); this.scene.add(this.pivot);

    const key = new THREE.DirectionalLight(0xfff1dc, 2.6); key.position.set(-14, 22, 18); key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048); key.shadow.radius = 5; key.shadow.bias = -0.0004; key.shadow.normalBias = 0.04;
    Object.assign(key.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 70 });
    const rim = new THREE.DirectionalLight(0xdfeaff, 1.3); rim.position.set(16, 8, -20);
    this.pivot.add(key, rim, new THREE.HemisphereLight(0xffffff, 0xc9b79a, 0.55));

    const tex = veinTexture();
    this.petalMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, map: tex, bumpMap: tex, bumpScale: 0.7, roughness: 0.5, side: THREE.DoubleSide, sheen: 0.9, sheenRoughness: 0.45, sheenColor: new THREE.Color(0xffffff), clearcoat: 0.22, clearcoatRoughness: 0.4, envMapIntensity: 0.9 });
    this.leafMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, side: THREE.DoubleSide, envMapIntensity: 0.7 });
    this.petals = [];
    const add = (type, nu, nv, mat, parent = this.flower) => { const p = new Petal(type, nu, nv, mat); parent.add(p.mesh); this.petals.push(p); return p; };
    for (let k = 0; k < 3; k++) add('std', 40, 28, this.petalMat);
    for (let k = 0; k < 3; k++) add('fall', 52, 34, this.petalMat);
    for (let k = 0; k < 3; k++) add('style', 26, 12, this.petalMat);
    this.stemGroup = new THREE.Group(); this.flower.add(this.stemGroup);
    for (let k = 0; k < 4; k++) add('leaf', 34, 8, this.leafMat, this.stemGroup);
    const curve = new THREE.CatmullRomCurve3([0, 1, 2, 3, 4].map(i => new THREE.Vector3(Math.sin(i * 0.5) * 0.4, 0, -0.3 - i * 4.3)));
    this.stem = new THREE.Mesh(new THREE.TubeGeometry(curve, 30, 0.34, 10), new THREE.MeshStandardMaterial({ color: 0x4f7a3f, roughness: 0.5 }));
    this.stem.castShadow = true; this.stemGroup.add(this.stem);

    // beard: instanced bristles on the falls
    const bg = new THREE.ConeGeometry(0.05, 0.62, 5); bg.translate(0, 0.31, 0);
    this.beardCount = 3 * 72;
    this.beard = new THREE.InstancedMesh(bg, new THREE.MeshStandardMaterial({ color: 0xf6c24a, emissive: 0x6b4a08, roughness: 0.6 }), this.beardCount);
    this.beard.frustumCulled = false; this.flower.add(this.beard);
    this.seeds = Array.from({ length: 72 }, () => ({ u: 0.07 + Math.random() * 0.4, v: (Math.random() - 0.5) * 0.34, a: Math.random() * 6.28, t: 0.5 + Math.random() * 0.8 }));

    this.cur = { sl: 5.8, sw: 3.0, pl: 4.3, pw: 1.3, bloom: 0, yaw: 0, pitch: 0, m0: 0, m1: 1, m2: 0, hover: 0 };
    this.tgt = { ...this.cur };
    this.pointer = { x: 0, y: 0 }; this.drag = null;
    this.pose = { x: 0, y: 0, s: 0.01, stem: 0 };
    this.target = { x: 0, y: 0, s: 0.01, stem: 0, name: '', snap: true };
    this.t = 0; this.sig = ''; this.bloomAnim = null;
    this.onActive = null; this.onFrame = null; this.activeName = '';
    this.v3 = new THREE.Vector3(); this.dummy = new THREE.Object3D();
    this.pts = {};
    addEventListener('resize', () => this.resize()); this.resize();
    this.last = performance.now();
    requestAnimationFrame(t => this.loop(t));
  }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  setTraits(t, snap) { Object.assign(this.tgt, t); if (snap) Object.assign(this.cur, t); }
  setMix(p, snap) { this.tgt.m0 = p[0]; this.tgt.m1 = p[1]; this.tgt.m2 = p[2]; if (snap) { this.cur.m0 = p[0]; this.cur.m1 = p[1]; this.cur.m2 = p[2]; } }
  setBloom(b) { this.bloomAnim = null; this.cur.bloom = b; }
  bloom(to = 1, dur = 2.6, delay = 0) { this.bloomAnim = { from: this.cur.bloom, to, dur, delay, t: 0 }; }
  setPointer(x, y) { this.pointer.x = x; this.pointer.y = y; }
  pulse() { this.cur.hover = 1; }

  _pal() {
    const c = this.cur, w = [Math.max(0, c.m0), Math.max(0, c.m1), Math.max(0, c.m2)], s = w[0] + w[1] + w[2] || 1, out = {};
    for (const key of ['body', 'rim', 'claw', 'vein', 'gold']) out[key] = [0, 1, 2].map(ch => (w[0] * PALETTE[0][key][ch] + w[1] * PALETTE[1][key][ch] + w[2] * PALETTE[2][key][ch]) / s);
    return out;
  }

  _descriptors() {
    const c = this.cur, b = clamp(c.bloom, 0, 1);
    const fallL = c.sl * 1.05, fallW = 0.7 + c.sw * 0.78, stdL = 0.7 + c.pl * 0.95, stdW = 0.45 + c.pw * 1.7;
    const out = [];
    for (let k = 0; k < 3; k++) out.push({
      ox: 0, oy: 0, oz: 0.2, theta: (-30 + k * 120) * D2R, L: stdL * (0.45 + 0.55 * b), W: stdW,
      a0: lerp(88, 56, b) * D2R, a1: lerp(88, 90, b) * D2R, bend: 1.15, cup: 0.5, ruffle: 0.28, rf: 8, seed: k * 2.1, claw: 0.18, peak: 0.5, tip: 0.7, r0: 0.25,
    });
    for (let k = 0; k < 3; k++) out.push({
      ox: 0, oy: 0, oz: 0.1, theta: (-90 + k * 120) * D2R, L: fallL * (0.5 + 0.5 * b), W: fallW,
      a0: lerp(80, 24, b) * D2R, a1: lerp(82, -58, b) * D2R, bend: 1.3, cup: 0.3, ruffle: 0.7, rf: 12, seed: k * 3.7, claw: 0.14, peak: 0.52, tip: 0.66, r0: 0.3,
    });
    for (let k = 0; k < 3; k++) out.push({
      ox: 0, oy: 0, oz: 0.55, theta: (-90 + k * 120) * D2R, L: (0.5 * fallL + 0.6) * (0.4 + 0.6 * b), W: fallW * 0.42,
      a0: lerp(80, 58, b) * D2R, a1: lerp(84, -12, b) * D2R, bend: 1.05, cup: 0.9, ruffle: 0.25, rf: 9, seed: k + 9, claw: 0.55, peak: 0.25, tip: 0.55, tipPow: 1.7, r0: 0.35,
    });
    const L = [{ th: 200, L: 17, W: 1.6, a1: -8 }, { th: 330, L: 15, W: 1.4, a1: -2 }, { th: 262, L: 19, W: 1.7, a1: 20 }, { th: 100, L: 14, W: 1.3, a1: 25 }];
    L.forEach((l, i) => out.push({
      ox: Math.cos(l.th * D2R) * 0.3, oy: Math.sin(l.th * D2R) * 0.3, oz: -17 + 0.2 + i * 0.3, theta: l.th * D2R,
      L: l.L * (0.35 + 0.65 * b), W: l.W, a0: 84 * D2R, a1: lerp(80, l.a1, b) * D2R, bend: 1.9, cup: 0.9, ruffle: 0, rf: 1, seed: i, claw: 0.55, peak: 0.12, tip: 0.55, tipPow: 1.5, r0: 0.1,
    }));
    return out;
  }

  _rebuild() {
    const pal = this._pal(), d = this._descriptors();
    this.petals.forEach((p, i) => p.update(d[i], pal));
    // beard
    const falls = this.petals.slice(3, 6), n = new THREE.Vector3(), q = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    let m = 0;
    falls.forEach(f => {
      const nr = f.geo.attributes.normal;
      for (const s of this.seeds) {
        const i = clamp(Math.round(s.u * f.nu), 0, f.nu), j = clamp(Math.round((0.5 + s.v) * f.nv), 0, f.nv);
        f.at(i, j, q);
        const k = (i * (f.nv + 1) + j);
        n.set(nr.getX(k), nr.getY(k), nr.getZ(k)); if (n.z < 0) n.negate();
        n.x += Math.cos(s.a) * 0.35; n.y += Math.sin(s.a) * 0.35; n.normalize();
        this.dummy.position.copy(q); this.dummy.quaternion.setFromUnitVectors(up, n);
        const sc = (0.7 + s.t) * (0.6 + 0.4 * this.cur.bloom); this.dummy.scale.set(sc, sc, sc);
        this.dummy.updateMatrix(); this.beard.setMatrixAt(m++, this.dummy.matrix);
      }
    });
    this.beard.instanceMatrix.needsUpdate = true;
    const g = this.petals[0].at.bind(this.petals[0]);
    const f0 = this.petals[3], s0 = this.petals[0];
    this.pts = {
      fallTip: f0.at(f0.nu, f0.nv / 2, new THREE.Vector3()), fallBase: f0.at(2, f0.nv / 2, new THREE.Vector3()),
      fallEdge: f0.at(Math.round(f0.nu * 0.8), f0.nv, new THREE.Vector3()),
      stdTip: s0.at(s0.nu, s0.nv / 2, new THREE.Vector3()), stdEdge: s0.at(Math.round(s0.nu * 0.62), s0.nv, new THREE.Vector3()),
    };
  }

  _anchor() {
    const els = document.querySelectorAll('[data-flower]');
    const vh = innerHeight, vw = innerWidth;
    let best = null, bd = 1e9;
    els.forEach(e => {
      const r = e.getBoundingClientRect();
      if (r.bottom < -vh * 0.2 || r.top > vh * 1.2 || r.width < 4) return;
      const d = Math.abs(r.top + r.height / 2 - vh * 0.5);
      if (d < bd) { bd = d; best = { e, r }; }
    });
    if (!best) return;
    const { e, r } = best;
    const name = e.dataset.flower;
    const dist = this.camera.position.z, halfH = Math.tan(this.camera.fov * D2R / 2) * dist, halfW = halfH * this.camera.aspect;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const px = Math.min(r.width, r.height * (e.dataset.fit === 'h' ? 99 : 1));
    const s = (px / vh) * 2 * halfH / 12.5;
    this.target = { x: (cx / vw * 2 - 1) * halfW, y: -(cy / vh * 2 - 1) * halfH, s, stem: e.dataset.stem === '1' ? 1 : 0, name };
    if (name !== this.activeName) { this.activeName = name; this.onActive && this.onActive(name); }
  }

  loop(ts) {
    const dt = Math.min(0.05, (ts - this.last) / 1000); this.last = ts; this.t += dt;
    this._anchor();
    const c = this.cur, g = this.tgt, k = 1 - Math.exp(-dt * 6.5);
    for (const key of ['sl', 'sw', 'pl', 'pw', 'm0', 'm1', 'm2']) c[key] += (g[key] - c[key]) * k;
    if (this.bloomAnim) {
      const a = this.bloomAnim; a.t += dt;
      const p = clamp((a.t - a.delay) / a.dur, 0, 1);
      c.bloom = lerp(a.from, a.to, 1 - Math.pow(1 - p, 3.2));
      if (p >= 1) this.bloomAnim = null;
    }
    c.hover *= Math.exp(-dt * 2.4);
    const sig = [c.sl, c.sw, c.pl, c.pw, c.bloom, c.m0, c.m1, c.m2].map(v => Math.round(v * 400)).join(',');
    if (sig !== this.sig) { this.sig = sig; this._rebuild(); }

    // pose glide
    const P = this.pose, T = this.target, kp = 1 - Math.exp(-dt * 5.5);
    P.x += (T.x - P.x) * kp; P.y += (T.y - P.y) * kp; P.s += (T.s - P.s) * kp; P.stem += (T.stem - P.stem) * kp;
    this.pivot.position.set(P.x, P.y, 0); this.pivot.scale.setScalar(Math.max(P.s, 0.0001));
    this.stemGroup.visible = P.stem > 0.02; this.stemGroup.scale.set(1, Math.max(P.stem, 0.001), 1);

    const sway = Math.sin(this.t * 0.35) * 0.22 + Math.sin(this.t * 0.17) * 0.1;
    const ks = 1 - Math.exp(-dt * 3);
    c.yaw += ((this.drag ? this.drag.yaw : sway + this.pointer.x * 0.5) - c.yaw) * ks;
    c.pitch += ((this.drag ? this.drag.pitch : this.pointer.y * 0.16) - c.pitch) * ks;
    this.tilt.rotation.x = 0.62 + c.pitch;
    this.spin.rotation.y = c.yaw + 0.5 + scrollY * 0.0004;

    this.renderer.render(this.scene, this.camera);
    if (this.onFrame) {
      this.pivot.updateMatrixWorld(true);
      const proj = v => { const w = this.v3.copy(v); this.flower.localToWorld(w); w.project(this.camera); return [(w.x * 0.5 + 0.5) * innerWidth, (-w.y * 0.5 + 0.5) * innerHeight]; };
      const settled = Math.abs(P.x - T.x) + Math.abs(P.y - T.y) < 0.6 && Math.abs(P.s - T.s) / Math.max(T.s, 1e-3) < 0.06;
      this.onFrame({
        name: this.activeName, settled, bloom: c.bloom, hover: c.hover,
        fallEdge: proj(this.pts.fallEdge), stdEdge: proj(this.pts.stdEdge), fallBase: proj(this.pts.fallBase), fallTip: proj(this.pts.fallTip),
      });
    }
    requestAnimationFrame(t => this.loop(t));
  }
}
