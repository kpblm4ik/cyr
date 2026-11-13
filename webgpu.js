/* cyr — WebGPU-рендер сцены.
   Тот же интерфейс, что у WebGL-рендера в cyr.html:
     begin(pick,bg,VP,vd) → hi(v) / box(t,s,c,r) / cyl(t,s,c) → end()
     read(x,y) → Promise<Uint8Array[4]>  (пиксель карты нажатий, y сверху)
   Все боксы и цилиндры кадра собираются в буферы экземпляров и рисуются тремя instanced-вызовами:
     скруглённые боксы (сетка 7×7 на грань), плоские боксы (36 вершин — для деталей, где скругление меньше ~1.5 px) и цилиндры. */
window.cyrWebGPU = async function (cv, geo) {
  if (!window.isSecureContext) throw new Error('нужен https');
  if (!navigator.gpu) throw new Error('браузер не поддерживает WebGPU');

  /* адаптер: сначала дискретная видеокарта, потом любая, в крайнем случае — программный */
  let ad = null;
  for (const o of [{ powerPreference: 'high-performance' }, {}, { forceFallbackAdapter: true }]) {
    try { ad = await navigator.gpu.requestAdapter(o); } catch (e) { ad = null; }
    if (ad) break;
  }
  if (!ad) throw new Error('нет доступного GPU-адаптера');
  const dev = await ad.requestDevice();
  dev.onuncapturederror = e => console.warn('WebGPU:', e.error && e.error.message);

  const ctx = cv.getContext('webgpu');
  if (!ctx) { dev.destroy(); throw new Error('canvas не дал контекст webgpu'); }
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device: dev, format: fmt, alphaMode: 'opaque' });

  const info = ad.info || {};
  const label = [info.vendor, info.architecture || info.description].filter(Boolean).join(' ') || 'GPU';

  const WGSL = `
struct U { vp: mat4x4f, vd: vec3f, pick: f32 };
struct I { t: vec3f, r: f32, s: vec3f, hi: f32, col: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> ins: array<I>;
struct O { @builtin(position) p: vec4f, @location(0) n: vec3f, @location(1) @interpolate(flat) c: vec4f };
/* z из диапазона WebGL (-1..1) в диапазон WebGPU (0..1) */
fn clip(p: vec4f) -> vec4f { return vec4f(p.xy, p.z * .5 + p.w * .5, p.w); }
@vertex fn vbox(@location(0) g: vec3f, @builtin(instance_index) k: u32) -> O {
  let d = ins[k]; let a = abs(g); let sg = sign(g);
  let n = normalize(sg * max((a - .5) * 2., vec3f(0.)));
  let b = d.s * .5 - d.r;
  var o: O;
  o.p = clip(u.vp * vec4f(sg * min(a * 2., vec3f(1.)) * b + d.r * n + d.t, 1.));
  o.n = n; o.c = vec4f(d.col.rgb, d.hi); return o;
}
/* плоский бокс без скругления: позиция ±1 и нормаль грани берутся из вершинного буфера */
@vertex fn vflat(@location(0) g: vec3f, @location(1) nr: vec3f, @builtin(instance_index) k: u32) -> O {
  let d = ins[k];
  var o: O;
  o.p = clip(u.vp * vec4f(g * d.s * .5 + d.t, 1.));
  o.n = nr; o.c = vec4f(d.col.rgb, d.hi); return o;
}
@vertex fn vcyl(@location(0) cp: vec4f, @builtin(instance_index) k: u32) -> O {
  let d = ins[k];
  var n = vec3f(0., 0., cp.w);
  if (cp.w == 0.) { n = normalize(vec3f(cp.xy, 0.)); }
  var o: O;
  o.p = clip(u.vp * vec4f(cp.xyz * d.s + d.t, 1.));
  o.n = n; o.c = vec4f(d.col.rgb, d.hi); return o;
}
@fragment fn fs(v: O) -> @location(0) vec4f {
  if (u.pick > .5) { return vec4f(v.c.rgb, 1.); }
  let n = normalize(v.n); let L = normalize(vec3f(-.45, .85, .5)); let h = dot(n, L) * .5 + .5;
  var c = v.c.rgb * (.62 + .5 * h * h);
  c = mix(c, c * vec3f(.9, .94, 1.06), 1. - h);
  let sp = pow(max(dot(n, normalize(L + u.vd)), 0.), 22.) * .22;
  let rim = pow(1. - max(dot(n, u.vd), 0.), 3.) * .2;
  return vec4f(c + vec3f(sp + rim + v.c.a), 1.);
}`;
  const mod = dev.createShaderModule({ code: WGSL });
  const bgl = dev.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
  ] });
  const layout = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipe = (entry, stride, attrs, format, samples) => dev.createRenderPipeline({
    layout,
    vertex: { module: mod, entryPoint: entry, buffers: [{ arrayStride: stride, attributes: attrs }] },
    fragment: { module: mod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: samples }
  });
  const A3 = [{ shaderLocation: 0, offset: 0, format: 'float32x3' }];
  const A4 = [{ shaderLocation: 0, offset: 0, format: 'float32x4' }];
  const AF = [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }];
  const mk = (format, samples) => ({ box: pipe('vbox', 12, A3, format, samples), flat: pipe('vflat', 24, AF, format, samples), cyl: pipe('vcyl', 16, A4, format, samples) });
  /* основной кадр — с 4x MSAA (как antialias:true в WebGL), карта нажатий — без сглаживания, чтобы id не смешивались */
  const P = { main: mk(fmt, 4), pick: mk('rgba8unorm', 1) };

  const vbuf = a => {
    const b = dev.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
    new Float32Array(b.getMappedRange()).set(a); b.unmap(); return b;
  };
  const bbuf = vbuf(new Float32Array(geo.V)), cbuf = vbuf(new Float32Array(geo.CV));
  const NV = geo.V.length / 3, NC = geo.CV.length / 4;
  /* плоский куб: 6 граней × 6 вершин, у каждой вершины позиция (±1) и нормаль грани */
  const FLT = [];
  for (let a = 0; a < 3; a++) for (const f of [1, -1]) {
    const u = (a + 1) % 3, v = (a + 2) % 3;
    for (const [q, w] of [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]]) {
      const p = [0, 0, 0], n = [0, 0, 0];
      p[a] = f; p[u] = q ? 1 : -1; p[v] = w ? 1 : -1; n[a] = f;
      FLT.push(p[0], p[1], p[2], n[0], n[1], n[2]);
    }
  }
  const fbuf = vbuf(new Float32Array(FLT)), NF = 36;
  const ubuf = dev.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const U = new Float32Array(20);

  /* буфер экземпляров: 12 float на деталь (t,r,s,hi,col,_), растёт по мере надобности */
  const F = 12;
  function batch() {
    return { a: new Float32Array(F * 256), n: 0, buf: null, bg: null, cap: 0 };
  }
  const B = batch(), FB = batch(), C = batch();
  function push(q, t, s, c, r, hi) {
    if ((q.n + 1) * F > q.a.length) { const na = new Float32Array(q.a.length * 2); na.set(q.a); q.a = na; }
    const o = q.n * F, a = q.a;
    a[o] = t[0]; a[o + 1] = t[1]; a[o + 2] = t[2]; a[o + 3] = r;
    a[o + 4] = s[0]; a[o + 5] = s[1]; a[o + 6] = s[2]; a[o + 7] = hi;
    a[o + 8] = c[0]; a[o + 9] = c[1]; a[o + 10] = c[2]; a[o + 11] = 1;
    q.n++;
  }
  function upload(q) {
    const bytes = Math.max(q.n, 1) * F * 4;
    if (bytes > q.cap) {
      if (q.buf) q.buf.destroy();
      q.cap = Math.max(bytes, q.a.byteLength);
      q.buf = dev.createBuffer({ size: q.cap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      q.bg = dev.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: q.buf } }] });
    }
    if (q.n) dev.queue.writeBuffer(q.buf, 0, q.a.buffer, 0, q.n * F * 4);
  }

  /* цели рендера пересоздаются при смене размера canvas; виды (views) кэшируются, а цели для карты нажатий создаются только при первом тапе */
  let tw = 0, th = 0, texs = [], msaaV = null, depth4V = null, pickTex = null, depth1 = null, pickV = null, depth1V = null;
  function targets(w, h) {
    if (w === tw && h === th) return;
    texs.forEach(t => t.destroy()); texs = [];
    if (pickTex) { pickTex.destroy(); depth1.destroy(); pickTex = depth1 = pickV = depth1V = null; }
    tw = w; th = h;
    const RA = GPUTextureUsage.RENDER_ATTACHMENT, size = [w, h];
    const m = dev.createTexture({ size, format: fmt, sampleCount: 4, usage: RA });
    const d = dev.createTexture({ size, format: 'depth24plus', sampleCount: 4, usage: RA });
    texs.push(m, d); msaaV = m.createView(); depth4V = d.createView();
  }
  function pickTargets() {
    if (pickTex) return;
    const RA = GPUTextureUsage.RENDER_ATTACHMENT, size = [tw, th];
    depth1 = dev.createTexture({ size, format: 'depth24plus', usage: RA });
    pickTex = dev.createTexture({ size, format: 'rgba8unorm', usage: RA | GPUTextureUsage.COPY_SRC });
    depth1V = depth1.createView(); pickV = pickTex.createView();
  }

  let pick = false, bg = [0, 0, 0], hi = 0, lost = false, ppu = 100;
  const R = {
    name: 'WebGPU', label, onlost: null,
    begin(pk, bgc, VP, vd) {
      pick = !!pk; bg = bgc; hi = 0; B.n = 0; FB.n = 0; C.n = 0;
      U.set(VP, 0); U[16] = vd[0]; U[17] = vd[1]; U[18] = vd[2]; U[19] = pick ? 1 : 0;
      /* пикселей на единицу мира (камера ортографическая) — по нему решаем, нужно ли скругление */
      ppu = Math.hypot(VP[0], VP[4], VP[8]) * cv.width * .5;
    },
    hi(v) { hi = v; },
    box(t, s, c, r) { push(r * ppu < 1.5 ? FB : B, t, s, c, r, hi); },
    cyl(t, s, c) { push(C, t, s, c, 0, hi); },
    end() {
      if (lost) return;
      const w = cv.width, h = cv.height;
      if (!w || !h) return;
      targets(w, h);
      if (pick) pickTargets();
      dev.queue.writeBuffer(ubuf, 0, U);
      upload(B); upload(FB); upload(C);
      const enc = dev.createCommandEncoder();
      const ca = pick
        ? { view: pickV, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }
        : { view: msaaV, resolveTarget: ctx.getCurrentTexture().createView(), clearValue: { r: bg[0], g: bg[1], b: bg[2], a: 1 }, loadOp: 'clear', storeOp: 'discard' };
      const pass = enc.beginRenderPass({
        colorAttachments: [ca],
        depthStencilAttachment: { view: pick ? depth1V : depth4V, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' }
      });
      const pp = pick ? P.pick : P.main;
      if (B.n) { pass.setPipeline(pp.box); pass.setBindGroup(0, B.bg); pass.setVertexBuffer(0, bbuf); pass.draw(NV, B.n); }
      if (FB.n) { pass.setPipeline(pp.flat); pass.setBindGroup(0, FB.bg); pass.setVertexBuffer(0, fbuf); pass.draw(NF, FB.n); }
      if (C.n) { pass.setPipeline(pp.cyl); pass.setBindGroup(0, C.bg); pass.setVertexBuffer(0, cbuf); pass.draw(NC, C.n); }
      pass.end();
      dev.queue.submit([enc.finish()]);
    },
    async read(x, y) {
      const px = new Uint8Array(4);
      if (lost || !pickTex) return px;
      x = Math.max(0, Math.min(tw - 1, x)); y = Math.max(0, Math.min(th - 1, y));
      const rb = dev.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = dev.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: pickTex, origin: { x, y } }, { buffer: rb, bytesPerRow: 256 }, [1, 1]);
      dev.queue.submit([enc.finish()]);
      try { await rb.mapAsync(GPUMapMode.READ); px.set(new Uint8Array(rb.getMappedRange(), 0, 4)); rb.unmap(); }
      catch (e) { console.warn('WebGPU read:', e); }
      rb.destroy();
      return px;
    }
  };
  dev.lost.then(i => { lost = true; if (i.reason !== 'destroyed' && R.onlost) R.onlost(i.message || 'устройство потеряно'); });
  return R;
};
