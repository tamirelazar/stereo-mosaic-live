// Pure strip geometry — mirrors smlive/render.py::column_for_frame and the
// center/bounds computation. No DOM, no WebGL: node-testable.

// Source column (frame pixel coords) to centre frame i's strip on.
export function columnForFrame(i, viewpoint, mode, n, w) {
  const base = viewpoint * w;
  if (mode === "pushbroom") return base;
  const t = i / Math.max(n - 1, 1), span = w * 0.5;
  if (mode === "xslit") return base + (t - 0.5) * span;
  if (mode === "forward") return base + (t - 0.5) * (t - 0.5) * span;
  throw new Error("unknown mode: " + mode);
}

// apply a 3×3 affine (bottom row [0,0,1]) to [col, h/2], return warped x.
export function warpX(col, H, h) { return H[0][0] * col + H[0][1] * Math.floor(h / 2) + H[0][2]; }

// per-frame source columns for a discrete sampling mode.
export function columnsForMode(M, viewpoint, mode) {
  const cols = new Float64Array(M.n);
  for (let i = 0; i < M.n; i++) cols[i] = columnForFrame(i, viewpoint, mode, M.n, M.w);
  return cols;
}

// panorama-space strip centre for an explicit per-frame column array.
export function centersFromColumns(M, cols) {
  const c = new Float64Array(M.n);
  for (let i = 0; i < M.n; i++)
    c[i] = warpX(cols[i], M.homographies[i], M.h) - M.global_offset[0];
  return c;
}

// panorama-space strip centre for every frame (discrete mode).
export function centers(M, viewpoint, mode) {
  return centersFromColumns(M, columnsForMode(M, viewpoint, mode));
}

// strip boundaries = midpoints of consecutive centres, clamped, ends 0..W.
export function bounds(c, panoramaW, n) {
  const b = new Float64Array(n + 1);
  b[0] = 0; b[n] = panoramaW;
  for (let i = 0; i < n - 1; i++)
    b[i + 1] = Math.min(Math.max((c[i] + c[i + 1]) / 2, 0), panoramaW);
  return b;
}

// column-major inverse of a 3×3 affine, for WebGL uniformMatrix3fv.
export function invAffine3(H) {
  const a = H[0][0], b = H[0][1], c = H[0][2], d = H[1][0], e = H[1][1], f = H[1][2];
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, ic = (b * f - c * e) / det,
        id = -d / det, ie = a / det, iff = (c * d - a * f) / det;
  return [ia, id, 0, ib, ie, 0, ic, iff, 1];  // column-major
}
