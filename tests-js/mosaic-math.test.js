import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { centers, bounds, invAffine3, columnForFrame, columnsForMode, centersFromColumns } from "../docs/js/mosaic-math.js";
import M from "../docs/asset/manifest.json" with { type: "json" };

const fx = JSON.parse(readFileSync(new URL("./parity.json", import.meta.url)));
const Msmall = fx.manifest;
const EPS = 1e-6;

// Multiply two 3×3 matrices (row-major arrays) and return the result.
function matmul3(A, B) {
  const C = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      for (let k = 0; k < 3; k++)
        C[r][c] += A[r][k] * B[k][c];
  return C;
}

// Reconstruct a row-major 3×3 from a column-major flat array [m00,m10,m20,m01,m11,m21,m02,m12,m22].
function colMajorToRowMajor(arr) {
  return [
    [arr[0], arr[3], arr[6]],
    [arr[1], arr[4], arr[7]],
    [arr[2], arr[5], arr[8]],
  ];
}

test("invAffine3 round-trip: pure translation", () => {
  const H = [[1, 0, 5], [0, 1, -3], [0, 0, 1]];
  const Hinv = colMajorToRowMajor(invAffine3(H));
  const I = matmul3(Hinv, H);
  const expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const EPS9 = 1e-9;
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      assert.ok(Math.abs(I[r][c] - expected[r][c]) < EPS9,
        `Hinv·H[${r}][${c}] expected ${expected[r][c]} got ${I[r][c]}`);
});

test("invAffine3 round-trip: scale+shear affine", () => {
  const H = [[2, 0, 4], [0, 0.5, 1], [0, 0, 1]];
  const Hinv = colMajorToRowMajor(invAffine3(H));
  const I = matmul3(Hinv, H);
  const expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const EPS9 = 1e-9;
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      assert.ok(Math.abs(I[r][c] - expected[r][c]) < EPS9,
        `Hinv·H[${r}][${c}] expected ${expected[r][c]} got ${I[r][c]}`);
});

test("columnsForMode matches columnForFrame per index", () => {
  const cols = columnsForMode(M, 0.5, "pushbroom");
  assert.equal(cols.length, M.n);
  for (const i of [0, 108, M.n - 1])
    assert.equal(cols[i], columnForFrame(i, 0.5, "pushbroom", M.n, M.w));
});

test("centersFromColumns(columnsForMode) equals centers", () => {
  for (const mode of ["pushbroom", "xslit", "forward"]) {
    const viaCols = centersFromColumns(M, columnsForMode(M, 0.5, mode));
    const direct = centers(M, 0.5, mode);
    for (let i = 0; i < M.n; i++)
      assert.ok(Math.abs(viaCols[i] - direct[i]) < 1e-9, `mode ${mode} idx ${i}`);
  }
});

test("centers[108] pushbroom is the money number", () => {
  assert.ok(Math.abs(centers(M, 0.5, "pushbroom")[108] - 1890.04) < 0.01);
});

for (const mode of fx.modes) {
  for (const vp of fx.vps) {
    test(`centers/bounds parity ${mode} vp=${vp}`, () => {
      const exp = fx.expected[`${mode}_${vp.toFixed(1)}`];
      const cen = centers(Msmall, vp, mode);
      const bnd = bounds(cen, Msmall.panorama_size[0], Msmall.n);
      assert.equal(cen.length, exp.centers.length);
      for (let i = 0; i < cen.length; i++)
        assert.ok(Math.abs(cen[i] - exp.centers[i]) < EPS, `center ${i}`);
      for (let i = 0; i < bnd.length; i++)
        assert.ok(Math.abs(bnd[i] - exp.bounds[i]) < EPS, `bound ${i}`);
    });
  }
}
