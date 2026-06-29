import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { centers, bounds } from "../docs/js/mosaic-math.js";

const fx = JSON.parse(readFileSync(new URL("./parity.json", import.meta.url)));
const M = fx.manifest;
const EPS = 1e-6;

for (const mode of fx.modes) {
  for (const vp of fx.vps) {
    test(`centers/bounds parity ${mode} vp=${vp}`, () => {
      const exp = fx.expected[`${mode}_${vp.toFixed(1)}`];
      const cen = centers(M, vp, mode);
      const bnd = bounds(cen, M.panorama_size[0], M.n);
      assert.equal(cen.length, exp.centers.length);
      for (let i = 0; i < cen.length; i++)
        assert.ok(Math.abs(cen[i] - exp.centers[i]) < EPS, `center ${i}`);
      for (let i = 0; i < bnd.length; i++)
        assert.ok(Math.abs(bnd[i] - exp.bounds[i]) < EPS, `bound ${i}`);
    });
  }
}
