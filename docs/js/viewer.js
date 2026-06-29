import { loadAsset } from "./loader.js";
import { MosaicGL } from "./mosaic-gl.js";

// Wire a canvas + a controls container to a fresh engine instance.
// controlsEl is expected to contain .modes button[data-m], input.vp, input.slices (optional).
export async function initViewer(canvasEl, controlsEl, baseUrl) {
  const gl = canvasEl.getContext("webgl2", { antialias: false });
  const { manifest, textures } = await loadAsset(gl, baseUrl);
  const m = new MosaicGL(gl, manifest, textures);
  const vp = controlsEl.querySelector("input.vp");
  const slices = controlsEl.querySelector("input.slices");
  const onMode = (btn) => {
    controlsEl.querySelectorAll(".modes button").forEach((b) => b.classList.remove("on"));
    btn.classList.add("on"); m.setMode(btn.dataset.m); m.render();
  };
  controlsEl.querySelectorAll(".modes button").forEach((b) =>
    b.addEventListener("click", () => onMode(b)));
  if (vp) vp.addEventListener("input", () => { m.setViewpoint(+vp.value); m.render(); });
  if (slices) slices.addEventListener("change", () => { m.setSlices(slices.checked); m.render(); });
  m.render();
  return m;
}
