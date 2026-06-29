import { loadAsset } from "./loader.js";
import { MosaicGL } from "./mosaic-gl.js";

// Wire a canvas + a controls container to a fresh engine instance.
// controlsEl is expected to contain .modes button[data-m], input.vp, input.slices (optional).
// opts.drag: enable grab-and-scrub the panorama strip to change viewpoint (hero only).
export async function initViewer(canvasEl, controlsEl, baseUrl, opts = {}) {
  const gl = canvasEl.getContext("webgl2", { antialias: false });
  if (!gl) throw new Error("WebGL2 not supported");
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
  // Grab-and-scrub: drag horizontally across the strip to move the viewpoint. Drives the .vp
  // slider (so the read-out, render, and idle-attract stop all stay wired through one input event)
  // and lifts the strip off the background via the .grabbing class while held.
  if (opts.drag && vp) {
    let downX = 0, startVp = 0, dragging = false;
    canvasEl.addEventListener("pointerdown", (e) => {
      dragging = true; downX = e.clientX; startVp = +vp.value;
      try { canvasEl.setPointerCapture(e.pointerId); } catch {}
      canvasEl.classList.add("grabbing");
      e.preventDefault();
    });
    canvasEl.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = (e.clientX - downX) / (canvasEl.clientWidth || 1);
      const nv = Math.min(1, Math.max(0, startVp + dx * 1.3));
      vp.value = nv;
      vp.dispatchEvent(new Event("input"));
    });
    const end = (e) => {
      if (!dragging) return;
      dragging = false;
      canvasEl.classList.remove("grabbing");
      try {
        if (e && e.pointerId != null && canvasEl.hasPointerCapture(e.pointerId))
          canvasEl.releasePointerCapture(e.pointerId);
      } catch {}
    };
    canvasEl.addEventListener("pointerup", end);
    canvasEl.addEventListener("pointercancel", end);
  }
  m.render();
  return m;
}
