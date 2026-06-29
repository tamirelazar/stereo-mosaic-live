import { initViewer } from "./viewer.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Mount a viewer on a canvas + controls container; returns the MosaicGL or null on failure.
export async function mountViewer(canvasId, controlsId, { mode } = {}) {
  const canvas = document.getElementById(canvasId);
  const controls = document.getElementById(controlsId);
  try {
    const m = await initViewer(canvas, controls, "./asset");
    if (mode) { m.setMode(mode); m.render(); }
    const vp = controls.querySelector("input.vp");
    const val = controls.querySelector(".val");
    if (vp && val) vp.addEventListener("input", () => { val.textContent = (+vp.value).toFixed(2); });
    return m;
  } catch (e) {
    document.body.classList.add("no-webgl2"); // Task 7 styles the fallback
    return null;
  }
}

// Hero layout: console is bottom-anchored in CSS; center the strip between head and console so
// extra window height splits evenly above/below the strip, and keep the chevron's margins fixed.
function heroLayout() {
  const stage = document.querySelector(".hero .stage");
  const head = document.querySelector(".hero .head");
  const con = document.getElementById("hero-controls");
  const hint = document.getElementById("scrollhint");
  if (!stage || !head || !con) return;
  const hb = head.getBoundingClientRect().bottom + scrollY;
  const ct = con.getBoundingClientRect().top + scrollY;
  stage.style.top = ((hb + ct) / 2) + "px";
  if (hint) {
    const cb = con.getBoundingClientRect().bottom + scrollY;
    hint.style.top = (((cb + innerHeight) / 2) - hint.offsetHeight / 2) + "px";
  }
}
function fadeHint() {
  const hint = document.getElementById("scrollhint");
  if (hint) hint.style.opacity = Math.max(0, 1 - scrollY / 120);
}

// Subtle idle attract loop: gently pans the viewpoint until the user touches any control.
function attract(m, controls) {
  if (reduceMotion || !m) return;
  let t = 0, on = true;
  const stop = () => { on = false; };
  controls.addEventListener("pointerdown", stop, { once: true });
  controls.addEventListener("keydown", stop, { once: true });
  (function loop() {
    if (!on) return;
    t += 0.0035;
    m.setViewpoint(0.5 + 0.32 * Math.sin(t));
    m.render();
    requestAnimationFrame(loop);
  })();
}

async function main() {
  const hero = await mountViewer("hero-canvas", "hero-controls", { mode: "xslit" });
  heroLayout(); fadeHint();
  await mountViewer("slices-canvas", "slices-controls", { mode: "xslit" });
  await mountViewer("lineage-canvas", "lineage-controls", { mode: "pushbroom" });
  addEventListener("resize", () => { heroLayout(); fadeHint(); });
  addEventListener("scroll", fadeHint, { passive: true });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(heroLayout);
  attract(hero, document.getElementById("hero-controls"));
}
main();
