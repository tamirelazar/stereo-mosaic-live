// Fetch the manifest and load all frame JPGs into WebGL textures, in parallel.
export async function loadAsset(gl, baseUrl) {
  const manifest = await (await fetch(baseUrl + "/manifest.json")).json();
  const load = (url) => new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      res(t);
    };
    img.onerror = rej;
    img.src = url;
  });
  const urls = [];
  for (let i = 1; i <= manifest.n; i++)
    urls.push(baseUrl + "/frame" + String(i).padStart(4, "0") + ".jpg");
  const textures = await Promise.all(urls.map(load));
  return { manifest, textures };
}
