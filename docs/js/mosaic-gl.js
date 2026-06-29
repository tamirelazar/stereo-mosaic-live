import { centers, bounds, invAffine3 } from "./mosaic-math.js";

const VS = `attribute vec2 a_pos; varying vec2 v_pano; uniform vec2 u_panoSize;
void main(){ v_pano = a_pos; vec2 c=(a_pos/u_panoSize)*2.0-1.0; gl_Position=vec4(c.x,-c.y,0.0,1.0); }`;
const FS = `precision highp float; varying vec2 v_pano; uniform sampler2D u_tex;
uniform mat3 u_Hinv; uniform vec2 u_goff,u_wh,u_bounds; uniform float u_feather,u_left,u_right,u_premul;
void main(){
  vec3 src=u_Hinv*vec3(v_pano.x+u_goff.x, v_pano.y+u_goff.y, 1.0);
  vec2 uv=vec2(src.x/u_wh.x, src.y/u_wh.y);
  if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0) discard;
  float w=1.0;
  if(u_feather>0.0){
    if(u_left>0.5)  w*=clamp((v_pano.x-(u_bounds.x-u_feather))/u_feather,0.0,1.0);
    if(u_right>0.5) w*=clamp(((u_bounds.y+u_feather)-v_pano.x)/u_feather,0.0,1.0);
  }
  vec3 col=texture2D(u_tex,uv).rgb;
  if(u_premul>0.5){ gl_FragColor=vec4(col*w,w); }
  else {
    float seam=1.5;
    float d=min(abs(v_pano.x-u_bounds.x), abs(v_pano.x-u_bounds.y));
    float dark=mix(0.12,1.0,smoothstep(0.0,seam,d));
    gl_FragColor=vec4(col*dark,w);
  }
}`;
const VS2 = `attribute vec2 a_p; varying vec2 v_uv;
void main(){ v_uv=(a_p+1.0)*0.5; gl_Position=vec4(a_p,0.0,1.0); }`;
const FS2 = `precision highp float; varying vec2 v_uv; uniform sampler2D u_acc;
void main(){ vec4 a=texture2D(u_acc,v_uv);
  if(a.a<=0.0001){ gl_FragColor=vec4(0.0,0.0,0.0,1.0); return; }
  gl_FragColor=vec4(a.rgb/a.a, 1.0); }`;

const FEATHER = 8.0;

function compile(gl, t, s){ const sh=gl.createShader(t); gl.shaderSource(sh,s); gl.compileShader(sh);
  if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(sh); return sh; }
function link(gl, vs, fs){ const p=gl.createProgram(); gl.attachShader(p,compile(gl,gl.VERTEX_SHADER,vs));
  gl.attachShader(p,compile(gl,gl.FRAGMENT_SHADER,fs)); gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw gl.getProgramInfoLog(p); return p; }

export class MosaicGL {
  constructor(gl, manifest, textures){
    if(!gl) throw new Error("WebGL2 required");
    if(!gl.getExtension("EXT_color_buffer_float") && !gl.getExtension("EXT_color_buffer_half_float"))
      throw new Error("RGBA16F render target unsupported");
    this.gl=gl; this.M=manifest; this.tex=textures;
    this.mode="xslit"; this.vp=0.5; this.slices=false;
    this.pStrip=link(gl,VS,FS); this.pResolve=link(gl,VS2,FS2);
    this.loc={}; for(const u of ["u_panoSize","u_Hinv","u_goff","u_wh","u_bounds","u_feather","u_left","u_right","u_tex","u_premul"])
      this.loc[u]=gl.getUniformLocation(this.pStrip,u);
    this.locPos=gl.getAttribLocation(this.pStrip,"a_pos");
    this.loc2={ p:gl.getAttribLocation(this.pResolve,"a_p"), acc:gl.getUniformLocation(this.pResolve,"u_acc") };
    this.buf=gl.createBuffer(); this.quad2=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,this.quad2);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
    const W=manifest.panorama_size[0], H=manifest.panorama_size[1];
    this.canvas=gl.canvas; this.canvas.width=W; this.canvas.height=H;
    this._makeFBO(W,H);
  }
  setMode(m){ this.mode=m; } setViewpoint(v){ this.vp=v; } setSlices(b){ this.slices=b; }
  _makeFBO(W,H){ const gl=this.gl;
    this.accTex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,this.accTex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,W,H,0,gl.RGBA,gl.HALF_FLOAT,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    this.fbo=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.accTex,0);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) throw new Error("FBO incomplete");
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  }
  _drawStrips(b,W,H){ const gl=this.gl, M=this.M;
    gl.useProgram(this.pStrip);
    gl.uniform2f(this.loc.u_panoSize,W,H); gl.uniform2f(this.loc.u_goff,M.global_offset[0],M.global_offset[1]);
    gl.uniform2f(this.loc.u_wh,M.w,M.h); gl.uniform1f(this.loc.u_feather,FEATHER); gl.uniform1i(this.loc.u_tex,0);
    gl.uniform1f(this.loc.u_premul, this.slices?0.0:1.0);
    for(let i=0;i<M.n;i++){
      const x0=Math.max(b[i]-FEATHER,0), x1=Math.min(b[i+1]+FEATHER,W); if(x1<=x0) continue;
      const bb=M.bounding_boxes[i], y0=Math.max(bb[0][1],0), y1=Math.min(bb[1][1],H);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.tex[i]);
      gl.uniformMatrix3fv(this.loc.u_Hinv,false,invAffine3(M.homographies[i]));
      gl.uniform2f(this.loc.u_bounds,b[i],b[i+1]);
      gl.uniform1f(this.loc.u_left,i>0?1:0); gl.uniform1f(this.loc.u_right,i<M.n-1?1:0);
      const q=new Float32Array([x0,y0,x1,y0,x0,y1,x0,y1,x1,y0,x1,y1]);
      gl.bindBuffer(gl.ARRAY_BUFFER,this.buf); gl.bufferData(gl.ARRAY_BUFFER,q,gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.locPos); gl.vertexAttribPointer(this.locPos,2,gl.FLOAT,false,0,0);
      gl.drawArrays(gl.TRIANGLES,0,6);
    }
  }
  render(){ const gl=this.gl, M=this.M, W=M.panorama_size[0], H=M.panorama_size[1];
    const c=centers(M,this.vp,this.mode), b=bounds(c,W,M.n);
    if(this.slices){
      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,this.canvas.width,this.canvas.height);
      gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
      this._drawStrips(b,W,H); return;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.fbo); gl.viewport(0,0,W,H);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);
    this._drawStrips(b,W,H);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,this.canvas.width,this.canvas.height);
    gl.disable(gl.BLEND); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.pResolve); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.accTex);
    gl.uniform1i(this.loc2.acc,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.quad2);
    gl.enableVertexAttribArray(this.loc2.p); gl.vertexAttribPointer(this.loc2.p,2,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.TRIANGLES,0,6);
  }
}
