(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))f(a);new MutationObserver(a=>{for(const i of a)if(i.type==="childList")for(const u of i.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&f(u)}).observe(document,{childList:!0,subtree:!0});function T(a){const i={};return a.integrity&&(i.integrity=a.integrity),a.referrerPolicy&&(i.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?i.credentials="include":a.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function f(a){if(a.ep)return;a.ep=!0;const i=T(a);fetch(a.href,i)}})();const ze=`@group(0) @binding(0) var samplerLinear : sampler;\r
@group(0) @binding(1) var inputTexture : texture_2d<f32>;\r
\r
struct VertexOutput {\r
  @builtin(position) Position : vec4<f32>,\r
  @location(0) fragUV : vec2<f32>,\r
};\r
\r
@vertex\r
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {\r
  let positions = array<vec2<f32>, 4>(\r
    vec2(-1.0, -1.0),\r
    vec2( 1.0, -1.0),\r
    vec2(-1.0,  1.0),\r
    vec2( 1.0,  1.0)\r
  );\r
  let uvs = array<vec2<f32>, 4>(\r
    vec2(0.0, 1.0),\r
    vec2(1.0, 1.0),\r
    vec2(0.0, 0.0),\r
    vec2(1.0, 0.0)\r
  );\r
\r
  var output : VertexOutput;\r
  output.Position = vec4(positions[vertexIndex], 0.0, 1.0);\r
  output.fragUV = uvs[vertexIndex];\r
  return output;\r
}\r
\r
@fragment\r
fn fs_main(@location(0) fragUV: vec2<f32>) -> @location(0) vec4<f32> {\r
  let texSize = vec2<f32>(textureDimensions(inputTexture));\r
  let offset = vec2<f32>(1.0) / texSize;\r
\r
  var color = vec4<f32>(0.0);\r
\r
  // Increased to 9x9 blur kernel for stronger effect\r
  let kernelSize = 9;\r
  let halfKernel = kernelSize / 2;\r
  let weight = 1.0 / f32(kernelSize * kernelSize);\r
\r
  for (var x = -halfKernel; x <= halfKernel; x++) {\r
    for (var y = -halfKernel; y <= halfKernel; y++) {\r
      let sampleUV = fragUV + vec2<f32>(f32(x), f32(y)) * offset;\r
      color += textureSample(inputTexture, samplerLinear, sampleUV) * weight;\r
    }\r
  }\r
\r
  return vec4(color.rgb, 1.0);\r
}\r
`,De=`/* cnn_segmentation.wgsl  – 3×3 conv on luminance, sigmoid, 4-nbr smooth\r
   Centre pixel is multiplied by its weight again, so different crops\r
   really produce different masks → non-zero gradients.                */\r
\r
@group(0) @binding(0) var<uniform> texSize : vec2<f32>;\r
@group(0) @binding(1) var           input   : texture_2d<f32>;\r
@group(0) @binding(2) var           weight  : texture_2d<f32>;  // r32float 3×3\r
@group(0) @binding(3) var           output  : texture_storage_2d<rgba8unorm, write>;\r
\r
fn luminance(rgb: vec3<f32>) -> f32 {\r
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));   // BT-709 Y\r
}\r
\r
fn σ(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }\r
\r
fn conv_at(px: vec2<i32>) -> f32 {\r
  let dims = vec2<i32>(texSize);\r
  var sum  = 0.0;\r
\r
  for (var dx = -1; dx <= 1; dx++) {\r
    for (var dy = -1; dy <= 1; dy++) {\r
      let texel = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);\r
      let lum   = luminance(textureLoad(input, texel, 0).rgb);\r
      let w     = textureLoad(weight, vec2<i32>(dx + 1, dy + 1), 0).r;\r
      sum += lum * w;                      // centre pixel multiplied again\r
    }\r
  }\r
  /* simple bias: a constant +0.1 (acts like centre-weight was before) */\r
  return sum + 0.1;\r
}\r
\r
@compute @workgroup_size(8, 8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\r
  if (gid.x >= u32(texSize.x) || gid.y >= u32(texSize.y)) { return; }\r
\r
  let uv        = vec2<i32>(gid.xy);\r
  let score     = conv_at(uv);\r
  let p_self    = σ(score);\r
\r
  /* 4-neighbour smoothing */\r
  var p_sum = p_self;\r
  let nbr = array<vec2<i32>,4>(vec2(1,0), vec2(-1,0), vec2(0,1), vec2(0,-1));\r
  for (var i=0u; i<4u; i++) {\r
    let q = clamp(uv + nbr[i], vec2<i32>(0), vec2<i32>(texSize) - 1);\r
    p_sum += σ(conv_at(q));\r
  }\r
  let p = p_sum / 5.0;\r
\r
  textureStore(output, uv, vec4<f32>(p, p, p, 1.0));\r
}\r
`,Fe=`struct AugParams {        // offset.x, offset.y, scale.x, scale.y\r
  offScale : vec4<f32>,   // colour jitter packed in .zw‐component of next vec4\r
  colour   : vec4<f32>,   // r,g,b,unused   (values around 1.0 = no change)\r
};\r
@group(0) @binding(0) var<uniform> params : AugParams;\r
@group(0) @binding(1) var samp   : sampler;\r
@group(0) @binding(2) var srcTex : texture_2d<f32>;\r
@group(0) @binding(3) var dstTex : texture_storage_2d<rgba8unorm, write>;\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>)\r
{\r
  let dims = textureDimensions(dstTex);\r
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
  let uv   = (vec2<f32>(gid.xy) / vec2<f32>(dims)) * params.offScale.zw + params.offScale.xy;\r
  let pix  = textureSampleLevel(srcTex, samp, uv, 0.0).rgb * params.colour.xyz;\r
  textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(pix, 1.0));\r
}\r
`,Re=`// Very small affine warp: inverse crop from view-B → view-A space.\r
struct Warp { offScale : vec4<f32>, };\r
@group(0) @binding(0) var src : texture_2d<f32>;\r
@group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;\r
@group(0) @binding(2) var<uniform> mat : Warp;\r
@group(0) @binding(3) var samp : sampler;\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>)\r
{\r
  let dims = textureDimensions(dst);\r
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
  let uv  = (vec2<f32>(gid.xy) / vec2<f32>(dims)) * mat.offScale.zw + mat.offScale.xy;\r
  let m   = textureSampleLevel(src, samp, uv, 0.0).r;\r
  textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(m, m, m, 1.0));\r
}\r
`,qe=`/* loss.wgsl  – AugCo gradient, integer-atomics version  */\r
const SCALE : f32 = 1e6;              // converts float-grad ➞ fixed-point\r
\r
struct Grad { vals : array<atomic<i32>, 9>, };\r
\r
@group(0) @binding(0) var maskA  : texture_2d<f32>;\r
@group(0) @binding(1) var maskBW : texture_2d<f32>;     // B mask warped → A\r
@group(0) @binding(2) var feats  : texture_2d<f32>;      // augA intensities\r
@group(0) @binding(3) var<storage, read_write> grad : Grad;\r
\r
fn σp(s: f32) -> f32 { return s * (1.0 - s); }          // sigmoid derivative\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>)\r
{\r
  let dims = textureDimensions(maskA);\r
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
  let uv   = vec2<i32>(gid.xy);\r
  let a    = textureLoad(maskA , uv, 0).r;\r
  let b    = textureLoad(maskBW, uv, 0).r;\r
  let err  = a - b;                       // dL/ds  (½‖a−b‖²)\r
\r
  var k = 0u;\r
  for (var dx = -1; dx <= 1; dx++) {\r
    for (var dy = -1; dy <= 1; dy++) {\r
      let coord = clamp(uv + vec2<i32>(dx,dy),\r
                        vec2<i32>(0), vec2<i32>(dims) - 1);\r
      let x   = textureLoad(feats, coord, 0).r;\r
      let g   = err * σp(a) * x;          // ∂L/∂w\r
      let gi  = i32(round(g * SCALE));    // fixed-point\r
      atomicAdd(&grad.vals[k], gi);\r
      k += 1u;\r
    }\r
  }\r
}\r
`,Ie=`@group(0) @binding(0) var prevTex : texture_2d<f32>;\r
@group(0) @binding(1) var currTex : texture_2d<f32>;\r
@group(0) @binding(2) var flowTex : texture_storage_2d<rgba16float, write>;   // xy-flow, z unused\r
\r
const K = 1.0; // Horn-Schunck single-pass approximation\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>)\r
{\r
  let dims = textureDimensions(currTex);\r
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
  let uv   = vec2<i32>(gid.xy);\r
  let i_x  = (textureLoad(currTex, uv + vec2<i32>(1,0),0).r -\r
              textureLoad(currTex, uv + vec2<i32>(-1,0),0).r) * 0.5;\r
  let i_y  = (textureLoad(currTex, uv + vec2<i32>(0,1),0).r -\r
              textureLoad(currTex, uv + vec2<i32>(0,-1),0).r) * 0.5;\r
  let i_t  = textureLoad(currTex, uv, 0).r - textureLoad(prevTex, uv, 0).r;\r
\r
  // one-shot HS update\r
  let denom = K*K + i_x*i_x + i_y*i_y;\r
  let u = -K * i_x * i_t / denom;\r
  let v = -K * i_y * i_t / denom;\r
\r
  textureStore(flowTex, uv, vec4<f32>(u, v, 0.0, 0.0));\r
}\r
`,Ne=`@group(0) @binding(0) var srcFlow : texture_2d<f32>;\r
@group(0) @binding(1) var samp    : sampler;\r
@group(0) @binding(2) var dstFlow : texture_storage_2d<rgba16float, write>;\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\r
  let dstDims = textureDimensions(dstFlow);\r
  if (gid.x >= dstDims.x || gid.y >= dstDims.y) { return; }\r
\r
  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dstDims);\r
  let flow = textureSampleLevel(srcFlow, samp, uv, 0.0).xy * 2.0;  // scale flow by 2× per pyramid level\r
\r
  textureStore(dstFlow, vec2<i32>(gid.xy), vec4<f32>(flow, 0.0, 0.0));\r
}\r
`,Ke=`/* Show webcam + green-tinted alpha mask (probability).  */\r
\r
@group(0) @binding(0) var samp      : sampler;\r
@group(0) @binding(1) var videoTex  : texture_2d<f32>;\r
@group(0) @binding(2) var maskTex   : texture_2d<f32>;\r
\r
struct VOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32>, };\r
\r
@vertex\r
fn vs_main(@builtin(vertex_index) i : u32) -> VOut {\r
  var pos = array<vec2<f32>,4>(vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));\r
  var uv  = array<vec2<f32>,4>(vec2(0,1),  vec2(1,1),  vec2(0,0),  vec2(1,0));\r
  var o : VOut;\r
  o.pos = vec4<f32>(pos[i], 0.0, 1.0);\r
  o.uv  = uv[i];\r
  return o;\r
}\r
\r
@fragment\r
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {\r
  let rgb   = textureSample(videoTex, samp, uv).rgb;\r
  let m     = textureSample(maskTex , samp, uv).r;   // 0‒1\r
  let tint  = mix(rgb, vec3(0.0, 1.0, 0.0), m * 0.7); // 70 % green where mask=1\r
  return vec4<f32>(tint, 1.0);\r
}`,Ye=`// Warp previous mask by dense optical flow\r
@group(0) @binding(0) var prevMask : texture_2d<f32>;\r
@group(0) @binding(1) var flowTex  : texture_2d<f32>;\r
@group(0) @binding(2) var outWarp  : texture_storage_2d<rgba8unorm, write>;\r
@group(0) @binding(3) var samp     : sampler;\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\r
  let dims = textureDimensions(outWarp);\r
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
  let uv = vec2<f32>(gid.xy) / vec2<f32>(dims);\r
  // fetch flow (u,v) in pixel space\r
  let f  = textureLoad(flowTex, vec2<i32>(gid.xy), 0).xy;\r
  // adjust sampling coords\r
  let prevUV = uv + f / vec2<f32>(dims);\r
  let val    = textureSampleLevel(prevMask, samp, prevUV, 0.0).r;\r
  textureStore(outWarp, vec2<i32>(gid.xy), vec4<f32>(val, val, val, 1.0));\r
}\r
`,je=`// Compute L2 error between current & warped‐previous masks\r
const SCALE_F : f32 = 1e6;  // if you want fixed‐point\r
\r
struct Grad { vals: array<atomic<i32>, 1>, }; // one global accumulator\r
\r
@group(0) @binding(0) var currMask : texture_2d<f32>;\r
@group(0) @binding(1) var prevWarp : texture_2d<f32>;\r
@group(0) @binding(2) var<storage, read_write> grad  : Grad;\r
\r
@compute @workgroup_size(8,8)\r
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\r
  let dims = textureDimensions(currMask);\r
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
  let uvi = vec2<i32>(gid.xy);\r
  let c   = textureLoad(currMask, uvi, 0).r;\r
  let p   = textureLoad(prevWarp, uvi, 0).r;\r
  let err = c - p;               // dL/dc for ½‖c−p‖²\r
  let gi  = i32(round(err * SCALE_F));\r
  atomicAdd(&grad.vals[0], gi);\r
}\r
`,S=document.getElementById("canvas"),y=document.getElementById("video"),He=document.getElementById("grad-l1"),Xe=document.getElementById("video-fps"),Je=document.getElementById("inference-fps");(async()=>{const k=await navigator.gpu.requestAdapter();if(!k){alert("WebGPU not supported");return}const e=await k.requestDevice(),T=S.getContext("webgpu"),f=navigator.gpu.getPreferredCanvasFormat();T.configure({device:e,format:f});const a=await navigator.mediaDevices.getUserMedia({video:!0});y.srcObject=a,await new Promise(n=>y.onloadedmetadata=n),S.width=y.videoWidth,S.height=y.videoHeight;const i=[S.width,S.height],u=GPUTextureUsage.TEXTURE_BINDING,l=GPUTextureUsage.STORAGE_BINDING,L=GPUTextureUsage.RENDER_ATTACHMENT,v=GPUTextureUsage.COPY_DST,se=GPUTextureUsage.COPY_SRC,s=(n,t="rgba8unorm",o=i)=>e.createTexture({size:o,format:t,usage:n}),p=e.createSampler({magFilter:"linear",minFilter:"linear"}),M=3,d=Array.from({length:M},(n,t)=>[Math.max(1,i[0]>>t),Math.max(1,i[1]>>t)]);let w=d.map(n=>s(u|L|v,"bgra8unorm",n)),x=d.map(n=>s(u|L|v,"bgra8unorm",n)),_=d.map(n=>s(u|l|v,"rgba16float",n));const P=s(u|v|L),U=s(u|l),D=s(u|l),F=s(u|l|L),R=s(u|l),q=s(u|l),V=s(u|l|se,"rgba8unorm",i),I=s(u|l,"rgba8unorm"),N=s(u|v,"rgba8unorm",i);let E=new Float32Array(9).fill(.1);const h=e.createTexture({size:[3,3],format:"r32float",usage:u|v});e.queue.writeTexture({texture:h},E,{bytesPerRow:3*4},[3,3]);const K=e.createShaderModule({code:ze}),de=e.createShaderModule({code:De}),le=e.createShaderModule({code:Fe}),pe=e.createShaderModule({code:Re}),ge=e.createShaderModule({code:qe}),me=e.createShaderModule({code:Ie}),fe=e.createShaderModule({code:Ne}),Y=e.createShaderModule({code:Ke}),ve=e.createShaderModule({code:Ye}),xe=e.createShaderModule({code:je}),j=e.createRenderPipeline({layout:"auto",vertex:{module:K,entryPoint:"vs_main"},fragment:{module:K,entryPoint:"fs_main",targets:[{format:f}]},primitive:{topology:"triangle-strip"}}),H=e.createComputePipeline({layout:"auto",compute:{module:fe,entryPoint:"main"}}),X=e.createComputePipeline({layout:"auto",compute:{module:me,entryPoint:"main"}}),b=e.createComputePipeline({layout:"auto",compute:{module:de,entryPoint:"main"}}),A=e.createComputePipeline({layout:"auto",compute:{module:le,entryPoint:"main"}}),J=e.createComputePipeline({layout:"auto",compute:{module:pe,entryPoint:"main"}}),Q=e.createComputePipeline({layout:"auto",compute:{module:ge,entryPoint:"main"}}),Z=e.createComputePipeline({layout:"auto",compute:{module:ve,entryPoint:"main"}}),$=e.createComputePipeline({layout:"auto",compute:{module:xe,entryPoint:"main"}}),ee=e.createRenderPipeline({layout:"auto",vertex:{module:Y,entryPoint:"vs_main"},fragment:{module:Y,entryPoint:"fs_main",targets:[{format:f}]},primitive:{topology:"triangle-strip"}}),B=e.createBuffer({size:9*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),C=e.createBuffer({size:9*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),O=(n=32)=>e.createBuffer({size:n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),re=O(),ne=O(),te=O(16),G=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});new Float32Array(G.getMappedRange()).set(i),G.unmap();function be(n,t){return e.createBindGroup({layout:H.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:p},{binding:2,resource:t.createView()}]})}function ye(n,t,o){return e.createBindGroup({layout:X.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:t.createView()},{binding:2,resource:o.createView()}]})}function we(n,t,o){return e.createBindGroup({layout:Z.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:t.createView()},{binding:2,resource:o.createView()},{binding:3,resource:p}]})}function _e(n,t){return e.createBindGroup({layout:$.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:t.createView()},{binding:2,resource:{buffer:B}}]})}function Pe(){return e.createBindGroup({layout:ee.getBindGroupLayout(0),entries:[{binding:0,resource:p},{binding:1,resource:P.createView()},{binding:2,resource:V.createView()}]})}const he=e.createBindGroup({layout:A.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:re}},{binding:1,resource:p},{binding:2,resource:P.createView()},{binding:3,resource:U.createView()}]}),Be=e.createBindGroup({layout:A.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ne}},{binding:1,resource:p},{binding:2,resource:P.createView()},{binding:3,resource:D.createView()}]}),Ge=e.createBindGroup({layout:b.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:G}},{binding:1,resource:U.createView()},{binding:2,resource:h.createView()},{binding:3,resource:F.createView()}]}),Se=e.createBindGroup({layout:b.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:G}},{binding:1,resource:D.createView()},{binding:2,resource:h.createView()},{binding:3,resource:R.createView()}]}),Te=e.createBindGroup({layout:J.getBindGroupLayout(0),entries:[{binding:0,resource:R.createView()},{binding:1,resource:q.createView()},{binding:2,resource:{buffer:te}},{binding:3,resource:p}]}),Le=e.createBindGroup({layout:Q.getBindGroupLayout(0),entries:[{binding:0,resource:F.createView()},{binding:1,resource:q.createView()},{binding:2,resource:U.createView()},{binding:3,resource:{buffer:B}}]}),Me=e.createBindGroup({layout:b.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:G}},{binding:1,resource:P.createView()},{binding:2,resource:h.createView()},{binding:3,resource:V.createView()}]});function ie(){const n=.6+Math.random()*.3,t=Math.random()*(1-n),o=Math.random()*(1-n),r=()=>.1+Math.random()*.8;return new Float32Array([t,o,n,n,r(),r(),r(),0])}function Ve(){return new Float32Array([(Math.random()-.5)*.2,(Math.random()-.5)*.2,1,1])}function g(n,t,o){n.setPipeline(t),n.setBindGroup(0,o),n.dispatchWorkgroups(Math.ceil(i[0]/8),Math.ceil(i[1]/8))}function oe(n,t){const o=e.createCommandEncoder(),r=o.beginRenderPass({colorAttachments:[{view:t.createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,1]}]}),c=e.createBindGroup({layout:j.getBindGroupLayout(0),entries:[{binding:0,resource:p},{binding:1,resource:n.createView()}]});r.setPipeline(j),r.setBindGroup(0,c),r.draw(4),r.end(),e.queue.submit([o.finish()])}const Ae=10,Ce=.01,ke=1e6;async function Ue(){const n=e.createCommandEncoder();n.copyBufferToBuffer(B,0,C,0,B.size),e.queue.submit([n.finish()]),await C.mapAsync(GPUMapMode.READ);const t=new Int32Array(C.getMappedRange());let o=0;for(let r=0;r<9;r++){const c=t[r]/ke;o+=Math.abs(c),E[r]-=Ce*c}C.unmap(),e.queue.writeTexture({texture:h},E,{bytesPerRow:3*4},[3,3]),e.queue.writeBuffer(B,0,new Float32Array(9)),He.textContent=o.toExponential(3)}let Ee=0,W=0,z=0,ae=performance.now(),ue=performance.now();function Oe(){++W&&performance.now()-ae>=1e3&&(Xe.textContent=String(W),W=0,ae=performance.now())}function We(){++z&&performance.now()-ue>=1e3&&(Je.textContent=String(z),z=0,ue=performance.now())}function ce(){e.queue.copyExternalImageToTexture({source:y},{texture:P},i),Oe(),e.queue.copyExternalImageToTexture({source:y},{texture:x[0]},i);for(let r=1;r<M;r++)oe(x[r-1],x[r]),oe(w[r-1],w[r]);for(let r=M-1;r>=0;r--){const c=e.createCommandEncoder(),m=c.beginComputePass();r<M-1?(m.setPipeline(H),m.setBindGroup(0,be(_[r+1],_[r])),m.dispatchWorkgroups(Math.ceil(d[r][0]/8),Math.ceil(d[r][1]/8))):e.queue.writeTexture({texture:_[r]},new Float32Array(d[r][0]*d[r][1]*4).fill(0),{bytesPerRow:d[r][0]*16},d[r]),m.setPipeline(X),m.setBindGroup(0,ye(w[r],x[r],_[r])),m.dispatchWorkgroups(Math.ceil(d[r][0]/8),Math.ceil(d[r][1]/8)),m.end(),e.queue.submit([c.finish()])}{const r=e.createCommandEncoder(),c=r.beginComputePass();c.setPipeline(Z),c.setBindGroup(0,we(N,_[0],I)),c.dispatchWorkgroups(Math.ceil(i[0]/8),Math.ceil(i[1]/8)),c.end(),e.queue.submit([r.finish()])}{const r=e.createCommandEncoder(),c=r.beginComputePass();c.setPipeline($),c.setBindGroup(0,_e(V,I)),c.dispatchWorkgroups(Math.ceil(i[0]/8),Math.ceil(i[1]/8)),c.end(),e.queue.submit([r.finish()])}e.queue.writeBuffer(re,0,ie()),e.queue.writeBuffer(ne,0,ie()),e.queue.writeBuffer(te,0,Ve());const n=e.createCommandEncoder(),t=n.beginComputePass();g(t,A,he),g(t,A,Be),g(t,b,Ge),g(t,b,Se),g(t,J,Te),g(t,Q,Le),g(t,b,Me),t.end();const o=n.beginRenderPass({colorAttachments:[{view:T.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,1]}]});o.setPipeline(ee),o.setBindGroup(0,Pe()),o.draw(4),o.end(),e.queue.submit([n.finish()]),We();{const r=e.createCommandEncoder();r.copyTextureToTexture({texture:V},{texture:N},i),e.queue.submit([r.finish()])}++Ee%Ae===0&&Ue().catch(console.error),[x,w]=[w,x],requestAnimationFrame(ce)}requestAnimationFrame(ce)})().catch(console.error);
