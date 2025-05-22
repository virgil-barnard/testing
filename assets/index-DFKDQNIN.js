(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))x(s);new MutationObserver(s=>{for(const o of s)if(o.type==="childList")for(const u of o.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&x(u)}).observe(document,{childList:!0,subtree:!0});function L(s){const o={};return s.integrity&&(o.integrity=s.integrity),s.referrerPolicy&&(o.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?o.credentials="include":s.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function x(s){if(s.ep)return;s.ep=!0;const o=L(s);fetch(s.href,o)}})();const Xe=`@group(0) @binding(0) var samplerLinear : sampler;\r
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
`,Je=`/* cnn_segmentation.wgsl  – 3×3 conv on luminance, sigmoid, 4-nbr smooth\r
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
`,Qe=`struct AugParams {        // offset.x, offset.y, scale.x, scale.y\r
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
`,Ze=`// Very small affine warp: inverse crop from view-B → view-A space.\r
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
`,$e=`/* loss.wgsl  – AugCo gradient, integer-atomics version  */\r
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
`,er=`@group(0) @binding(0) var prevTex : texture_2d<f32>;\r
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
`,rr=`@group(0) @binding(0) var srcFlow : texture_2d<f32>;\r
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
`,nr=`/* Show webcam + green-tinted alpha mask (probability).  */\r
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
}`,tr=`// Warp previous mask by dense optical flow\r
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
`,ir=`// Compute L2 error between current & warped‐previous masks\r
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
`,or=`// normalise.wgsl  – per-channel mean-std normalisation\r
\r
struct Stats {\r
    mean : vec3<f32>,\r
    invStd : vec3<f32>,   // 1 / σ  (pre-computed on CPU each frame)\r
};\r
\r
@group(0) @binding(0) var<uniform> stats : Stats;\r
@group(0) @binding(1) var samp        : sampler;\r
@group(0) @binding(2) var srcTex      : texture_2d<f32>;\r
@group(0) @binding(3) var dstTex      : texture_storage_2d<rgba8unorm, write>;\r
\r
@compute @workgroup_size(8, 8)\r
fn main(@builtin(global_invocation_id) gid : vec3<u32>)\r
{\r
    let dims = textureDimensions(dstTex);\r
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }\r
\r
    /* -- fetch + normalise -------------------------------------------------- */\r
    let rgb  = textureSampleLevel(srcTex, samp,\r
                                  (vec2<f32>(gid.xy)+0.5) / vec2<f32>(dims), 0).rgb;\r
    let z    = (rgb - stats.mean) * stats.invStd;     // zero-mean, unit-var\r
    let gain = 0.25;                                  // tame the range\r
    let out  = clamp(z * gain, vec3<f32>(0.0), vec3<f32>(1.0));\r
\r
    textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(out, 1.0));\r
}\r
`,T=document.getElementById("canvas"),v=document.getElementById("video"),ar=document.getElementById("grad-l1"),ur=document.getElementById("video-fps"),sr=document.getElementById("inference-fps");(async()=>{const O=await navigator.gpu.requestAdapter();if(!O){alert("WebGPU not supported");return}const e=await O.requestDevice(),L=T.getContext("webgpu"),x=navigator.gpu.getPreferredCanvasFormat();L.configure({device:e,format:x});const s=await navigator.mediaDevices.getUserMedia({video:!0});v.srcObject=s,await new Promise(n=>v.onloadedmetadata=n),T.width=v.videoWidth,T.height=v.videoHeight;const o=[T.width,T.height],u=GPUTextureUsage.TEXTURE_BINDING,g=GPUTextureUsage.STORAGE_BINDING,M=GPUTextureUsage.RENDER_ATTACHMENT,b=GPUTextureUsage.COPY_DST,ve=GPUTextureUsage.COPY_SRC,d=(n,i="rgba8unorm",a=o)=>e.createTexture({size:a,format:i,usage:n}),p=e.createSampler({magFilter:"linear",minFilter:"linear"}),C=3,l=Array.from({length:C},(n,i)=>[Math.max(1,o[0]>>i),Math.max(1,o[1]>>i)]);let _=l.map(n=>d(u|M|b,"bgra8unorm",n)),m=l.map(n=>d(u|M|b,"bgra8unorm",n)),P=l.map(n=>d(u|g|b,"rgba16float",n));const y=d(u|b|M);d(u|g);const z=d(u|g),I=d(u|g),N=d(u|g|M),K=d(u|g),Y=d(u|g),V=d(u|g|ve,"rgba8unorm",o),H=d(u|g,"rgba8unorm"),j=d(u|b,"rgba8unorm",o);let W=new Float32Array(9).fill(.1);const h=e.createTexture({size:[3,3],format:"r32float",usage:u|b});e.queue.writeTexture({texture:h},W,{bytesPerRow:3*4},[3,3]);const X=e.createShaderModule({code:Xe}),xe=e.createShaderModule({code:Je}),be=e.createShaderModule({code:Qe}),ye=e.createShaderModule({code:Ze}),we=e.createShaderModule({code:$e}),_e=e.createShaderModule({code:er}),Pe=e.createShaderModule({code:rr}),J=e.createShaderModule({code:nr}),he=e.createShaderModule({code:tr}),Be=e.createShaderModule({code:ir}),Se=e.createShaderModule({code:or}),Q=e.createRenderPipeline({layout:"auto",vertex:{module:X,entryPoint:"vs_main"},fragment:{module:X,entryPoint:"fs_main",targets:[{format:x}]},primitive:{topology:"triangle-strip"}}),Z=e.createComputePipeline({layout:"auto",compute:{module:Pe,entryPoint:"main"}}),$=e.createComputePipeline({layout:"auto",compute:{module:_e,entryPoint:"main"}}),w=e.createComputePipeline({layout:"auto",compute:{module:xe,entryPoint:"main"}}),A=e.createComputePipeline({layout:"auto",compute:{module:be,entryPoint:"main"}}),ee=e.createComputePipeline({layout:"auto",compute:{module:ye,entryPoint:"main"}}),re=e.createComputePipeline({layout:"auto",compute:{module:we,entryPoint:"main"}}),ne=e.createComputePipeline({layout:"auto",compute:{module:he,entryPoint:"main"}}),te=e.createComputePipeline({layout:"auto",compute:{module:Be,entryPoint:"main"}}),ie=e.createRenderPipeline({layout:"auto",vertex:{module:J,entryPoint:"vs_main"},fragment:{module:J,entryPoint:"fs_main",targets:[{format:x}]},primitive:{topology:"triangle-strip"}}),oe=e.createComputePipeline({layout:"auto",compute:{module:Se,entryPoint:"main"}}),B=e.createBuffer({size:9*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),U=e.createBuffer({size:9*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),D=(n=32)=>e.createBuffer({size:n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),ae=D(),ue=D(),se=D(16),S=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});new Float32Array(S.getMappedRange()).set(o),S.unmap();const ce=e.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});function Ge(n,i){return e.createBindGroup({layout:Z.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:p},{binding:2,resource:i.createView()}]})}function Te(n,i,a){return e.createBindGroup({layout:$.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:i.createView()},{binding:2,resource:a.createView()}]})}function Le(n,i,a){return e.createBindGroup({layout:ne.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:i.createView()},{binding:2,resource:a.createView()},{binding:3,resource:p}]})}function Me(n,i){return e.createBindGroup({layout:te.getBindGroupLayout(0),entries:[{binding:0,resource:n.createView()},{binding:1,resource:i.createView()},{binding:2,resource:{buffer:B}}]})}function Ce(){return e.createBindGroup({layout:ie.getBindGroupLayout(0),entries:[{binding:0,resource:p},{binding:1,resource:y.createView()},{binding:2,resource:V.createView()}]})}const Ve=e.createBindGroup({layout:A.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ae}},{binding:1,resource:p},{binding:2,resource:y.createView()},{binding:3,resource:z.createView()}]}),Ae=e.createBindGroup({layout:A.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ue}},{binding:1,resource:p},{binding:2,resource:y.createView()},{binding:3,resource:I.createView()}]}),Ue=e.createBindGroup({layout:w.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:S}},{binding:1,resource:z.createView()},{binding:2,resource:h.createView()},{binding:3,resource:N.createView()}]}),ke=e.createBindGroup({layout:w.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:S}},{binding:1,resource:I.createView()},{binding:2,resource:h.createView()},{binding:3,resource:K.createView()}]}),Ee=e.createBindGroup({layout:ee.getBindGroupLayout(0),entries:[{binding:0,resource:K.createView()},{binding:1,resource:Y.createView()},{binding:2,resource:{buffer:se}},{binding:3,resource:p}]}),Oe=e.createBindGroup({layout:re.getBindGroupLayout(0),entries:[{binding:0,resource:N.createView()},{binding:1,resource:Y.createView()},{binding:2,resource:z.createView()},{binding:3,resource:{buffer:B}}]}),ze=e.createBindGroup({layout:w.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:S}},{binding:1,resource:y.createView()},{binding:2,resource:h.createView()},{binding:3,resource:V.createView()}]}),We=e.createBindGroup({layout:oe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ce}},{binding:1,resource:p},{binding:2,resource:y.createView()},{binding:3,resource:m[0].createView()}]});function de(){const n=.6+Math.random()*.3,i=Math.random()*(1-n),a=Math.random()*(1-n),r=()=>.1+Math.random()*.8;return new Float32Array([i,a,n,n,r(),r(),r(),0])}function De(){return new Float32Array([(Math.random()-.5)*.2,(Math.random()-.5)*.2,1,1])}function f(n,i,a){n.setPipeline(i),n.setBindGroup(0,a),n.dispatchWorkgroups(Math.ceil(o[0]/8),Math.ceil(o[1]/8))}function le(n,i){const a=e.createCommandEncoder(),r=a.beginRenderPass({colorAttachments:[{view:i.createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,1]}]}),t=e.createBindGroup({layout:Q.getBindGroupLayout(0),entries:[{binding:0,resource:p},{binding:1,resource:n.createView()}]});r.setPipeline(Q),r.setBindGroup(0,t),r.draw(4),r.end(),e.queue.submit([a.finish()])}const k=32,E=24,ge=new OffscreenCanvas(k,E).getContext("2d"),F=new Float32Array(8);function Fe(){ge.drawImage(v,0,0,k,E);const n=ge.getImageData(0,0,k,E).data,i=[0,0,0],a=[0,0,0],r=k*E;for(let t=0;t<n.length;t+=4)for(let c=0;c<3;c++){const G=n[t+c]/255;i[c]+=G,a[c]+=G*G}for(let t=0;t<3;t++){const c=i[t]/r,G=a[t]/r-c*c,je=1/Math.sqrt(G+1e-6);F[t]=c,F[4+t]=je}e.queue.writeBuffer(ce,0,F)}const Re=10,qe=.01,Ie=1e6;async function Ne(){const n=e.createCommandEncoder();n.copyBufferToBuffer(B,0,U,0,B.size),e.queue.submit([n.finish()]),await U.mapAsync(GPUMapMode.READ);const i=new Int32Array(U.getMappedRange());let a=0;for(let r=0;r<9;r++){const t=i[r]/Ie;a+=Math.abs(t),W[r]-=qe*t}U.unmap(),e.queue.writeTexture({texture:h},W,{bytesPerRow:3*4},[3,3]),e.queue.writeBuffer(B,0,new Float32Array(9)),ar.textContent=a.toExponential(3)}let Ke=0,R=0,q=0,pe=performance.now(),me=performance.now();function Ye(){++R&&performance.now()-pe>=1e3&&(ur.textContent=String(R),R=0,pe=performance.now())}function He(){++q&&performance.now()-me>=1e3&&(sr.textContent=String(q),q=0,me=performance.now())}function fe(){e.queue.copyExternalImageToTexture({source:v},{texture:y},o),Ye(),Fe();{const r=e.createCommandEncoder(),t=r.beginComputePass();t.setPipeline(oe),t.setBindGroup(0,We),t.dispatchWorkgroups(Math.ceil(o[0]/8),Math.ceil(o[1]/8)),t.end(),e.queue.submit([r.finish()])}e.queue.copyExternalImageToTexture({source:v},{texture:m[0]},o);for(let r=1;r<C;r++)le(m[r-1],m[r]),le(_[r-1],_[r]);for(let r=C-1;r>=0;r--){const t=e.createCommandEncoder(),c=t.beginComputePass();r<C-1?(c.setPipeline(Z),c.setBindGroup(0,Ge(P[r+1],P[r])),c.dispatchWorkgroups(Math.ceil(l[r][0]/8),Math.ceil(l[r][1]/8))):e.queue.writeTexture({texture:P[r]},new Float32Array(l[r][0]*l[r][1]*4).fill(0),{bytesPerRow:l[r][0]*16},l[r]),c.setPipeline($),c.setBindGroup(0,Te(_[r],m[r],P[r])),c.dispatchWorkgroups(Math.ceil(l[r][0]/8),Math.ceil(l[r][1]/8)),c.end(),e.queue.submit([t.finish()])}{const r=e.createCommandEncoder(),t=r.beginComputePass();t.setPipeline(ne),t.setBindGroup(0,Le(j,P[0],H)),t.dispatchWorkgroups(Math.ceil(o[0]/8),Math.ceil(o[1]/8)),t.end(),e.queue.submit([r.finish()])}{const r=e.createCommandEncoder(),t=r.beginComputePass();t.setPipeline(te),t.setBindGroup(0,Me(V,H)),t.dispatchWorkgroups(Math.ceil(o[0]/8),Math.ceil(o[1]/8)),t.end(),e.queue.submit([r.finish()])}e.queue.writeBuffer(ae,0,de()),e.queue.writeBuffer(ue,0,de()),e.queue.writeBuffer(se,0,De());const n=e.createCommandEncoder(),i=n.beginComputePass();f(i,A,Ve),f(i,A,Ae),f(i,w,Ue),f(i,w,ke),f(i,ee,Ee),f(i,re,Oe),f(i,w,ze),i.end();const a=n.beginRenderPass({colorAttachments:[{view:L.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,1]}]});a.setPipeline(ie),a.setBindGroup(0,Ce()),a.draw(4),a.end(),e.queue.submit([n.finish()]),He();{const r=e.createCommandEncoder();r.copyTextureToTexture({texture:V},{texture:j},o),e.queue.submit([r.finish()])}++Ke%Re===0&&Ne().catch(console.error),[m,_]=[_,m],requestAnimationFrame(fe)}requestAnimationFrame(fe)})().catch(console.error);
