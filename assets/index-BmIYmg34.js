(function(){const d=document.createElement("link").relList;if(d&&d.supports&&d.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))s(r);new MutationObserver(r=>{for(const n of r)if(n.type==="childList")for(const a of n.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&s(a)}).observe(document,{childList:!0,subtree:!0});function e(r){const n={};return r.integrity&&(n.integrity=r.integrity),r.referrerPolicy&&(n.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?n.credentials="include":r.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function s(r){if(r.ep)return;r.ep=!0;const n=e(r);fetch(r.href,n)}})();const w=`@group(0) @binding(0) var samplerLinear : sampler;\r
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
`,S=`/* cnn_segmentation.wgsl  – 3×3 conv on luminance, sigmoid, 4-nbr smooth\r
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
`,h=`struct AugParams {        // offset.x, offset.y, scale.x, scale.y\r
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
`,T=`// Very small affine warp: inverse crop from view-B → view-A space.\r
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
`,L=`/* loss.wgsl  – AugCo gradient, integer-atomics version  */\r
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
`,P=`@group(0) @binding(0) var prevTex : texture_2d<f32>;\r
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
`,M=`@group(0) @binding(0) var srcFlow : texture_2d<f32>;\r
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
`,k=`/* Show webcam + green-tinted alpha mask (probability).  */\r
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
}`,z=`// Warp previous mask by dense optical flow\r
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
`,C=`// Compute L2 error between current & warped‐previous masks\r
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
`,v=document.getElementById("canvas"),m=document.getElementById("video");document.getElementById("grad-l1");document.getElementById("video-fps");document.getElementById("inference-fps");device.createShaderModule({code:w});device.createShaderModule({code:S});device.createShaderModule({code:h});device.createShaderModule({code:T});device.createShaderModule({code:L});device.createShaderModule({code:P});device.createShaderModule({code:M});device.createShaderModule({code:k});device.createShaderModule({code:z});device.createShaderModule({code:C});(async function(){const d=await navigator.gpu.requestAdapter();if(!d){alert("WebGPU not supported");return}const e=await d.requestDevice(),s=v.getContext("webgpu"),r=navigator.gpu.getPreferredCanvasFormat();s.configure({device:e,format:r});const n=await navigator.mediaDevices.getUserMedia({video:!0});m.srcObject=n,await new Promise(t=>m.onloadedmetadata=t),v.width=m.videoWidth,v.height=m.videoHeight;const a=[v.width,v.height],i=GPUTextureUsage.TEXTURE_BINDING,c=GPUTextureUsage.STORAGE_BINDING,p=GPUTextureUsage.RENDER_ATTACHMENT,l=GPUTextureUsage.COPY_DST,A=GPUTextureUsage.COPY_SRC,o=(t,u="rgba8unorm",N=a)=>e.createTexture({size:N,format:u,usage:t}),W=e.createSampler({minFilter:"linear",magFilter:"linear"}),g=Array.from({length:3},(t,u)=>[Math.max(1,a[0]>>u),Math.max(1,a[1]>>u)]);g.map(t=>o(i|p|l,"bgra8unorm",t)),g.map(t=>o(i|p|l,"bgra8unorm",t)),g.map(t=>o(i|c|l,"rgba16float",t));const f=o(i|l|p);o(i|c),o(i|c),o(i|c|p),o(i|c),o(i|c);const O=o(i|c|A,"rgba8unorm",a);let U=new Float32Array(9).fill(.1);const D=e.createTexture({size:[3,3],format:"r32float",usage:i|l});e.queue.writeTexture({texture:D},U,{bytesPerRow:3*4},[3,3]);const x=e.createShaderModule({code:w}),E=e.createShaderModule({code:S}),V=e.createShaderModule({code:h}),B=e.createShaderModule({code:T}),I=e.createShaderModule({code:L}),F=e.createShaderModule({code:P}),R=e.createShaderModule({code:M}),y=e.createShaderModule({code:k}),K=e.createShaderModule({code:z}),q=e.createShaderModule({code:C});e.createRenderPipeline({layout:"auto",vertex:{module:x,entryPoint:"vs_main"},fragment:{module:x,entryPoint:"fs_main",targets:[{format:r}]},primitive:{topology:"triangle-strip"}}),e.createComputePipeline({layout:"auto",compute:{module:R,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:F,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:E,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:V,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:B,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:I,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:K,entryPoint:"main"}}),e.createComputePipeline({layout:"auto",compute:{module:q,entryPoint:"main"}});const b=e.createRenderPipeline({layout:"auto",vertex:{module:y,entryPoint:"vs_main"},fragment:{module:y,entryPoint:"fs_main",targets:[{format:r}]},primitive:{topology:"triangle-strip"}});function _(){e.queue.copyExternalImageToTexture({source:m},{texture:f},a);const t=e.createCommandEncoder(),u=t.beginRenderPass({colorAttachments:[{view:s.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,1]}]});u.setPipeline(b),u.setBindGroup(0,e.createBindGroup({layout:b.getBindGroupLayout(0),entries:[{binding:0,resource:W},{binding:1,resource:f.createView()},{binding:2,resource:O.createView()}]})),u.draw(4),u.end(),e.queue.submit([t.finish()]),requestAnimationFrame(_)}requestAnimationFrame(_)})().catch(console.error);
