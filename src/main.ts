/* ────────────────────────────────────────────────────────────────
   main.ts – webcam ▶ augment ▶ segment ▶ self-sup. loss ▸ SGD
   prints grad-L1 every TRAIN_INTERVAL frames
───────────────────────────────────────────────────────────────── */

/* ------------ DOM handles ------------ */
const canvas      = document.getElementById('canvas') as HTMLCanvasElement;
const video       = document.getElementById('video')  as HTMLVideoElement;
const gradL1Elem  = document.getElementById('grad-l1')!;
const videoFpsElem= document.getElementById('video-fps')!;
const infFpsElem  = document.getElementById('inference-fps')!;

/* ------------ WGSL sources (bundled by Vite) ------------ */
import blurWGSL     from './shaders/blur.wgsl?raw';
import cnnWGSL      from './shaders/cnn_segmentation.wgsl?raw';
import augWGSL      from './shaders/augment.wgsl?raw';
import warpWGSL     from './shaders/warp.wgsl?raw';
import lossWGSL     from './shaders/loss.wgsl?raw';
import flowWGSL     from './shaders/flow.wgsl?raw';
import upscaleWGSL  from './shaders/upscale_flow.wgsl?raw';
import overlayWGSL  from './shaders/overlay.wgsl?raw';
import tmpWarpWGSL  from './shaders/temp_consistency.wgsl?raw';
import tmpLossWGSL  from './shaders/temp_loss.wgsl?raw';
import normWGSL     from './shaders/normalise.wgsl?raw';       /* ★ NEW */

/* ─────────────────────── main IIFE ─────────────────────────── */
(async () => {
/* ---------- GPU SET-UP ---------- */
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { alert('WebGPU not supported'); return; }
const device  = await adapter.requestDevice();

const ctx   = canvas.getContext('webgpu')!;
const format= navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format });

/* ---------- WEBCAM ---------- */
const stream = await navigator.mediaDevices.getUserMedia({ video:true });
video.srcObject = stream;
await new Promise(r => (video.onloadedmetadata = r));
canvas.width  = video.videoWidth;
canvas.height = video.videoHeight;
const wh = [canvas.width, canvas.height] as const;

/* ---------- SHORT-HAND CONSTANTS ---------- */
const TEX  = GPUTextureUsage.TEXTURE_BINDING;
const STOR = GPUTextureUsage.STORAGE_BINDING;
const REND = GPUTextureUsage.RENDER_ATTACHMENT;
const COPY = GPUTextureUsage.COPY_DST;
const COPY_SRC = GPUTextureUsage.COPY_SRC;

const makeTex = (usage:number, fmt='rgba8unorm', size=wh)=>
  device.createTexture({ size, format:fmt as GPUTextureFormat, usage });

const sampler = device.createSampler({ magFilter:'linear', minFilter:'linear' });

/* ---------- MULTI-SCALE PYRAMIDS ---------- */
const levels  = 3;
const pyrDims = Array.from({length:levels},(_,i)=>[
  Math.max(1, wh[0] >> i), Math.max(1, wh[1] >> i)
]);
let prevPyr = pyrDims.map((dim, i) =>
  i === 0
    // level-0: we write into it, so we need STORAGE_BINDING and rgba8unorm
    ? makeTex(TEX | REND | COPY | STOR, 'rgba8unorm', dim)
    // levels 1-n: unchanged
    : makeTex(TEX | REND | COPY,        'bgra8unorm', dim)
);
let currPyr = pyrDims.map((dim, i) =>
  i === 0
    // level-0: we write into it, so we need STORAGE_BINDING and rgba8unorm
    ? makeTex(TEX | REND | COPY | STOR, 'rgba8unorm', dim)
    // levels 1-n: unchanged
    : makeTex(TEX | REND | COPY,        'bgra8unorm', dim)
);
let flowPyr = pyrDims.map(d=>makeTex(TEX|STOR|COPY,'rgba16float',d));

/* ---------- WORKING TEXTURES ---------- */
const webcamTex   = makeTex(TEX|COPY|REND);
const normTex     = makeTex(TEX|STOR);
const augATex     = makeTex(TEX|STOR);
const augBTex     = makeTex(TEX|STOR);
const segATex     = makeTex(TEX|STOR|REND);
const segBTex     = makeTex(TEX|STOR);
const warpBTex    = makeTex(TEX|STOR);
const fullMaskTex = makeTex(TEX|STOR|COPY_SRC,'rgba8unorm',wh);
const warpPrevMaskTex = makeTex(TEX|STOR,'rgba8unorm');
const prevMaskTex     = makeTex(TEX|COPY,'rgba8unorm',wh);

/* ---------- WEIGHTS ---------- */
let weights = new Float32Array(9).fill(0.1);
const weightsTex = device.createTexture({
  size:[3,3], format:'r32float', usage:TEX|COPY
});
device.queue.writeTexture({texture:weightsTex},
  weights, {bytesPerRow:3*4}, [3,3]);

/* ---------- SHADER MODULES ---------- */
const blurMod   = device.createShaderModule({code:blurWGSL});
const cnnMod    = device.createShaderModule({code:cnnWGSL});
const augMod    = device.createShaderModule({code:augWGSL});
const warpMod   = device.createShaderModule({code:warpWGSL});
const lossMod   = device.createShaderModule({code:lossWGSL});
const flowMod   = device.createShaderModule({code:flowWGSL});
const upMod     = device.createShaderModule({code:upscaleWGSL});
const overlayMod= device.createShaderModule({code:overlayWGSL});
const tWarpMod  = device.createShaderModule({code:tmpWarpWGSL});
const tLossMod  = device.createShaderModule({code:tmpLossWGSL});
const normMod   = device.createShaderModule({ code:normWGSL});

/* ---------- PIPELINES ---------- */
const blurPipe    = device.createRenderPipeline({
  layout:'auto',
  vertex:{module:blurMod, entryPoint:'vs_main'},
  fragment:{module:blurMod, entryPoint:'fs_main', targets:[{format}]},
  primitive:{topology:'triangle-strip'}
});
const upscalePipe = device.createComputePipeline({layout:'auto',
  compute:{module:upMod, entryPoint:'main'}});
const flowPipe    = device.createComputePipeline({layout:'auto',
  compute:{module:flowMod, entryPoint:'main'}});
const cnnPipe     = device.createComputePipeline({layout:'auto',
  compute:{module:cnnMod, entryPoint:'main'}});
const augPipe     = device.createComputePipeline({layout:'auto',
  compute:{module:augMod, entryPoint:'main'}});
const warpPipe    = device.createComputePipeline({layout:'auto',
  compute:{module:warpMod, entryPoint:'main'}});
const lossPipe    = device.createComputePipeline({layout:'auto',
  compute:{module:lossMod, entryPoint:'main'}});
const tWarpPipe   = device.createComputePipeline({layout:'auto',
  compute:{module:tWarpMod, entryPoint:'main'}});
const tLossPipe   = device.createComputePipeline({layout:'auto',
  compute:{module:tLossMod, entryPoint:'main'}});
const overlayPipe = device.createRenderPipeline({
  layout:'auto',
  vertex:{module:overlayMod, entryPoint:'vs_main'},
  fragment:{module:overlayMod, entryPoint:'fs_main', targets:[{format}]},
  primitive:{topology:'triangle-strip'}
});
  const normPipe = device.createComputePipeline({                /* ★ NEW */
    layout : 'auto',
    compute: { module: normMod, entryPoint:'main' }
  });

/* ---------- BUFFERS ---------- */
const gradBuf = device.createBuffer({
  size:9*4, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST
});
const gradReadBuf = device.createBuffer({
  size:9*4, usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ
});
const paramBuf = (bytes=32)=>
  device.createBuffer({size:bytes, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
const augAParams = paramBuf();
const augBParams = paramBuf();
const warpParams = paramBuf(16);
const whBuf = device.createBuffer({
  size:8, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST, mappedAtCreation:true
});
new Float32Array(whBuf.getMappedRange()).set(wh); whBuf.unmap();
/* ---------- uniform buffer for <mean, invStd> ----------------- ★ NEW */
const normStatsBuf = device.createBuffer({
  size : 32,                     // two floats
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

/* ---------- BIND GROUP HELPERS ---------- */
function upscaleBG(src:GPUTexture,dst:GPUTexture){
  return device.createBindGroup({layout:upscalePipe.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:src.createView()},
      {binding:1,resource:sampler},
      {binding:2,resource:dst.createView()}
  ]});
}
function flowBG(prev:GPUTexture,curr:GPUTexture,dst:GPUTexture){
  return device.createBindGroup({layout:flowPipe.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:prev.createView()},
      {binding:1,resource:curr.createView()},
      {binding:2,resource:dst .createView()}
  ]});
}
function tempWarpBG(prevMask:GPUTexture,flow:GPUTexture,out:GPUTexture){
  return device.createBindGroup({layout:tWarpPipe.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:prevMask.createView()},
      {binding:1,resource:flow    .createView()},
      {binding:2,resource:out     .createView()},
      {binding:3,resource:sampler}
  ]});
}
function tempLossBG(curr:GPUTexture,prevW:GPUTexture){
  return device.createBindGroup({layout:tLossPipe.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:curr .createView()},
      {binding:1,resource:prevW.createView()},
      {binding:2,resource:{buffer:gradBuf}}
  ]});
}
function overlayBG(){
  return device.createBindGroup({layout:overlayPipe.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:sampler},
      {binding:1,resource:webcamTex.createView()},
      {binding:2,resource:fullMaskTex.createView()}
  ]});
}

/* ---------- AUG / CNN / WARP / LOSS BIND GROUPS ---------- */
const augABG = device.createBindGroup({layout:augPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:{buffer:augAParams}},
    {binding:1,resource:sampler},
    {binding:2,resource:webcamTex.createView()},
    {binding:3,resource:augATex .createView()}
]});
const augBBG = device.createBindGroup({layout:augPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:{buffer:augBParams}},
    {binding:1,resource:sampler},
    {binding:2,resource:webcamTex.createView()},
    {binding:3,resource:augBTex .createView()}
]});
const cnnBG_A = device.createBindGroup({layout:cnnPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:{buffer:whBuf}},
    {binding:1,resource:augATex.createView()},
    {binding:2,resource:weightsTex.createView()},
    {binding:3,resource:segATex.createView()}
]});
const cnnBG_B = device.createBindGroup({layout:cnnPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:{buffer:whBuf}},
    {binding:1,resource:augBTex.createView()},
    {binding:2,resource:weightsTex.createView()},
    {binding:3,resource:segBTex.createView()}
]});
const warpBG = device.createBindGroup({layout:warpPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:segBTex.createView()},
    {binding:1,resource:warpBTex.createView()},
    {binding:2,resource:{buffer:warpParams}},
    {binding:3,resource:sampler}
]});
const lossBG = device.createBindGroup({layout:lossPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:segATex.createView()},
    {binding:1,resource:warpBTex.createView()},
    {binding:2,resource:augATex.createView()},
    {binding:3,resource:{buffer:gradBuf}}
]});
const cnnBG_full = device.createBindGroup({layout:cnnPipe.getBindGroupLayout(0),
  entries:[
    {binding:0,resource:{buffer:whBuf}},
    {binding:1,resource:webcamTex.createView()},
    {binding:2,resource:weightsTex.createView()},
    {binding:3,resource:fullMaskTex.createView()}
]});
const normBG = device.createBindGroup({
  layout : normPipe.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: normStatsBuf } },      // Stats
    { binding: 1, resource: sampler },                   // samp
    { binding: 2, resource: webcamTex.createView() },    // src  ← unchanged
    { binding: 3, resource: currPyr[0].createView() }    // dst  ← new
]});
/* ---------- UTILS ---------- */
function randAug(){
  const scale = 0.6 + Math.random()*0.3;
  const ox = Math.random()*(1-scale);
  const oy = Math.random()*(1-scale);
  const jitter = ()=>0.1 + Math.random()*0.8;
  return new Float32Array([ox,oy,scale,scale,jitter(),jitter(),jitter(),0]);
}
function randWarp(){
  const maxOff = 0.2;
  return new Float32Array([
    (Math.random()-0.5)*maxOff,
    (Math.random()-0.5)*maxOff,
    1.0, 1.0
  ]);
}
function dispatch(cp:GPUComputePassEncoder,pipe:GPUComputePipeline,bg:GPUBindGroup){
  cp.setPipeline(pipe);
  cp.setBindGroup(0,bg);
  cp.dispatchWorkgroups(Math.ceil(wh[0]/8),Math.ceil(wh[1]/8));
}
function downscale(src:GPUTexture,dst:GPUTexture){
  const enc=device.createCommandEncoder();
  const rp = enc.beginRenderPass({
    colorAttachments:[{
      view:dst.createView(), loadOp:'clear',
      storeOp:'store', clearValue:[0,0,0,1]
    }]
  });
  const bg=device.createBindGroup({layout:blurPipe.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:sampler},
      {binding:1,resource:src.createView()}
  ]});
  rp.setPipeline(blurPipe);
  rp.setBindGroup(0,bg);
  rp.draw(4);
  rp.end();
  device.queue.submit([enc.finish()]);
}
const sampleW = 32, sampleH = 24;
const offCanvas = new OffscreenCanvas(sampleW, sampleH);
const offCtx    = offCanvas.getContext('2d')!;

const tmp = new Float32Array(8);                // = 32 bytes
function updateNormStats() {
  /* sample a smaller frame to keep CPU work reasonable */
  offCtx.drawImage(video, 0, 0, sampleW, sampleH);
  const px = offCtx.getImageData(0, 0, sampleW, sampleH).data;

  /* running sums per channel */
  const sum   = [0, 0, 0];
  const sumSq = [0, 0, 0];
  const N     = sampleW * sampleH;

  for (let i = 0; i < px.length; i += 4) {
    for (let c = 0; c < 3; c++) {              // R-G-B
      const v = px[i + c] / 255;               // → 0‒1
      sum  [c] += v;
      sumSq[c] += v * v;
    }
  }

  for (let c = 0; c < 3; c++) {
    const mean   = sum[c]   / N;
    const var_   = sumSq[c] / N - mean * mean;
    const invStd = 1 / Math.sqrt(var_ + 1e-6);

    tmp[c]     = mean;       // mean.xyz   → indices 0-2
    tmp[4 + c] = invStd;     // invStd.xyz → indices 4-6
  }
  /* tmp[3] and tmp[7] stay 0.0 (padding) */

  device.queue.writeBuffer(normStatsBuf, 0, tmp);
}
/* ---------- SGD APPLY ---------- */
const TRAIN_INTERVAL = 10;
const LR = 0.01;
const SCALE = 1e6;
async function applySGD(){
  const enc=device.createCommandEncoder();
  enc.copyBufferToBuffer(gradBuf,0,gradReadBuf,0,gradBuf.size);
  device.queue.submit([enc.finish()]);
  await gradReadBuf.mapAsync(GPUMapMode.READ);
  const ints = new Int32Array(gradReadBuf.getMappedRange());
  let l1=0;
  for(let i=0;i<9;i++){
    const g = ints[i]/SCALE;
    l1+=Math.abs(g);
    weights[i] -= LR*g;
  }
  gradReadBuf.unmap();
  device.queue.writeTexture({texture:weightsTex},
    weights,{bytesPerRow:3*4},[3,3]);
  device.queue.writeBuffer(gradBuf,0,new Float32Array(9));
  gradL1Elem.textContent = l1.toExponential(3);
}

/* ---------- FPS HELPERS ---------- */
let frameNum=0;
let vFrames=0,iFrames=0;
let lastVideoT=performance.now(), lastInfT=performance.now();
function bumpVideoFPS(){
  if(++vFrames && performance.now()-lastVideoT>=1000){
    videoFpsElem.textContent=String(vFrames);
    vFrames=0; lastVideoT=performance.now();
  }
}
function bumpInfFPS(){
  if(++iFrames && performance.now()-lastInfT>=1000){
    infFpsElem.textContent=String(iFrames);
    iFrames=0; lastInfT=performance.now();
  }
}

/* ---------- MAIN LOOP ---------- */
function tick(){
  /* copy camera frame */
  device.queue.copyExternalImageToTexture({source:video},
                                          {texture:webcamTex},wh);

  bumpVideoFPS();
  updateNormStats();
  {
    const enc = device.createCommandEncoder();
    const cp  = enc.beginComputePass();
    cp.setPipeline(normPipe);
    cp.setBindGroup(0, normBG);
    cp.dispatchWorkgroups(Math.ceil(wh[0] / 8), Math.ceil(wh[1] / 8));
    cp.end();
    device.queue.submit([enc.finish()]);
  }

  /* ── 2. copy the normalised texture into the current pyramid ─ */
  // {
  //   const enc = device.createCommandEncoder();          // NEW
  //   enc.copyTextureToTexture(                           // encoder-level API
  //     { texture: normTex },                             //   src
  //     { texture: currPyr[0] },                          //   dst
  //     wh                                                //   copy size
  //   );
  //   device.queue.submit([enc.finish()]);                // submit the copy
  // }

  /* ── 3. update webcam frame for the next iteration (unchanged) ─ */
  device.queue.copyExternalImageToTexture(
    { source: video },
    { texture: currPyr[0] },
    wh
  );

  /* build Gaussian pyramids */
  for(let i=1;i<levels;i++){
    downscale(currPyr[i-1],currPyr[i]);
    downscale(prevPyr[i-1],prevPyr[i]);
  }

  /* coarse-to-fine optical flow */
  for(let i=levels-1;i>=0;i--){
    const enc=device.createCommandEncoder();
    const c=enc.beginComputePass();
    if(i<levels-1){
      c.setPipeline(upscalePipe);
      c.setBindGroup(0,upscaleBG(flowPyr[i+1],flowPyr[i]));
      c.dispatchWorkgroups(Math.ceil(pyrDims[i][0]/8),Math.ceil(pyrDims[i][1]/8));
    }else{
      device.queue.writeTexture({texture:flowPyr[i]},
        new Float32Array(pyrDims[i][0]*pyrDims[i][1]*4).fill(0),
        {bytesPerRow:pyrDims[i][0]*16},pyrDims[i]);
    }
    c.setPipeline(flowPipe);
    c.setBindGroup(0,flowBG(prevPyr[i],currPyr[i],flowPyr[i]));
    c.dispatchWorkgroups(Math.ceil(pyrDims[i][0]/8),Math.ceil(pyrDims[i][1]/8));
    c.end(); device.queue.submit([enc.finish()]);
  }

  /* warp previous mask by flow */
  {
    const enc=device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(tWarpPipe);
    cp.setBindGroup(0,tempWarpBG(prevMaskTex,flowPyr[0],warpPrevMaskTex));
    cp.dispatchWorkgroups(Math.ceil(wh[0]/8),Math.ceil(wh[1]/8));
    cp.end(); device.queue.submit([enc.finish()]);
  }

  /* temporal loss */
  {
    const enc=device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(tLossPipe);
    cp.setBindGroup(0,tempLossBG(fullMaskTex,warpPrevMaskTex));
    cp.dispatchWorkgroups(Math.ceil(wh[0]/8),Math.ceil(wh[1]/8));
    cp.end(); device.queue.submit([enc.finish()]);
  }

  /* augment → segment → warp → loss → full-frame inference */
  device.queue.writeBuffer(augAParams,0,randAug());
  device.queue.writeBuffer(augBParams,0,randAug());
  device.queue.writeBuffer(warpParams,0,randWarp());

  const enc=device.createCommandEncoder();
  const cp = enc.beginComputePass();
  dispatch(cp,augPipe,augABG);
  dispatch(cp,augPipe,augBBG);
  dispatch(cp,cnnPipe,cnnBG_A);
  dispatch(cp,cnnPipe,cnnBG_B);
  dispatch(cp,warpPipe,warpBG);
  dispatch(cp,lossPipe,lossBG);
  dispatch(cp,cnnPipe,cnnBG_full);
  cp.end();

  /* overlay render pass */
  const rp = enc.beginRenderPass({
    colorAttachments:[{
      view:ctx.getCurrentTexture().createView(),
      loadOp:'clear', storeOp:'store', clearValue:[0,0,0,1]
    }]
  });
  rp.setPipeline(overlayPipe);
  rp.setBindGroup(0,overlayBG());
  rp.draw(4); rp.end();

  device.queue.submit([enc.finish()]);
  bumpInfFPS();

  /* copy full mask → prevMask for next frame */
  {
    const enc=device.createCommandEncoder();
    enc.copyTextureToTexture(
      {texture:fullMaskTex}, {texture:prevMaskTex}, wh
    );
    device.queue.submit([enc.finish()]);
  }

  if(++frameNum % TRAIN_INTERVAL===0) applySGD().catch(console.error);
  [currPyr,prevPyr] = [prevPyr,currPyr];
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
})().catch(console.error);
