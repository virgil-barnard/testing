/* ────────────────────────────────────────────────────────────
   main.ts – webcam ▶ augment ▶ segment ▶ self-sup. loss ▸ SGD
   prints grad-L1 every TRAIN_INTERVAL frames
──────────────────────────────────────────────────────────── */

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const video  = document.getElementById('video') as HTMLVideoElement;
const info   = document.getElementById('info');

/* ---------- GPU setup ---------- */
const adapter  = await navigator.gpu.requestAdapter();
const device   = await adapter!.requestDevice();
const context  = canvas.getContext('webgpu')!;

const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });



/* ---------- webcam initialisation ---------- */
await (async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(r => (video.onloadedmetadata = r));
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
})();

const wh = [canvas.width, canvas.height] as const;

/* ---------- helpers ---------- */
const TEXBIN = GPUTextureUsage.TEXTURE_BINDING;
const STOR   = GPUTextureUsage.STORAGE_BINDING;
const REND   = GPUTextureUsage.RENDER_ATTACHMENT;
const COPY   = GPUTextureUsage.COPY_DST;
const COPY_SRC = GPUTextureUsage.COPY_SRC;

const makeTex = (usage: number, fmt='rgba8unorm', size=wh) =>
  device.createTexture({ size, format: fmt as GPUTextureFormat, usage });

const sampler = device.createSampler({ magFilter:'linear', minFilter:'linear' });

/* ---------- multi-scale pyramids setup ---------- */
const levels = 3;
const pyrDims = Array.from({ length: levels }, (_, i) => [Math.max(1, wh[0] >> i), Math.max(1, wh[1] >> i)]);
let prevPyramid = pyrDims.map(dim => makeTex(TEXBIN | REND | COPY, 'bgra8unorm', dim));
let currPyramid = pyrDims.map(dim => makeTex(TEXBIN | REND | COPY, 'bgra8unorm', dim));
let flowPyramid = pyrDims.map(dim => makeTex(TEXBIN | STOR | COPY, 'rgba16float', dim));

/* ---------- textures ---------- */
const webcamTex  = makeTex(TEXBIN | COPY | REND);
const augATex    = makeTex(TEXBIN | STOR);
const augBTex    = makeTex(TEXBIN | STOR);
const segATex    = makeTex(TEXBIN | STOR | REND);  // mask for view A
const segBTex    = makeTex(TEXBIN | STOR);         // mask for view B
const warpBTex   = makeTex(TEXBIN | STOR);         // B-mask warped → A coords
const fullMaskTex = makeTex(TEXBIN | STOR | COPY_SRC, 'rgba8unorm', wh);

/* ---------- tiny 3×3 kernel (weights) ---------- */
let weights = new Float32Array(9).fill(0.1);
const weightsTex = device.createTexture({
  size: [3,3], format:'r32float',
  usage: TEXBIN | COPY
});
device.queue.writeTexture({ texture:weightsTex },
                          weights, { bytesPerRow:3*4 }, [3,3]);

/* -------- shader sources bundled at build time -------- */
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

/* ---------- pipelines ---------- */
const blurPipe    = device.createRenderPipeline({ layout:'auto', vertex:{ module:blurMod, entryPoint:'vs_main' }, fragment:{ module:blurMod, entryPoint:'fs_main', targets:[{ format }]}, primitive:{ topology:'triangle-strip' }});
const upscalePipe = device.createComputePipeline({ layout:'auto', compute:{ module: upscaleMod, entryPoint:'main' }});
const flowPipe    = device.createComputePipeline({ layout:'auto', compute:{ module: flowMod, entryPoint:'main' }});
const cnnPipe   = device.createComputePipeline({ layout:'auto', compute:{module:cnnMod, entryPoint:'main'}});
const augPipe   = device.createComputePipeline({ layout:'auto', compute:{module:augMod, entryPoint:'main'}});
const warpPipe  = device.createComputePipeline({ layout:'auto', compute:{module:warpMod,entryPoint:'main'}});
const lossPipe  = device.createComputePipeline({ layout:'auto', compute:{module:lossMod,entryPoint:'main'}});

/* ---------- helpers ---------- */
/* gradient buffer (atomic adds in loss shader) */
const gradBuf         = device.createBuffer({ size: 9*4,
                         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const gradReadbackBuf = device.createBuffer({ size: 9*4,
                         usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
/* ---------- uniform / param buffers ---------- */
function paramBuf(bytes=32){ return device.createBuffer({size:bytes,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}); }
const augAParams = paramBuf();
const augBParams = paramBuf();
const warpParams = paramBuf(16);

const whBuf = device.createBuffer({
  size:8, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST, mappedAtCreation:true
});
new Float32Array(whBuf.getMappedRange()).set(wh);
whBuf.unmap();

function upscaleBG(src: GPUTexture, dst: GPUTexture) {
  return device.createBindGroup({ layout: upscalePipe.getBindGroupLayout(0), entries:[
    { binding:0, resource:src.createView() },
    { binding:1, resource:sampler },
    { binding:2, resource:dst.createView() },
  ]});
}

function flowBG(prevTex:GPUTexture,currTex:GPUTexture,prevFlow:GPUTexture,dstFlow:GPUTexture){
  return device.createBindGroup({ layout: flowPipe.getBindGroupLayout(0), entries:[
    { binding:0, resource:prevTex.createView() },
    { binding:1, resource:currTex.createView() },
    { binding:2, resource:dstFlow.createView() },
  ]});
}

function downscale(src:GPUTexture, dst:GPUTexture){
  const enc=device.createCommandEncoder();
  const pass=enc.beginRenderPass({ colorAttachments:[{view:dst.createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
  const bg=device.createBindGroup({layout:blurPipe.getBindGroupLayout(0),entries:[{binding:0,resource:sampler},{binding:1,resource:src.createView()}]});
  pass.setPipeline(blurPipe); pass.setBindGroup(0,bg); pass.draw(4); pass.end();
  device.queue.submit([enc.finish()]);
}

const overlayMod  = await mod('/src/shaders/overlay.wgsl');
const overlayPipe = device.createRenderPipeline({
  layout : 'auto',
  vertex : { module: overlayMod, entryPoint: 'vs_main' },
  fragment:{ module: overlayMod, entryPoint: 'fs_main', targets:[{ format }]},
  primitive:{ topology:'triangle-strip' }
});

const warpPrevMaskTex = makeTex(TEXBIN|STOR, 'rgba8unorm');
const prevMaskTex = makeTex(TEXBIN | COPY, 'rgba8unorm', wh);
const tempWarpMod     = await mod('/src/shaders/temp_consistency.wgsl');
const tempWarpPipe    = device.createComputePipeline({ layout:'auto', compute:{ module:tempWarpMod,   entryPoint:'main' }});
const tempLossMod     = await mod('/src/shaders/temp_loss.wgsl');
const tempLossPipe    = device.createComputePipeline({ layout:'auto', compute:{ module: tempLossMod, entryPoint:'main' }});

function tempWarpBG(prevMask:GPUTexture, flow:GPUTexture, out:GPUTexture) {
  return device.createBindGroup({ layout: tempWarpPipe.getBindGroupLayout(0), entries:[
    { binding:0, resource:prevMask.createView() },
    { binding:1, resource:flow      .createView() },
    { binding:2, resource:out       .createView() },
    { binding:3, resource:sampler }
  ]});
}
function tempLossBG(curr:GPUTexture, prevWarp:GPUTexture, gradBuf:GPUBuffer) {
  return device.createBindGroup({ layout: tempLossPipe.getBindGroupLayout(0), entries:[
    { binding:0, resource: curr     .createView() },
    { binding:1, resource: prevWarp .createView() },
    { binding:2, resource:{ buffer: gradBuf } }
  ]});
}

function overlayBG() {
  return device.createBindGroup({
    layout : overlayPipe.getBindGroupLayout(0),
    entries:[
      { binding:0, resource:sampler },
      { binding:1, resource:webcamTex.createView() },
      { binding:2, resource:fullMaskTex.createView() }   // or emaTex if you use smoothing
]});
}

const augABG = device.createBindGroup({
  layout: augPipe.getBindGroupLayout(0),
  entries:[
    { binding:0, resource:{buffer:augAParams}},
    { binding:1, resource:sampler },
    { binding:2, resource: webcamTex.createView() },
    { binding:3, resource: augATex.createView() }
]});
const augBBG = device.createBindGroup({
  layout: augPipe.getBindGroupLayout(0),
  entries:[
    { binding:0, resource:{buffer:augBParams}},
    { binding:1, resource:sampler },
    { binding:2, resource: webcamTex.createView() },
    { binding:3, resource: augBTex.createView() }
]});
/* CNN BGs (view A & B) */
const cnnBG_A = device.createBindGroup({
  layout: cnnPipe.getBindGroupLayout(0),
  entries:[
    { binding:0, resource:{buffer:whBuf}},
    { binding:1, resource: augATex.createView() },
    { binding:2, resource: weightsTex.createView() },
    { binding:3, resource: segATex.createView() }
]});
const cnnBG_B = device.createBindGroup({
  layout: cnnPipe.getBindGroupLayout(0),
  entries:[
    { binding:0, resource:{buffer:whBuf}},
    { binding:1, resource: augBTex.createView() },
    { binding:2, resource: weightsTex.createView() },
    { binding:3, resource: segBTex.createView() }
]});
/* warp BG */
const warpBG = device.createBindGroup({
  layout: warpPipe.getBindGroupLayout(0),
  entries:[
    { binding:0, resource: segBTex.createView() },   // src
    { binding:1, resource: warpBTex.createView() },  // dst
    { binding:2, resource:{buffer:warpParams}},      // affine
    { binding:3, resource:sampler }
]});
/* loss BG */
const lossBG = device.createBindGroup({
  layout: lossPipe.getBindGroupLayout(0),
  entries:[
    { binding:0, resource: segATex.createView() },   // mask A
    { binding:1, resource: warpBTex.createView() },  // warped B
    { binding:2, resource: augATex.createView() },   // features
    { binding:3, resource:{buffer:gradBuf}}
]});

/* blur (render) BG – created each frame (needs latest segATex view) */
function blurBG(){
  return device.createBindGroup({
    layout: blurPipe.getBindGroupLayout(0),
    entries:[
      { binding:0, resource:sampler},
      { binding:1, resource: segATex.createView() }
  ]});
}

/* ---------- augmentation helpers ---------- */
function randAug(): Float32Array {
  // offScale [ox,oy,sx,sy]  colour [r,g,b,_]
  const scale = 0.6 + Math.random()*0.3;
  const ox = Math.random()*(1-scale);
  const oy = Math.random()*(1-scale);
  const jitter = () => 0.1 + Math.random()*0.8;
  return new Float32Array([ox,oy,scale,scale, jitter(),jitter(),jitter(),0]);
}

/* ---------- utility to dispatch compute ---------- */
function dispatch(passEnc: GPUComputePassEncoder,
                  pipe: GPUComputePipeline, bg: GPUBindGroup)
{
  passEnc.setPipeline(pipe);
  passEnc.setBindGroup(0,bg);
  passEnc.dispatchWorkgroups(Math.ceil(wh[0]/8), Math.ceil(wh[1]/8));
}

/* ---------- weights update (CPU SGD on 9 params) ---------- */
/* ---------- training parameters ---------- */
const TRAIN_INTERVAL = 10;  // frames
const LR             = 0.01;
/* constant used in WGSL */
const SCALE = 1e6;
function randWarp(): Float32Array {
  const maxOff = 0.2;               // up to ±20% of the image
  const ox = (Math.random() - 0.5) * maxOff;
  const oy = (Math.random() - 0.5) * maxOff;
  const sx = 1.0, sy = 1.0;         // keep scale = 1
  return new Float32Array([ox, oy, sx, sy]);
}
async function applySGD()
{
  /* copy gradBuf → gradReadbackBuf */
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(gradBuf,0, gradReadbackBuf,0, gradBuf.size);
  device.queue.submit([enc.finish()]);

  await gradReadbackBuf.mapAsync(GPUMapMode.READ);
  const int32 = new Int32Array(gradReadbackBuf.getMappedRange());

  let l1 = 0;
  for (let i = 0; i < 9; i++) {
    const g = int32[i] / SCALE;        // back to float
    l1 += Math.abs(g);
    weights[i] -= LR * g;              // SGD step
  }
  gradReadbackBuf.unmap();

  /* upload updated kernel to GPU texture */
  device.queue.writeTexture({texture:weightsTex},
                            weights, {bytesPerRow:3*4}, [3,3]);

  /* zero gradient buffer */
  device.queue.writeBuffer(gradBuf,0,new Float32Array(9).fill(0));

  //if (info) info.textContent = `grad L1: ${l1.toExponential(3)}`;
  gradL1Elem.textContent = l1.toExponential(3);
}

/* ---------- FPS ---------- */
let frameNum=0,lastVideoTime=performance.now(),videoFrames=0,lastInfTime=performance.now(),infFrames=0;
const videoFpsElem=document.getElementById('video-fps')!;
const infFpsElem=document.getElementById('inference-fps')!;

function updateVideoFPS(){if(++videoFrames&&performance.now()-lastVideoTime>=1000){videoFpsElem.textContent=`${videoFrames}`;videoFrames=0;lastVideoTime=performance.now();}}
function updateInferenceFPS(){if(++infFrames&&performance.now()-lastInfTime>=1000){infFpsElem.textContent=`${infFrames}`;infFrames=0;lastInfTime=performance.now();}}

const cnnBG_full = device.createBindGroup({
  layout: cnnPipe.getBindGroupLayout(0),
  entries: [
    { binding:0, resource:{buffer:whBuf}},          // same size
    { binding:1, resource: webcamTex.createView() },// *un-augmented* input
    { binding:2, resource: weightsTex.createView() },
    { binding:3, resource: fullMaskTex.createView() }
]});


/* ---------- main loop ---------- */
let gradL1Elem = document.getElementById('grad-l1')!;
function tick() {
  updateVideoFPS();

  device.queue.copyExternalImageToTexture(
    { source: video }, { texture: webcamTex }, wh
  );
  device.queue.copyExternalImageToTexture(
    { source: video }, { texture: currPyramid[0] }, wh
  );

  /* generate Gaussian pyramid */
  for (let i = 1; i < levels; i++) {
    downscale(currPyramid[i-1], currPyramid[i]);
    downscale(prevPyramid[i-1], prevPyramid[i]);
  }

  /* multi-scale optical flow (coarse-to-fine) */
  for (let i = levels - 1; i >= 0; i--) {
    const enc = device.createCommandEncoder();
    const c = enc.beginComputePass();

    if (i < levels - 1) {
      c.setPipeline(upscalePipe);
      c.setBindGroup(0, upscaleBG(flowPyramid[i+1], flowPyramid[i]));
      c.dispatchWorkgroups(Math.ceil(pyrDims[i][0]/8), Math.ceil(pyrDims[i][1]/8));
    } else {
      device.queue.writeTexture({texture: flowPyramid[i]}, 
        new Float32Array(pyrDims[i][0]*pyrDims[i][1]*4).fill(0), 
        {bytesPerRow: pyrDims[i][0]*16}, pyrDims[i]);
    }

    c.setPipeline(flowPipe);
    c.setBindGroup(0, flowBG(prevPyramid[i], currPyramid[i], flowPyramid[i], flowPyramid[i]));
    c.dispatchWorkgroups(Math.ceil(pyrDims[i][0]/8), Math.ceil(pyrDims[i][1]/8));
    c.end();
    device.queue.submit([enc.finish()]);
  }

  // ── Warp previous mask ──
  {
    const enc = device.createCommandEncoder();
    const c   = enc.beginComputePass();
    c.setPipeline(tempWarpPipe);
    c.setBindGroup(0, tempWarpBG(prevMaskTex, flowPyramid[0], warpPrevMaskTex));
    c.dispatchWorkgroups(
      Math.ceil(wh[0]/8),
      Math.ceil(wh[1]/8)
    );
    c.end();
    device.queue.submit([enc.finish()]);
  }

  // ── Temporal loss ──
  {
    const enc = device.createCommandEncoder();
    const c   = enc.beginComputePass();
    c.setPipeline(tempLossPipe);
    c.setBindGroup(0, tempLossBG(fullMaskTex, warpPrevMaskTex, gradBuf));
    c.dispatchWorkgroups(
      Math.ceil(wh[0]/8),
      Math.ceil(wh[1]/8)
    );
    c.end();
    device.queue.submit([enc.finish()]);
  }

  /* augmentation, segmentation, warp, loss, and inference passes (restored) */
  /* 2▸ randomise aug & warp params */
  device.queue.writeBuffer(augAParams, 0, randAug());
  device.queue.writeBuffer(augBParams, 0, randAug());
  // <<< replaced identity with random warp >>>
  device.queue.writeBuffer(warpParams, 0, randWarp());  
  const enc = device.createCommandEncoder();
  const c = enc.beginComputePass();
  dispatch(c, augPipe, augABG);      // augment view A
  dispatch(c, augPipe, augBBG);      // augment view B
  dispatch(c, cnnPipe, cnnBG_A);     // segment view A
  dispatch(c, cnnPipe, cnnBG_B);     // segment view B
  dispatch(c, warpPipe, warpBG);     // warp B→A
  dispatch(c, lossPipe, lossBG);     // compute loss and gradient
  dispatch(c, cnnPipe, cnnBG_full);  // full-frame inference mask
  c.end();

  /* render overlay pass (restored) */
  const rp = enc.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp :'clear',
      storeOp:'store',
      clearValue:[0,0,0,1]
    }]
  });
  rp.setPipeline(overlayPipe);
  rp.setBindGroup(0, overlayBG());
  rp.draw(4);
  rp.end();

  device.queue.submit([enc.finish()]);
  // copy fullMaskTex → prevMaskTex
  {
    const enc = device.createCommandEncoder();
    enc.copyTextureToTexture(
      { texture: fullMaskTex },
      { texture: prevMaskTex },
      wh
    );
    device.queue.submit([enc.finish()]);
  }
  updateInferenceFPS();

  if (++frameNum % TRAIN_INTERVAL === 0) applySGD().catch(console.error);

  [currPyramid, prevPyramid] = [prevPyramid, currPyramid];

  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
