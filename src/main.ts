/* ────────────────────────────────────────────────────────────────
   main.ts – webcam ▶ augment ▶ segment ▶ self-sup. loss ▸ SGD
   prints grad-L1 every TRAIN_INTERVAL frames
───────────────────────────────────────────────────────────────── */

// -- DOM elements (they already exist in index.html) ────────────
const canvas   = document.getElementById('canvas') as HTMLCanvasElement;
const video    = document.getElementById('video')  as HTMLVideoElement;
const gradL1El = document.getElementById('grad-l1')!;
const vFpsEl   = document.getElementById('video-fps')!;
const iFpsEl   = document.getElementById('inference-fps')!;

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

/* ─────────────────────── main IIFE ─────────────────────────── */
(async function main() {
  /* ---------- GPU setup ---------- */
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { alert('WebGPU not supported'); return; }
  const device  = await adapter.requestDevice();

  const ctx = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format });

  /* ---------- webcam ---------- */
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(r => (video.onloadedmetadata = r));
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  const wh = [canvas.width, canvas.height] as const;

  /* ---------- helpers ---------- */
  const TEX   = GPUTextureUsage.TEXTURE_BINDING;
  const STOR  = GPUTextureUsage.STORAGE_BINDING;
  const REND  = GPUTextureUsage.RENDER_ATTACHMENT;
  const COPY  = GPUTextureUsage.COPY_DST;
  const COPY_SRC = GPUTextureUsage.COPY_SRC;

  const makeTex = (usage: number, fmt = 'rgba8unorm', size = wh) =>
    device.createTexture({ size, format: fmt as GPUTextureFormat, usage });

  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  /* ---------- Gaussian pyramids ---------- */
  const levels   = 3;
  const pyrDims  = Array.from({ length: levels }, (_, i) =>
                     [Math.max(1, wh[0] >> i), Math.max(1, wh[1] >> i)]);
  let prevPyr = pyrDims.map(d => makeTex(TEX|REND|COPY, 'bgra8unorm', d));
  let currPyr = pyrDims.map(d => makeTex(TEX|REND|COPY, 'bgra8unorm', d));
  let flowPyr = pyrDims.map(d => makeTex(TEX|STOR|COPY, 'rgba16float', d));

  /* ---------- working textures ---------- */
  const webcamTex   = makeTex(TEX|COPY|REND);
  const augATex     = makeTex(TEX|STOR);
  const augBTex     = makeTex(TEX|STOR);
  const segATex     = makeTex(TEX|STOR|REND);
  const segBTex     = makeTex(TEX|STOR);
  const warpBTex    = makeTex(TEX|STOR);
  const fullMaskTex = makeTex(TEX|STOR|COPY_SRC, 'rgba8unorm', wh);

  /* ---------- tiny 3×3 weights texture ---------- */
  let weights = new Float32Array(9).fill(0.1);
  const weightsTex = device.createTexture({
    size : [3,3],
    format : 'r32float',
    usage : TEX|COPY
  });
  device.queue.writeTexture(
    { texture: weightsTex },
    weights, { bytesPerRow: 3*4 }, [3,3]
  );

  /* ---------- compile shader modules ---------- */
  const blurMod   = device.createShaderModule({ code: blurWGSL     });
  const cnnMod    = device.createShaderModule({ code: cnnWGSL      });
  const augMod    = device.createShaderModule({ code: augWGSL      });
  const warpMod   = device.createShaderModule({ code: warpWGSL     });
  const lossMod   = device.createShaderModule({ code: lossWGSL     });
  const flowMod   = device.createShaderModule({ code: flowWGSL     });
  const upMod     = device.createShaderModule({ code: upscaleWGSL  });
  const overlayMod= device.createShaderModule({ code: overlayWGSL  });
  const tWarpMod  = device.createShaderModule({ code: tmpWarpWGSL  });
  const tLossMod  = device.createShaderModule({ code: tmpLossWGSL  });

  /* ---------- pipelines ---------- */
  const blurPipe = device.createRenderPipeline({
    layout :'auto',
    vertex : { module: blurMod, entryPoint:'vs_main' },
    fragment:{ module: blurMod, entryPoint:'fs_main', targets:[{ format }]},
    primitive:{ topology:'triangle-strip' }
  });
  const upscalePipe = device.createComputePipeline({ layout:'auto',
        compute:{ module: upMod, entryPoint:'main' }});
  const flowPipe  = device.createComputePipeline({ layout:'auto',
        compute:{ module: flowMod, entryPoint:'main' }});
  const cnnPipe   = device.createComputePipeline({ layout:'auto',
        compute:{ module: cnnMod, entryPoint:'main' }});
  const augPipe   = device.createComputePipeline({ layout:'auto',
        compute:{ module: augMod, entryPoint:'main' }});
  const warpPipe  = device.createComputePipeline({ layout:'auto',
        compute:{ module: warpMod, entryPoint:'main' }});
  const lossPipe  = device.createComputePipeline({ layout:'auto',
        compute:{ module: lossMod, entryPoint:'main' }});
  const tWarpPipe = device.createComputePipeline({ layout:'auto',
        compute:{ module: tWarpMod, entryPoint:'main' }});
  const tLossPipe = device.createComputePipeline({ layout:'auto',
        compute:{ module: tLossMod, entryPoint:'main' }});
  const overlayPipe = device.createRenderPipeline({
    layout :'auto',
    vertex : { module: overlayMod, entryPoint:'vs_main' },
    fragment:{ module: overlayMod, entryPoint:'fs_main', targets:[{ format }]},
    primitive:{ topology:'triangle-strip' }
  });

  /* ---------- (Everything else from your original file) ----------
     – buffers, bind-groups, SGD helpers, FPS helpers, tick() loop …
     The logic is unchanged, so you can keep your existing code
     **starting right here**.  The only things we touched above were:
       • wrapping in an IIFE,
       • compiling shader strings → modules,
       • making sure pipelines are built afterwards.
  ----------------------------------------------------------------- */

  /* Example heartbeat: render the webcam with overlay to prove it works */
  function frame() {
    device.queue.copyExternalImageToTexture({ source: video },
                                            { texture: webcamTex }, wh);

    const enc = device.createCommandEncoder();
    const rp  = enc.beginRenderPass({
      colorAttachments:[{
        view: ctx.getCurrentTexture().createView(),
        loadOp :'clear',
        storeOp:'store',
        clearValue:[0,0,0,1]
      }]
    });
    rp.setPipeline(overlayPipe);
    rp.setBindGroup(0, device.createBindGroup({
      layout: overlayPipe.getBindGroupLayout(0),
      entries:[
        { binding:0, resource:sampler },
        { binding:1, resource:webcamTex.createView() },
        { binding:2, resource:fullMaskTex.createView() }
    ]}));
    rp.draw(4);
    rp.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})().catch(console.error);
