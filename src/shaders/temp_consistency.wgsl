// Warp previous mask by dense optical flow
@group(0) @binding(0) var prevMask : texture_2d<f32>;
@group(0) @binding(1) var flowTex  : texture_2d<f32>;
@group(0) @binding(2) var outWarp  : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var samp     : sampler;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(outWarp);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv = vec2<f32>(gid.xy) / vec2<f32>(dims);
  // fetch flow (u,v) in pixel space
  let f  = textureLoad(flowTex, vec2<i32>(gid.xy), 0).xy;
  // adjust sampling coords
  let prevUV = uv + f / vec2<f32>(dims);
  let val    = textureSampleLevel(prevMask, samp, prevUV, 0.0).r;
  textureStore(outWarp, vec2<i32>(gid.xy), vec4<f32>(val, val, val, 1.0));
}
