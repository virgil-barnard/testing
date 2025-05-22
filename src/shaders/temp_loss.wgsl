// Compute L2 error between current & warped‐previous masks
const SCALE_F : f32 = 1e6;  // if you want fixed‐point

struct Grad { vals: array<atomic<i32>, 1>, }; // one global accumulator

@group(0) @binding(0) var currMask : texture_2d<f32>;
@group(0) @binding(1) var prevWarp : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> grad  : Grad;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(currMask);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uvi = vec2<i32>(gid.xy);
  let c   = textureLoad(currMask, uvi, 0).r;
  let p   = textureLoad(prevWarp, uvi, 0).r;
  let err = c - p;               // dL/dc for ½‖c−p‖²
  let gi  = i32(round(err * SCALE_F));
  atomicAdd(&grad.vals[0], gi);
}
