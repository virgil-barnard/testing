/* loss.wgsl  – AugCo gradient, integer-atomics version  */
const SCALE : f32 = 1e6;              // converts float-grad ➞ fixed-point

struct Grad { vals : array<atomic<i32>, 9>, };

@group(0) @binding(0) var maskA  : texture_2d<f32>;
@group(0) @binding(1) var maskBW : texture_2d<f32>;     // B mask warped → A
@group(0) @binding(2) var feats  : texture_2d<f32>;      // augA intensities
@group(0) @binding(3) var<storage, read_write> grad : Grad;

fn σp(s: f32) -> f32 { return s * (1.0 - s); }          // sigmoid derivative

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>)
{
  let dims = textureDimensions(maskA);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv   = vec2<i32>(gid.xy);
  let a    = textureLoad(maskA , uv, 0).r;
  let b    = textureLoad(maskBW, uv, 0).r;
  let err  = a - b;                       // dL/ds  (½‖a−b‖²)

  var k = 0u;
  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      let coord = clamp(uv + vec2<i32>(dx,dy),
                        vec2<i32>(0), vec2<i32>(dims) - 1);
      let x   = textureLoad(feats, coord, 0).r;
      let g   = err * σp(a) * x;          // ∂L/∂w
      let gi  = i32(round(g * SCALE));    // fixed-point
      atomicAdd(&grad.vals[k], gi);
      k += 1u;
    }
  }
}
