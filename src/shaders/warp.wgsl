// Very small affine warp: inverse crop from view-B → view-A space.
struct Warp { offScale : vec4<f32>, };
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> mat : Warp;
@group(0) @binding(3) var samp : sampler;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>)
{
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv  = (vec2<f32>(gid.xy) / vec2<f32>(dims)) * mat.offScale.zw + mat.offScale.xy;
  let m   = textureSampleLevel(src, samp, uv, 0.0).r;
  textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(m, m, m, 1.0));
}
