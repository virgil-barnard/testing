struct AugParams {        // offset.x, offset.y, scale.x, scale.y
  offScale : vec4<f32>,   // colour jitter packed in .zw‐component of next vec4
  colour   : vec4<f32>,   // r,g,b,unused   (values around 1.0 = no change)
};
@group(0) @binding(0) var<uniform> params : AugParams;
@group(0) @binding(1) var samp   : sampler;
@group(0) @binding(2) var srcTex : texture_2d<f32>;
@group(0) @binding(3) var dstTex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>)
{
  let dims = textureDimensions(dstTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv   = (vec2<f32>(gid.xy) / vec2<f32>(dims)) * params.offScale.zw + params.offScale.xy;
  let pix  = textureSampleLevel(srcTex, samp, uv, 0.0).rgb * params.colour.xyz;
  textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(pix, 1.0));
}
