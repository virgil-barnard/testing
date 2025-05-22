/* cnn_segmentation.wgsl  – 3×3 conv on luminance, sigmoid, 4-nbr smooth
   Centre pixel is multiplied by its weight again, so different crops
   really produce different masks → non-zero gradients.                */

@group(0) @binding(0) var<uniform> texSize : vec2<f32>;
@group(0) @binding(1) var           input   : texture_2d<f32>;
@group(0) @binding(2) var           weight  : texture_2d<f32>;  // r32float 3×3
@group(0) @binding(3) var           output  : texture_storage_2d<rgba8unorm, write>;

fn luminance(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));   // BT-709 Y
}

fn σ(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

fn conv_at(px: vec2<i32>) -> f32 {
  let dims = vec2<i32>(texSize);
  var sum  = 0.0;

  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      let texel = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);
      let lum   = luminance(textureLoad(input, texel, 0).rgb);
      let w     = textureLoad(weight, vec2<i32>(dx + 1, dy + 1), 0).r;
      sum += lum * w;                      // centre pixel multiplied again
    }
  }
  /* simple bias: a constant +0.1 (acts like centre-weight was before) */
  return sum + 0.1;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= u32(texSize.x) || gid.y >= u32(texSize.y)) { return; }

  let uv        = vec2<i32>(gid.xy);
  let score     = conv_at(uv);
  let p_self    = σ(score);

  /* 4-neighbour smoothing */
  var p_sum = p_self;
  let nbr = array<vec2<i32>,4>(vec2(1,0), vec2(-1,0), vec2(0,1), vec2(0,-1));
  for (var i=0u; i<4u; i++) {
    let q = clamp(uv + nbr[i], vec2<i32>(0), vec2<i32>(texSize) - 1);
    p_sum += σ(conv_at(q));
  }
  let p = p_sum / 5.0;

  textureStore(output, uv, vec4<f32>(p, p, p, 1.0));
}
