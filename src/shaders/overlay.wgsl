/* Show webcam + green-tinted alpha mask (probability).  */

@group(0) @binding(0) var samp      : sampler;
@group(0) @binding(1) var videoTex  : texture_2d<f32>;
@group(0) @binding(2) var maskTex   : texture_2d<f32>;

struct VOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32>, };

@vertex
fn vs_main(@builtin(vertex_index) i : u32) -> VOut {
  var pos = array<vec2<f32>,4>(vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
  var uv  = array<vec2<f32>,4>(vec2(0,1),  vec2(1,1),  vec2(0,0),  vec2(1,0));
  var o : VOut;
  o.pos = vec4<f32>(pos[i], 0.0, 1.0);
  o.uv  = uv[i];
  return o;
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let rgb   = textureSample(videoTex, samp, uv).rgb;
  let m     = textureSample(maskTex , samp, uv).r;   // 0‒1
  let tint  = mix(rgb, vec3(0.0, 1.0, 0.0), m * 0.7); // 70 % green where mask=1
  return vec4<f32>(tint, 1.0);
}