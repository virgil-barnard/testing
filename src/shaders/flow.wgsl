@group(0) @binding(0) var prevTex : texture_2d<f32>;
@group(0) @binding(1) var currTex : texture_2d<f32>;
@group(0) @binding(2) var flowTex : texture_storage_2d<rgba16float, write>;   // xy-flow, z unused

const K = 1.0; // Horn-Schunck single-pass approximation

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>)
{
  let dims = textureDimensions(currTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv   = vec2<i32>(gid.xy);
  let i_x  = (textureLoad(currTex, uv + vec2<i32>(1,0),0).r -
              textureLoad(currTex, uv + vec2<i32>(-1,0),0).r) * 0.5;
  let i_y  = (textureLoad(currTex, uv + vec2<i32>(0,1),0).r -
              textureLoad(currTex, uv + vec2<i32>(0,-1),0).r) * 0.5;
  let i_t  = textureLoad(currTex, uv, 0).r - textureLoad(prevTex, uv, 0).r;

  // one-shot HS update
  let denom = K*K + i_x*i_x + i_y*i_y;
  let u = -K * i_x * i_t / denom;
  let v = -K * i_y * i_t / denom;

  textureStore(flowTex, uv, vec4<f32>(u, v, 0.0, 0.0));
}
