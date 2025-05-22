@group(0) @binding(0) var samplerLinear : sampler;
@group(0) @binding(1) var inputTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) fragUV : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
  let positions = array<vec2<f32>, 4>(
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
  );
  let uvs = array<vec2<f32>, 4>(
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 0.0)
  );

  var output : VertexOutput;
  output.Position = vec4(positions[vertexIndex], 0.0, 1.0);
  output.fragUV = uvs[vertexIndex];
  return output;
}

@fragment
fn fs_main(@location(0) fragUV: vec2<f32>) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(inputTexture));
  let offset = vec2<f32>(1.0) / texSize;

  var color = vec4<f32>(0.0);

  // Increased to 9x9 blur kernel for stronger effect
  let kernelSize = 9;
  let halfKernel = kernelSize / 2;
  let weight = 1.0 / f32(kernelSize * kernelSize);

  for (var x = -halfKernel; x <= halfKernel; x++) {
    for (var y = -halfKernel; y <= halfKernel; y++) {
      let sampleUV = fragUV + vec2<f32>(f32(x), f32(y)) * offset;
      color += textureSample(inputTexture, samplerLinear, sampleUV) * weight;
    }
  }

  return vec4(color.rgb, 1.0);
}
