// normalise.wgsl  – per-channel mean-std normalisation

struct Stats {
    mean : vec3<f32>,
    invStd : vec3<f32>,   // 1 / σ  (pre-computed on CPU each frame)
};

@group(0) @binding(0) var<uniform> stats : Stats;
@group(0) @binding(1) var samp        : sampler;
@group(0) @binding(2) var srcTex      : texture_2d<f32>;
@group(0) @binding(3) var dstTex      : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>)
{
    let dims = textureDimensions(dstTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    /* -- fetch + normalise -------------------------------------------------- */
    let rgb  = textureSampleLevel(srcTex, samp,
                                  (vec2<f32>(gid.xy)+0.5) / vec2<f32>(dims), 0).rgb;
    let z    = (rgb - stats.mean) * stats.invStd;     // zero-mean, unit-var
    let gain = 0.25;                                  // tame the range
    let out  = clamp(z * gain, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(out, 1.0));
}
