#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

struct MyExtendedMaterial {
    quantize_steps: u32,
    color: vec4<f32>
}

@group(2) @binding(100)
var<uniform> my_extended_material: MyExtendedMaterial;
@group(2) @binding(101) var<uniform> material_color: vec4<f32>;
@group(2) @binding(102) var material_color_texture: texture_2d<f32>;
@group(2) @binding(103) var material_color_sampler: sampler;
@group(2) @binding(104) var material_color_texture2: texture_2d<f32>;
@group(2) @binding(105) var material_color_sampler2: sampler;
fn axial_round(uv: vec2<f32>) -> vec2<f32> {
    var size = 1.0;

    var x = uv.x;
    var y = uv.y;
    let s = 1.7320508;
    let xgrid = round(uv.x);
    let ygrid = round(uv.y);
    x -= xgrid;
    y -= ygrid;

    var q = 0.0;
    var r = 0.0;
    if abs(x) > abs(y) {
        q = xgrid + round(x + 0.5 * y);
        r = ygrid;
    } else {
        r = ygrid + round(y + 0.5 * x);
        q = xgrid;
    }
    // r *= size;
    // q *= size;
    return vec2(q, r);
}

fn nearest_center(uv: vec2<f32>) -> vec2<f32> {
    let size = 1.0;
    var x = uv.x;
    var y = uv.y;

    let q = (sqrt(3.0) / 3.0 * x - 1.0 / 3.0 * y) / size;
    let r = 2.0 / 3.0 * y / size;

    let b = axial_round(vec2(q, r));
    var x3 = size * (sqrt(3.0) * b.x + sqrt(3.0) / 2.0 * b.y);
    var y3 = size * (3.0 / 2.0 * b.y);

    return vec2(x3, y3);
}
@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // generate a PbrInput struct from the StandardMaterial bindings
    // HEXAGON
    let uv = in.uv;
    let cen = nearest_center(uv);
    // let cen = vec2(10.0, 20.0);
    var  x = (uv.x - cen.x);
    var y = (uv.y - cen.y);
    let dis = sqrt(x * x + y * y);
    let s = vec2(1, 1.7320508);
    let p = abs(uv - cen);
    var c = max(dot(p, s * 0.5), p.x);
    let co = 0.85;
    if c > co {
        // c = 0.0 + (c - co) * 5.0;
        c = 0.8;
    } else {
        c = 0.0;
    }
    ////
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    var grass = textureSample(material_color_texture, material_color_sampler, in.uv/8.0 );
    var mountain = textureSample(material_color_texture2, material_color_sampler2, in.uv/12.0 );
    // grass[1] += 0.1;
    grass -= 0.13;
    mountain -= 0.05;

    // we can optionally modify the input before lighting and alpha_discard is applied
    // pbr_input.material.base_color = vec4(0.2, 0.3, 0.2, 1.0);
    // pbr_input.material.base_color = vec4(0.2, 0.3, 0.2, 1.0);

    // alpha discard
    // pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

// #ifdef PREPASS_PIPELINE
//     // in deferred mode we can't modify anything after that, as lighting is run in a separate fullscreen shader.
//     let out = deferred_output(in, pbr_input);
// #else
    var h = in.world_position[1] + 0.5;
    var f = 1.0/(1.0 + h*h*h*h/32.0);
    pbr_input.material.base_color = mix(material_color, mix(mountain, grass, f), 1.0);
    let hco = 6.5;
    if h > hco {
        
    pbr_input.material.base_color += (h-hco)/8.0;
    }
    pbr_input.material.base_color += c/2.0;
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);
        
    // out.color += h/20.0;

    // we can optionally modify the lit color before post-processing is applied
    // out.color = vec4<f32>(vec4<u32>(out.color * f32(my_extended_material.quantize_steps))) / f32(my_extended_material.quantize_steps);

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    // we can optionally modify the final result here
    // out.color = out.color * 2.0;
// #endif

    return out;
}

