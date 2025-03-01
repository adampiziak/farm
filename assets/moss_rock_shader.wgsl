#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import noisy_bevy::fbm_simplex_3d

struct CustomMaterial {
    color: vec4<f32>,
};
@group(2) @binding(0) var<uniform> material: CustomMaterial;
@group(2) @binding(101) var material_color_texture: texture_2d<f32>;
@group(2) @binding(102) var material_color_sampler: sampler;
@group(2) @binding(103) var material_color_texture2: texture_2d<f32>;
@group(2) @binding(104) var material_color_sampler2: sampler;
@group(2) @binding(105) var material_color_texture3: texture_2d<f32>;
@group(2) @binding(106) var material_color_sampler3: sampler;


struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normals: vec3<f32>,
    // @location(2) magnitudes: f32,
    @location(3) uvs: vec2<f32>,
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normals: vec3<f32>,
    @location(1) pos: vec3<f32>,
    // @location(2) magnitudes: f32,
    @location(3) uvs: vec2<f32>,
};


@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(vertex.instance_index),
        vec4<f32>(vertex.position, 1.0),
    );
    // out.clip_position = vec4<f32>(vertex.position, 1.0);
    out.normals = vertex.normals;
    out.pos = vertex.position;
    // out.magnitudes = vertex.magnitudes;
    out.uvs = vertex.uvs;
    return out;
}
struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normals: vec3<f32>,
    @location(1) pos: vec3<f32>,
    // @location(2) magnitudes: f32,
    @location(3) uvs: vec2<f32>
};

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    // return material.color * input.blend_color;
    let grass = textureSample(material_color_texture, material_color_sampler, input.uvs );
    let grass_normal1 = textureSample(material_color_texture3, material_color_sampler3, input.uvs);
    // let grass_normal2 = textureSample(material_color_texture3, material_color_sampler3, -input.uvs*0.25 );
    // let grass_normal3 = textureSample(material_color_texture3, material_color_sampler3, input.uvs*input.pos.xz);
    let grass_normal = grass_normal1;
    // let grass_normal = grass_normal1;
    // let tex_color1b = textureSample(material_color_texture, material_color_sampler, input.uvs*0.25);
    // let grass_color = tex_color1a*tex_color1b*80;
    let water = textureSample(material_color_texture2, material_color_sampler2, input.uvs);
    // tex_color2[0] -= 0.15;
    // tex_color2[2] -= 0.1;
    // let  mag = input.magnitudes;
    // var un = 1.0 - 1.0 / (1.0 + input.magnitudes *5.0);
    let light = dot(input.normals +(grass_normal.xyz)/1.0, normalize(vec3<f32>(0.5, 0.9, 0.0)));
    // let light = dot(input.normals, vec3<f32>(0.19, 0.98, 0.0));
    un /= 1.0;
    // var tex_color = mix(tex_color2, tex_color1, un);
    // if mag < 0.4 {

    // tex_color = tex_color2;
        
    // } 
    // if mag > 0.5 {
        
    // tex_color = tex_color1;
    // }
    
    // let tex_color = mix(tex_color2, tex_color1, input.magnitudes);
    // let tex_color = mix(tex_color1, tex_color2, 1.0);
    var warpx = input.pos;
    warpx[0] += 0.1;
    var warpy = input.pos;
    warpy[2] += 0.1;
    let warped_noisex = fbm_simplex_3d(warpx/10.0, 2, 2.0, 0.5);
    let warped_noisey = fbm_simplex_3d(warpy/100.0, 2, 2.0, 0.5);
    var warp2 = input.pos;
    let effect = 20.0;
    warp2[0] += effect*warped_noisex;
    var npos = input.pos;
    // npos[0] += warped_noisey;
    // npos[2]/= 5.0;

    // warp2[2] += effect*warped_noisey;
    var base1 = max(fbm_simplex_3d(npos*2.0, 2, 2.0, 0.5),0.1);
    var base2 = fbm_simplex_3d(npos, 2, 2.0, 0.5);
    // var base2 = fbm_simplex_3d(warp2*10.0, 3, 2.0, 0.5);
    var base3 = fbm_simplex_3d(npos*10.0, 2, 2.0, 0.5);
    var base4 = fbm_simplex_3d(npos*100.0, 2, 2.0, 0.5);
    let base = base1/2.0 + base2/4.0 +  base3/4.0 + base4;

    // var g = max(base/2.0,0.1) + 0.1;
    let h = input.pos[1];
    let he = 30.0;
    var g = 0.1;
    var r = 0.1;
    var b = 0.1;
    if h < 0.0 {
        // b = 0.4;
        let wf = 1.0/(h*-1.0+1.0);
        r = 0.1;
        b = wf + 0.1;
        g = 0.1;
    }

    if h > 0.0  {
        // g = 0.8 - max(h/50.0,0.0);
        // r = grass[0]/gf;
        // g = 1/(1+h/10.0);
        let yf = 0.5/pow(1+h, 3.0);
        let sf = 1.0 - 1.0/(1 + (pow(h/8.0, 4.0)));
        // let sf = 0.0;
        r = yf + sf;
        g = yf*1.1 + sf ;
        b = yf/4.0 + sf;
        let gf = h/20.0; 
        let gf2 = (h*h)/300.0; 
        g += gf;
        g = max(g - gf2, yf + sf);
        // b = grass[2]/gf;
        // r += 0.3/(1+h*0.8);

    }

    let gco = 0.0;
    if h > gco {
    }
    // if h > 10.0 {
    //     let gf = h*h/30.0; 
    //     g -= gf;
        
    // }

// if h > 0.0 {
    r *= light;
    g *= light;
    b *= light;
    
// }
    let sco = 4.0;
    if h > sco {
        let sf = 5.0;
        // r += (h-sco)/sf;
        // g += (h-sco)/sf;
        // b += (h-sco)/sf;
        
    }
    // var un = max(0.7 - input.magnitudes,0.0);
    // var un = 1.0 / (1.0 + input.magnitudes / 1.0);
    var un_h = 1.0 / (1.0 + h / 10.0);
    var neffect = 0.0;

    // var r = 0.0;
    // g *= un ;
    // g *= un_h ;
    // g = max(g, 0.1);
    // var b = 0.2;
    // if (un > 0.96) {
    // g += un/2.0;
    return vec4<f32>(r,g,b, 1.0);
    // return tex_color;

    // b = textureSample(material_color_texture, material_color_sampler, input.pos.xy);
    // } else {
    //     un = min(un/4.0, 0.3);
    //     r = 0.3 - un;
    //     g = 0.3 -un;
    //     b =0.3 -un;
    // }
    // if (up_normal > 0.5) {
    //     g = 1.0;
    // }
}



// @fragment
// fn fragment(
//     // @location(0) p: vec3<f32>,
//     // @location(1) uv: vec2<f32>,
//     // @location(2) normal: vec3<f32>,
// ) -> @location(0) vec4<f32> {
//     // let rock_color = vec3<f32>(0.8,0.8,0.8); // Base rock color
//     // return vec4<f32>(rock_color, 1.0);

//     // let pos = vec3f(p.x*4,p.y,p.z*4);
//     // let noise1 = fbm_simplex_3d(pos*90, 7, 0.4, 0.5);
//     // let noise2 = fbm_simplex_3d(pos*0.05, 7, 0.4, 0.5);
//     // let noise3 = fbm_simplex_3d(pos*0.1, 7, 0.4, 0.5);
//     // let noise4 = fbm_simplex_3d(pos*30, 7, 0.4, 0.5);
//     // let value2 = fbm_simplex_2d(scaled_uv2, 7, 3.0, 0.5);
//     // let value3 = fbm_simplex_2d(scaled_uv3, 7, 3.0, 0.5);
//     // let value4 = fbm_simplex_2d(scaled_uv4, 7, 3.0, 0.5);

//     // let con = 1 - 1/(1+value);

//     // var con = noise1 + noise2*noise2 - noise3*noise3 + noise4/6;
//     // con = con * con;
//     // let c = 0.2;
//     // if (con < 0.0) {
//     //     con = -1*con;
//     // }
//     // if (con < c) {
//     //     con = c;
//     // }
//     // let b = 0.3;
//     // if (con > b) {
//     //     con = b;
//     // }

//     let h = p.y;
//     let base = 0.0;
//     // con = con/16;
//     var red = h/32;
//     var blue = h/16;
//     var green = min(h/5,2.0);
//     if h < 1.5 {
//         blue = max(0.1 + h, 0.2);
//         red = 0.05;
//         green =0.1;
//     }
//     if h > 2.0 {
//         let o = h/30.0;
//         red = 0.4 - min(1.9*o, 0.36);
//         blue = 0.3 - min(1.5*o, 0.26);
//         green = 0.75 - min(3*o, 0.58);
//     }
//     let hco = 10.5;
//     if h > hco {
//         let diff = h - hco;
//         let o  = max(min(diff/20.0, 0.2), 0.1);
//         blue = 0.03 + o;
//         red = 0.0 + o;
//         green = 0.1 + min(o*2.1, 0.5);
//     }

//     // Define moss and rock colors
//     let rock_color = vec3<f32>(red, green, blue); // Base rock color
//     // let moss_color = vec3<f32>(77/255, 85/255, 28/255); // Mossy green
//     // let rock_color = vec3<f32>(77/255, 85/255, 28/255); // Mossy green

//     // Use noise to determine moss coverage
//     // let moss_factor = smoothstep(0.1, 0.8, noise); // Blend moss and rock
//     // let final_color = mix(rock_color, moss_color, moss_factor);

//     return vec4<f32>(rock_color, 1.0);
// }


