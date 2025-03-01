#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
// Uniforms
// @group(0) @binding(0)
// var<uniform> resolution: vec2<f32>;

@group(2) @binding(100) var<uniform> material_color: vec4<f32>;
@group(2) @binding(101) var<uniform> mc: vec4<f32>;


const SIZE: f32 = 1.0;

// Vertex Output Struct
struct VertexOutput {

    @builtin(position) position: vec4<f32>,
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>
};
struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    // @location(1) normals: vec3<f32>,
    // // @location(2) magnitudes: f32,
};

// Simple passthrough vertex shader
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.position = mesh_position_local_to_clip(
        get_world_from_local(vertex.instance_index),
        vec4<f32>(vertex.position, 1.0),
    );
    out.pos = vertex.position;
    out.uv = vertex.uv;
    // out.pos = vertex.pos;
    // out.uv = (vertex.uv + vec2<f32>(1.0)) * 0.5; // Convert from [-1,1] to [0,1]
    return out;
}



fn axial_round(uv: vec2<f32>) -> vec2<f32> {
    var size = SIZE;

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
    let size = SIZE;
    var x = uv.x;
    var y = uv.y;

    let q = (sqrt(3.0) / 3.0 * x - 1.0 / 3.0 * y) / size;
    let r = 2.0 / 3.0 * y / size;

    let b = axial_round(vec2(q, r));
    var x3 = size * (sqrt(3.0) * b.x + sqrt(3.0) / 2.0 * b.y);
    var y3 = size * (3.0 / 2.0 * b.y);

    return vec2(x3, y3);
}


// // Fragment Shader
@fragment
// fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
//     return vec4<f32>(1.0, 0.0, 1.0, 1.0);
// }
fn fragment(@location(0) position: vec3<f32>) -> @location(0) vec4<f32> {
    let uvx = position.x;
    let uvy = position.z;
    let uv = vec2(uvx, uvy);
    // let uv = vec2(uvx + sqrt(3.0), uvy);

    let cen = nearest_center(uv);
    // let cen = vec2(10.0, 20.0);
    var  x = (uv.x - cen.x);
    var y = (uv.y - cen.y);
    let dis = sqrt(x * x + y * y);
    let s = vec2(1, 1.7320508);
    let p = abs(uv - cen);
    var c = max(dot(p, s * 0.5), p.x);
    let co = 0.8;
    if c > co {
        c = 0.0 + (c - co) * 5.0;
    } else {
        c = 0.0;
    }
    let h = position.y;
    let he = 30.0;
    var r = material_color.x;
    var g = material_color.y;
    var b = material_color.z;
    if h < 0.0 {
        // b = 0.4;
        let wf = 1.0 / (h * -1.0 + 1.0);
        r = 0.1;
        b = wf + 0.1;
        g = 0.1;
    }

    if h > 0.0 {
        // g = 0.8 - max(h/50.0,0.0);
        // r = grass[0]/gf;
        // g = 1/(1+h/10.0);
        let yf = 0.5 / pow(1 + h, 3.0);
        let sf = 1.0 - 1.0 / (1 + (pow(h / 14.0, 4.0)));
        // let sf = 0.0;
        r += yf + sf;
        g += yf * 1.1 + sf ;
        b += yf / 4.0 + sf;
        let gf = h / 20.0;
        let gf2 = (h * h) / 300.0;
        g += gf;
        g = max(g - gf2, yf + sf);
        // b = grass[2]/gf;
        // r += 0.3/(1+h*0.8);
    }

    c /= 4.0;
    r += c ;
    g += c ;
    b += c ;
    r += mc.x ;
    g += mc.y ;
    b += mc.z ;
    return vec4<f32>(r, g, b, 1.0);
}
