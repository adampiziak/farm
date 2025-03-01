use std::f64::consts::PI;

use naturalneighbor::Interpolator;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Clone, Default)]
pub struct HexTile {
    pub start: i32,
    pub height: f64,
    pub mag: f64,
    pub neighbor_heights: [f32; 6],
    pub vertices: Vec<[f32; 3]>,
    pub vertices_mags: Vec<f32>,
    pub indices: Vec<u32>,
    pub outline_vertices: Vec<[f32; 3]>,
    pub outline_indices: Vec<u32>,
}

#[derive(Default)]
pub struct HexChunk {
    pub vertices: Vec<[f32; 3]>,
    pub vertices_mags: Vec<f32>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub outline_vertices: Vec<[f32; 3]>,
    pub outline_indices: Vec<u32>,
}

impl Point {
    fn midpoint(&self, other: &Point) -> Point {
        Point {
            x: (self.x + other.x) / 2.0,
            y: (self.y + other.y) / 2.0,
        }
    }

    fn distance_to_point(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt() as f32
    }

    fn distance_to_line(&self, p1: &Point, p2: &Point) -> f64 {
        let numerator =
            ((p2.y - p1.y) * self.x - (p2.x - p1.x) * self.y + p2.x * p1.y - p2.y * p1.x).abs();
        let denominator = ((p2.y - p1.y).powi(2) + (p2.x - p1.x).powi(2)).sqrt();
        numerator / denominator
    }
}

fn generate_hexagon_vertices(radius: f64) -> Vec<Point> {
    (0..6)
        .map(|i| {
            let angle = i as f64 * PI / 3.0 - PI / 6.0; // 60 degrees in radians
            Point {
                x: radius * angle.cos(),
                y: radius * angle.sin(),
            }
        })
        .collect()
}
pub fn generate_hexagon_outline(
    radius: f64,
    subdivisions: i32,
) -> (Vec<[f32; 3]>, Vec<u32>, Vec<[f32; 6]>) {
    let points: Vec<[f32; 3]> = (0..6)
        .map(|i| {
            let angle = i as f64 * PI / 3.0; // 60 degrees in radians
            [
                (radius * angle.cos()) as f32,
                0.0,
                (radius * angle.sin()) as f32,
            ]
        })
        .collect();
    let mut all = Vec::new();
    let mut distances: Vec<[f32; 6]> = Vec::new();
    for i in 0..6 {
        let divisions = 2_i32.pow(subdivisions as u32);
        let next_i = (i + 1) % 6;
        let pt_i = points[i];
        let pt_next = points[next_i];
        let xdis = pt_next[0] - pt_i[0];
        let ydis = pt_next[2] - pt_i[2];
        let xdiff = xdis / divisions as f32;
        let ydiff = ydis / divisions as f32;

        for j in 0..(divisions) {
            let mut dis = [0.0; 6];
            all.push([pt_i[0] + xdiff * j as f32, 0.0, pt_i[2] + ydiff * j as f32]);
            dis[i] = (divisions as f32 - j as f32) / divisions as f32;
            dis[next_i] = (j as f32) / divisions as f32;
            distances.push(dis);
        }
    }

    let mut indices = Vec::new();
    for i in 0..all.len() {
        let next = (i + 1) % all.len();
        indices.push(i as u32);
        indices.push(next as u32);
    }
    println!("OUTLINE LENGTH: {}", all.len());

    (all, indices, distances)
}

fn subdivide_triangle(p1: &Point, p2: &Point, p3: &Point, depth: usize) -> Vec<[Point; 3]> {
    if depth == 0 {
        return vec![[*p1, *p2, *p3]];
    }

    let m1 = p1.midpoint(p2);
    let m2 = p2.midpoint(p3);
    let m3 = p3.midpoint(p1);

    let mut triangles = Vec::new();
    triangles.extend(subdivide_triangle(p1, &m1, &m3, depth - 1));
    triangles.extend(subdivide_triangle(&m1, p2, &m2, depth - 1));
    triangles.extend(subdivide_triangle(&m3, &m2, p3, depth - 1));
    triangles.extend(subdivide_triangle(&m1, &m2, &m3, depth - 1));

    triangles
}

pub fn generate_subdivided_hexagon(
    radius: f64,
    subdivisions: usize,
) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u16>, Vec<[f64; 7]>) {
    let vertices = generate_hexagon_vertices(radius);
    let center = Point { x: 0.0, y: 0.0 };

    let mut positions = Vec::new();
    let uvs = Vec::new();
    let mut indices = Vec::new();
    let mut distances = Vec::new();
    // positions.push([0.0 as f32, 1.2 as f32, 0.0 as f32]);

    for (i, vertex) in vertices.iter().enumerate() {
        // positions.push([vertex.x as f32, 1.0 as f32, vertex.y as f32]);
        // uvs.push([
        //     vertex.x as f32 / radius as f32 * 0.5 + 0.5,
        //     vertex.y as f32 / radius as f32 * 0.5 + 0.5,
        // ]);
    }

    // for i in 0..6 {
    //     let next = (i + 1) % 6;
    //     indices.push((next + 1) as u16);
    //     indices.push((i + 1) as u16);
    //     indices.push(0);
    // }
    // indices.push(0);
    // indices.push((i) as u16);
    // indices.push((i + 1) as u16);
    // return (positions, uvs, indices, distances);

    // for vertex in &positions {
    //     let mut distance_array = [0.0; 6];
    //     for i in 0..6 {
    //         let p1 = vertices[i];
    //         let p2 = vertices[(i + 1) % 6];
    //         distance_array[i] = Point {
    //             x: vertex[0] as f64,
    //             y: vertex[1] as f64,
    //         }
    //         .distance_to_line(&p1, &p2);
    //     }
    //     distances.push(distance_array);
    // }

    let mut triangles = Vec::new();
    for i in 0..6 {
        let next = (i + 1) % 6;
        triangles.extend(subdivide_triangle(
            &center,
            &vertices[i],
            &vertices[next],
            subdivisions,
        ));
    }

    let mut midpoints = Vec::new();
    for i in 0..6 {
        let next = (i + 1) % 6;
        let mp = vertices[i].midpoint(&vertices[next]);
        midpoints.push(mp);
    }

    let mut hex_points = Vec::new();
    for i in 0..6 {
        hex_points.push(vertices[i]);
        hex_points.push(midpoints[i]);
    }
    hex_points.push(center);
    let mut edges = Vec::new();

    for i in 0..6 {
        let next = (i + 1) % 6;
        edges.push((vertices[i], vertices[next]));
    }

    let nnpoints: Vec<naturalneighbor::Point> = vertices
        .clone()
        .into_iter()
        .map(|p| naturalneighbor::Point { x: p.x, y: p.y })
        .collect();
    // nnpoints.push(naturalneighbor::Point {
    //     x: center.x,
    //     y: center.y,
    // });
    let interpolator = Interpolator::new(&nnpoints);

    // println!("WEIGHTS AT POINT");
    // println!("{:?}", weightatpoint);
    // let mut total = 0.0;
    // for w in weightatpoint {
    //     total += w.1;
    // }
    // println!("TOTAL: {total}");

    for triangle in triangles {
        for vertex in triangle.iter().rev() {
            positions.push([vertex.x as f32, 1.0, vertex.y as f32]);
            // uvs.push([
            //     vertex.x as f32 / radius as f32 * 0.5 + 0.5,
            //     vertex.y as f32 / radius as f32 * 0.5 + 0.5,
            // ]);
            indices.push(indices.len() as u16);

            // Get distances to hex edges
            // let mut distance_to_edges = Vec::new();
            let min_dis = 10000000.0;
            let min_dis_i = 0;

            let mut weights: [f64; 7] = [0.0; 7];

            // weights of each vertex and center
            // let mut weights = [100000.0; 7];
            let mut on_edge = false;
            for i in 0..6 {
                let p1 = edges[i].0;
                let p2 = edges[i].1;
                // let dis = vertex.distance_to_point(&p1);
                // vert_distances[i] = dis;
                // continue;
                let edge_dis = vertex.distance_to_line(&p1, &p2);
                // distance_to_edges.push(edge_dis);
                if edge_dis < 0.001 {
                    on_edge = true;
                    let dis1 = vertex.distance_to_point(&p1) as f64;
                    let dis2 = vertex.distance_to_point(&p2) as f64;
                    // vert_distances = [0.0; 7];
                    weights[i] = dis2 / (dis1 + dis2);
                    weights[(i + 1) % 6] = dis1 / (dis1 + dis2);
                    break;
                }
                // } else {
                //     let mut dis = vertex.distance_to_point(&p1);
                //     // if dis > 0.8 {
                //     //     dis = 10000.0;
                //     // }
                //     vert_distances[i] = dis;
                // }
                // let dis = vertex.distance_to_point(&vertices[i]);
                // vert_distances[i] = dis;
                // if dis < min_dis {
                //     min_dis = dis;
                //     min_dis_i = i;
                // }
                // distance_array[i] = vertex.distance_to_point(&hex_points[i]);
            }
            // vert_distances[6] = vertex.distance_to_point(&center);
            // vert_distances[6] = 10.0;
            if !on_edge {
                let inter_weight = interpolator
                    .query_weights(naturalneighbor::Point {
                        x: vertex.x,
                        y: vertex.y,
                    })
                    .unwrap()
                    .unwrap();

                for (i, w) in inter_weight {
                    weights[i] = w;
                }
                // vert_distances[6] = 10.0;
                // vert_distances[6] =
                //     (vertex.distance_to_point(&center) - 1. / (subdivisions * 16) as f32).max(0.0);
                // vert_distances[6] = vertex.distance_to_point(&center);
            }
            // vert_distances[6] = 200000.0;
            distances.push(weights);

            // // Determine weighting factor along edge and to center
            // // 1. To edge
            // let p1 = edges[min_dis_i].0;
            // let p2 = edges[min_dis_i].1;
            // let p1_dis = vertex.distance_to_point(&p1).powi(2);
            // let p2_dis = vertex.distance_to_point(&p2).powi(2);
            // let edge_factor_p1 = p2_dis / (p1_dis + p2_dis);
            // let edge_factor_p2 = p1_dis / (p1_dis + p2_dis);

            // // 2. To center
            // // if on edge only use edge factor, other interpolate between edge and center
            // let mut edge_factor = 1.0;
            // let mut center_factor = 0.0;
            // let center_dis = vertex.distance_to_point(&center).powi(2);
            // if min_dis != 0.0 {
            //     edge_factor = center_dis / (min_dis.powi(1) as f32 / 2. + center_dis);
            //     center_factor = 1 as f32 - edge_factor;
            // }

            // // center point weight
            // weights[6] = center_factor;
            // let next = (min_dis_i + 1) % 6;
            // weights[min_dis_i] = edge_factor_p1 * edge_factor;
            // weights[next] = edge_factor_p2 * edge_factor;
            // distances.push(weights);
        }
    }

    (positions, uvs, indices, distances)
}
