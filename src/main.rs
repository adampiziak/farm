use bevy::prelude::*;

use bevy::dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::reflect::Map;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{RenderPipelineDescriptor, SpecializedMeshPipelineError};
use bevy::utils::{HashMap, HashSet};
use bevy::{
    asset::RenderAssetUsages,
    pbr::wireframe::{Wireframe, WireframePlugin},
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_resource::{AsBindGroup, ShaderRef},
        settings::{WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
    text::FontSmoothing,
};
use geo::{coord, Contains, Coord, LineString, Polygon};
use misc::MapData;
use spade::handles::{VoronoiVertex::Inner, VoronoiVertex::Outer};

use fast_poisson::Poisson2D;
use hexx::{hex, Hex, HexLayout};
use math::generate_subdivided_hexagon;
use noise::{BasicMulti, MultiFractal, NoiseFn, SuperSimplex};
use noisy_bevy::NoisyShaderPlugin;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use spade::handles::VoronoiVertex;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use terrain::{Chunk, HexVertex};

const HEX_RADIUS: f32 = 1.0;
// const MAP_SIZE: [i32; 4] = [-300, 300, -250, 250];
// const MAP_SIZE: [i32; 4] = [-300, 300, -300, 300];
// const MAP_SIZE: [i32; 4] = [-100, 100, -100, 100];
// const MAP_SIZE: [i32; 4] = [-150, 150, -150, 150];
const MAP_SIZE: [i32; 4] = [-50, 50, -50, 50];

pub const SHADER_ASSET_PATH: &str = "hexagon_shader.wgsl";

mod math;
mod misc;
mod terrain;

fn main() {
    App::new()
        .insert_resource(MapData::default())
        .add_plugins((
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: bevy::render::settings::RenderCreation::Automatic(
                        WgpuSettings {
                            features: WgpuFeatures::POLYGON_MODE_LINE,
                            ..Default::default()
                        },
                    ),
                    ..Default::default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_config: TextFont {
                        // Here we define size of our overlay
                        font_size: 24.0,
                        // If we want, we can use a custom font
                        font: default(),

                        // We could also disable font smoothing,
                        font_smoothing: FontSmoothing::default(),
                    },
                    // We can also change color of the overlay
                    text_color: Color::WHITE,
                    enabled: true,
                },
            },
            WireframePlugin,
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, CustomMaterial>>::default(),
        ))
        .insert_resource(misc::EventTimer {
            field1: Timer::from_seconds(4.0, TimerMode::Repeating),
        })
        .add_plugins(NoisyShaderPlugin)
        .add_systems(Startup, generate_map)
        .add_systems(Startup, setup_lighting)
        .add_systems(Update, move_player)
        .add_systems(Startup, setup_camera)
        .add_systems(Update, toggle_wireframe)
        .run();
}

// fn draw_mesh_intersections(pointers: Query<&PointerInteraction>, mut gizmos: Gizmos) {
//     for i in pointers.iter()
//     // .filter_map(|interaction| interaction.get_nearest_hit())
//     // .filter_map(|(_entity, hit)| hit.position.zip(hit.normal))
//     {
//         // println!("{point:?}");
//         // gizmos.sphere(point, 0.05, RED_500);
//         // gizmos.arrow(point, point + normal.normalize() * 0.5, PINK_100);
//     }
// }

// This struct defines the data that will be passed to your shader
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct CustomMaterial {
    #[uniform(100)]
    color: LinearRgba,
    #[uniform(101)]
    mod_color: LinearRgba,
    // #[texture(101)]
    // #[sampler(102)]
    // color_texture: Option<Handle<Image>>,
    // #[texture(103)]
    // #[sampler(104)]
    // color_texture2: Option<Handle<Image>>,
    // #[texture(105)]
    // #[sampler(106)]
    // color_texture3: Option<Handle<Image>>,
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl MaterialExtension for CustomMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn specialize(
        _pipeline: &bevy::pbr::MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            // Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            //     ATTRIBUTE_MAGNITUDE_COLOR.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
    // fn specialize(
    //     _pipeline: &MaterialPipeline<Self>,
    //     descriptor: &mut RenderPipelineDescriptor,
    //     layout: &MeshVertexBufferLayoutRef,
    //     _key: MaterialPipelineKey<Self>,
    // ) -> Result<(), SpecializedMeshPipelineError> {
    //     let vertex_layout = layout.0.get_layout(&[
    //         Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
    //         Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
    //         ATTRIBUTE_MAGNITUDE_COLOR.at_shader_location(2),
    //     ])?;
    //     descriptor.vertex.buffers = vec![vertex_layout];
    //     Ok(())
    // }
}

fn distance_squared(a: Vec2, b: Vec2) -> f32 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2)
}

fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = Vec2 {
        x: b.x - a.x,
        y: b.y - a.y,
    };
    let ap = Vec2 {
        x: p.x - a.x,
        y: p.y - a.y,
    };
    let ab_len2 = ab.x * ab.x + ab.y * ab.y;

    if ab_len2 == 0.0 {
        return distance_squared(p, a).sqrt(); // a and b are the same point
    }

    let t = ((ap.x * ab.x + ap.y * ab.y) / ab_len2).clamp(0.0, 1.0);
    let closest = Vec2 {
        x: a.x + t * ab.x,
        y: a.y + t * ab.y,
    };
    distance_squared(p, closest).sqrt()
}

fn sdf_dis(sdfs: &Vec<(Vec2, Vec2)>, point: Vec2) -> f32 {
    let mut min_dis: f32 = 10000.0;
    for line in sdfs {
        let dis = distance_to_segment(point, line.0, line.1);
        if dis < 0.1 {
            return 0.0;
        }

        if dis < min_dis {
            min_dis = dis;
        }
    }
    // println!("{min_dis}");

    return min_dis;
}

#[derive(Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy)]
struct HashF32 {
    integral: i32,
    fractional: i32,
}

const HPER: f32 = 1000000000.0;

impl From<f32> for HashF32 {
    fn from(value: f32) -> Self {
        let sign = if value < 0.0 { -1 } else { 1 };
        HashF32 {
            integral: value.abs().floor() as i32 * sign,
            fractional: (value.fract() * HPER) as i32,
        }
    }
}

impl Into<f32> for HashF32 {
    fn into(self) -> f32 {
        self.integral as f32 + self.fractional as f32 / HPER
    }
}

fn to_coord(a: Vec2) -> Coord {
    coord! {
        x: a.x as f64,
        y: a.y as f64
    }
}

#[derive(Default)]
struct VRegion {
    center: Vec2,
    vertices: HashSet<(HashF32, HashF32)>,
    edges: Vec<(Vec2, Vec2)>,
    indices: Vec<u32>,
    tiles: HashSet<Hex>,
}

// Generate Voronoi regions
// Claims hex tiles for each regions
// Use simplex noise for terrain generation
// stitch region borderes
fn generate_map(
    mut commands: Commands,
    mut custom_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,

    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let layout = HexLayout::pointy().with_hex_size(1.0);
    let mut rng = rand::thread_rng();

    // Create all tiles and shuffle order
    let mut rect_hexes: Vec<hex::Hex> = hexx::shapes::pointy_rectangle(MAP_SIZE).collect();
    rect_hexes.shuffle(&mut thread_rng());

    // Take first 10 as starting region locations
    let mut rectangle_hexes = hexx::shapes::pointy_rectangle(MAP_SIZE).into_iter();

    let min_hex = layout.hex_to_world_pos(rectangle_hexes.next().unwrap());
    let max_hex = layout.hex_to_world_pos(rectangle_hexes.last().unwrap());
    let padding = 000.0;
    let width = (max_hex.x - min_hex.x) + padding * 2.0;
    let height = (max_hex.y - min_hex.y) + padding * 2.0;
    let reg_distance = 100.0;
    let points = Poisson2D::new().with_dimensions([width as f64, height as f64], reg_distance);
    let mut triangulation: DelaunayTriangulation<_> = DelaunayTriangulation::new();
    for p in points {
        let x: f32 = p[0] as f32 + min_hex.x - padding;
        let y: f32 = p[1] as f32 + min_hex.y - padding;
        triangulation
            .insert(Point2::new(x as f64, y as f64))
            .unwrap();
    }

    println!(
        "VORONOI FACE COUNT: {}",
        triangulation.voronoi_faces().len()
    );
    let rect_corners = vec![
        min_hex,
        Vec2::new(min_hex.x, max_hex.y),
        max_hex,
        Vec2::new(max_hex.x, min_hex.y),
    ];

    let rect_lines = vec![
        [rect_corners[0], rect_corners[1]],
        [rect_corners[1], rect_corners[2]],
        [rect_corners[2], rect_corners[3]],
        [rect_corners[3], rect_corners[0]],
    ];

    fn is_inner(pt2: Point2<f64>, corners: (Vec2, Vec2)) -> bool {
        let x = pt2.x as f32;
        let y = pt2.y as f32;
        x > corners.0.x && x < corners.1.x && y > corners.0.y && y < corners.1.y
    }

    let corner_pts = (min_hex, max_hex);

    let mut faces = Vec::new();

    for face in triangulation.voronoi_faces() {
        let mut vregion = VRegion::default();
        let mut face_intersections = HashSet::new();
        for edge in face.adjacent_edges() {
            match edge.as_undirected().vertices() {
                [VoronoiVertex::Inner(from), VoronoiVertex::Inner(to)] => {
                    let mut from_pos = from.circumcenter();
                    let mut to_pos = to.circumcenter();
                    if !is_inner(from_pos, corner_pts) && !is_inner(to_pos, corner_pts) {
                        continue;
                    }
                    let from_coord: geo::Coord<f32> = coord! {
                        x: from_pos.x as f32,
                        y: from_pos.y as f32
                    };
                    let to_coord: geo::Coord<f32> = coord! {
                        x: to_pos.x as f32,
                        y: to_pos.y as f32
                    };

                    let mut k = 0;

                    for l in &rect_lines {
                        let a1 = l[0];
                        let a2 = l[1];

                        let line1: geo::Line<f32> = geo::Line::new(
                            coord! {x: a1.x as f32, y: a1.y as f32},
                            coord! {x: a2.x as f32, y: a2.y as f32},
                        );
                        let line2 = geo::Line::new(from_coord, to_coord);
                        if let Some(geo::LineIntersection::SinglePoint { intersection, .. }) =
                            geo::line_intersection::line_intersection(line1, line2)
                        {
                            if !is_inner(to_pos, (min_hex, max_hex)) {
                                to_pos.x = intersection.x as f64;
                                to_pos.y = intersection.y as f64;
                            } else {
                                from_pos.x = intersection.x as f64;
                                from_pos.y = intersection.y as f64;
                            }

                            face_intersections.insert(k);
                            break;
                        }
                        k += 1;
                    }

                    let ver1 = (
                        HashF32::from(from_pos.x as f32),
                        HashF32::from(from_pos.y as f32),
                    );
                    let ver2 = (
                        HashF32::from(to_pos.x as f32),
                        HashF32::from(to_pos.y as f32),
                    );
                    vregion.vertices.insert(ver1);
                    vregion.vertices.insert(ver2);
                }
                [Inner(from), Outer(edge)] | [Outer(edge), Inner(from)] => {
                    let from_pos = from.circumcenter();
                    let edge_dir = edge.direction_vector();
                    if !is_inner(from_pos, corner_pts) {
                        continue;
                    }
                    let mut to_pos2 = None;

                    let mut k = 0;
                    for l in &rect_lines {
                        let a1 = l[0];
                        let a2 = l[1];
                        let a3 = Vec2::new(from_pos.x as f32, from_pos.y as f32);
                        let a4 = Vec2::new(edge_dir.x as f32, edge_dir.y as f32) * 2.0 + a3;

                        let line1 =
                            geo::Line::new(coord! {x: a1.x, y: a1.y}, coord! {x: a2.x, y: a2.y});
                        let line2 =
                            geo::Line::new(coord! {x: a3.x, y: a3.y}, coord! {x: a4.x, y: a4.y});

                        if let Some(geo::LineIntersection::SinglePoint { intersection, .. }) =
                            geo::line_intersection::line_intersection(line1, line2)
                        {
                            to_pos2 = Some(Vec2::new(intersection.x, intersection.y));
                            face_intersections.insert(k);
                        }
                        k += 1;
                    }

                    if let Some(to_pos) = to_pos2 {
                        let ver1 = (
                            HashF32::from(from_pos.x as f32),
                            HashF32::from(from_pos.y as f32),
                        );
                        let ver2 = (
                            HashF32::from(to_pos.x as f32),
                            HashF32::from(to_pos.y as f32),
                        );
                        vregion.vertices.insert(ver1);
                        vregion.vertices.insert(ver2);
                    }
                }
                [Outer(_), Outer(_)] => {}
            }
        }

        let c = face.as_delaunay_vertex().position();
        let center = Vec2::new(c.x as f32, c.y as f32);
        let mut fpoints = Vec::new();
        if face_intersections.len() == 2 {
            let mut fls: Vec<u32> = face_intersections.into_iter().collect();
            fls.sort();

            let points = (fls[0], fls[1]);
            let mut corner = None;
            match points {
                (0, 1) => corner = Some(rect_corners[1]),
                (1, 2) => corner = Some(rect_corners[2]),
                (2, 3) => corner = Some(rect_corners[3]),
                (0, 3) => corner = Some(rect_corners[0]),
                _ => {}
            }

            if let Some(cpt) = corner {
                let theta = (cpt.y.atan2(cpt.x).to_degrees() * 10.0) as i32;
                fpoints.push((cpt, theta));
            }
        }

        for ver in &vregion.vertices {
            let pt = Vec2::new(ver.0.into(), ver.1.into());
            let offset = pt - center;
            let theta = (offset.y.atan2(offset.x).to_degrees() * 10.0) as i32;
            fpoints.push((pt, theta));
        }

        fpoints.sort_by_key(|k| k.1);

        let mut fvers: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for (pt, _) in fpoints {
            fvers.push([pt.x, 10.0, pt.y]);
        }
        for i in 0..(fvers.len()) {
            let next = (i + 1) % fvers.len();

            indices.push(i as u32);
            indices.push(next as u32);
            let ept1 = Vec2::new(fvers[i][0], fvers[i][2]);
            let ept2 = Vec2::new(fvers[next][0], fvers[next][2]);
            vregion.edges.push((ept1, ept2));
        }
        vregion.center = center;
        vregion.indices = indices;

        faces.push(vregion);
    }

    let bounding_rect_vertices: Vec<[f32; 3]> = rect_corners
        .clone()
        .into_iter()
        .map(|p| [p[0] as f32, 10.0, p[1] as f32])
        .collect();
    let bounding_rect_indices = vec![0, 1, 1, 2, 2, 3, 3, 0];
    let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::all())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, bounding_rect_vertices)
        .with_inserted_indices(Indices::U32(bounding_rect_indices));
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(Color::srgb(0.0, 0.0, 1.0))),
    ));

    let mut shapes = Vec::new();
    for (_i, reg) in faces.iter().enumerate() {
        let linestring: Vec<Coord> = reg
            .edges
            .clone()
            .into_iter()
            .map(|(e1, _)| to_coord(e1))
            .collect();
        let polygon = Polygon::new(LineString::new(linestring), vec![]);
        shapes.push(polygon);
        //
    }

    // Claim tiles for regions
    let mut tile_set: HashSet<Hex> = HashSet::new();
    for t in rect_hexes {
        tile_set.insert(t);
    }

    for t in tile_set {
        let pos = layout.hex_to_world_pos(t);
        let pt = geo::Point::new(pos.x as f64, pos.y as f64);
        // println!("-------------");
        for (i, reg) in faces.iter_mut().enumerate() {
            let shape = &shapes[i];
            if shape.contains(&pt) {
                reg.tiles.insert(t);
                break;
            }
        }
    }
    println!("DONE");
    let radius = HEX_RADIUS;
    let subdivisions = 0;
    // Generate map
    let (hex_template_positions, _uvs, hex_template_indices, _hex_vertex_weights) =
        generate_subdivided_hexagon(radius.into(), subdivisions);
    let seed2 = rng.gen_range(0_u32..=1000000);
    let mut noise1 = BasicMulti::<SuperSimplex>::new(seed2);
    // let mut noise1 = RidgedMulti::<SuperSimplex>::new(seed2);
    noise1.frequency = 0.008;
    noise1 = noise1.set_octaves(8);
    // noise1.octaves = 1;
    let amp = 32.0;

    for reg in faces {
        println!("REG");
        let mut chunk = Chunk::default();
        // let amp = rng.gen_range(1_f64..30.0);
        // noise1 = noise1.set_frequency(rng.gen_range(0.003_f64..0.03));
        let sdfs = reg.edges;
        for hex in reg.tiles {
            let pos = layout.hex_to_world_pos(hex);
            // let tile_height = noise1.get([pos.x as f64, pos.y as f64]);
            let mut index_map: Vec<u32> = Vec::new();
            for vertex in &hex_template_positions {
                let x1 = vertex[0] + pos.x;
                let y1 = vertex[2] + pos.y;
                let dis = sdf_dis(&sdfs, Vec2::new(x1, y1));
                let f1 = (1.0 / (1.0 + dis.abs().powf(1.5) * 0.005)) as f64;
                let f2 = 1.0 - f1;
                let mut h = noise1.get([x1 as f64, y1 as f64]);
                let sign = if h < 0.0 { -1.0 } else { 1.0 };
                h = h.powf(2.0) * sign * 2.0;
                let ampf = amp * f2;
                h = ampf * h + 0.5 * f1;
                h += ampf / 5.0;
                if h < 0.0 {
                    h /= 4.0;
                }
                h += 0.5;
                let mut hex_vertex = HexVertex::new(vertex[0] + pos.x, h as f32, vertex[2] + pos.y);

                if let Some(vi) = chunk.vertex_set.get(&hex_vertex) {
                    hex_vertex = *vi;
                } else {
                    let index = chunk.vertices.len();
                    hex_vertex.index = index;
                    chunk.vertex_set.insert(hex_vertex);
                    chunk.vertices.push(hex_vertex.as_array());
                }
                // tile.vertices.push(hex_vertex);
                index_map.push(hex_vertex.index as u32);
            }
            for ind in &hex_template_indices {
                chunk.indices.push(index_map[*ind as usize]);
            }
        }
        // println!("REGION {j} has {} tiles", { reg.tiles.len() });
        let mut rng = rand::thread_rng();

        let mut uvs: Vec<[f32; 2]> = Vec::new();

        for v in &chunk.vertices {
            uvs.push([v[0], v[2]])
        }
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, chunk.vertices.clone())
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(chunk.indices.clone()))
            // .with_inserted_attribute(
            //     Mesh::ATTRIBUTE_COLOR,
            //     vec![[r, g, b, 1.0]; hex_template_positions.len()],
            // )
            .with_computed_normals();
        let mut rand_g = rng.gen_range(0.0_f32..0.2);
        let mut modc = 0.0;
        if reg.center.y.abs() > 200.0 {
            rand_g = 0.05;
            modc = (reg.center.y.abs() - 200.0) as f32 / 300.00;
        }
        let shader_mat = custom_materials.add(ExtendedMaterial {
            base: StandardMaterial::default(),
            extension: CustomMaterial {
                color: LinearRgba::new(0.05, rand_g, 0.05, 1.0),
                mod_color: LinearRgba::new(modc, modc, modc, 1.0),
            },
        });
        // commands.spawn((
        //     Mesh3d(meshes.add(mesh)),
        //     MeshMaterial3d(shader_mat.clone()),
        //     Terrain,
        // ));
    }

    // RIVER GENERATION
    let rect_hexes: Vec<hex::Hex> = hexx::shapes::pointy_rectangle(MAP_SIZE).collect();
    let mut tile_heightmap: HashMap<Hex, f64> = HashMap::new();
    for h in &rect_hexes {
        let pos = layout.hex_to_world_pos(*h);
        let tile_height = noise1.get([pos.x as f64, pos.y as f64]);
        tile_heightmap.insert(*h, tile_height * amp);
    }

    let mut river_hexes: Vec<Hex> = Vec::new();
    let mut non_river_hexes: Vec<Hex> = Vec::new();

    // iterate over all hexes
    for h in rect_hexes {
        let Some(hex_height) = tile_heightmap.get(&h) else {
            continue;
        };
        let mut lower = Vec::new();
        let mut higher = Vec::new();

        for (i, neighbor_hex) in h.all_neighbors().into_iter().enumerate() {
            let neighbor_height = tile_heightmap.get(&neighbor_hex).unwrap_or(hex_height);

            if neighbor_height < hex_height {
                lower.push((i, neighbor_height));
            } else {
                higher.push((i, neighbor_height));
            }
        }

        higher.sort_by_key(|h| h.0);

        if lower.len() == 1 {
            let lowest_hex_index = lower.get(0).unwrap().0;
            let (next_lowest_index, _) = higher
                .get(0)
                .expect("if lower is len 1, higher should have 5 members");

            let indexes = (lowest_hex_index, next_lowest_index);

            let mut is_adjacent = (lowest_hex_index as i32 - *next_lowest_index as i32).abs() == 1;

            match indexes {
                (0, 5) => {
                    is_adjacent = true;
                }
                (5, 0) => {
                    is_adjacent = true;
                }
                _ => {}
            };

            if !is_adjacent {
                river_hexes.push(h);
                continue;
            }
        }
        non_river_hexes.push(h);
    }

    let cube_size = 1.2;
    // cube mesh
    let hexagon_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, hex_template_positions)
        .with_inserted_indices(Indices::U16(hex_template_indices));
    let mesh_handle = meshes.add(hexagon_mesh);
    // let rivercube = meshes.add(Cuboid::new(cube_size, cube_size, cube_size));
    // blue
    let blue = materials.add(Color::srgb(0.0, 0.0, 1.0));
    let green = materials.add(Color::srgb(0.0, 1.0, 0.0));

    for river_hex in river_hexes {
        let pos = layout.hex_to_world_pos(river_hex);
        let Some(hex_height) = tile_heightmap.get(&river_hex) else {
            continue;
        };
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(blue.clone()),
            Transform::from_xyz(pos.x, *hex_height as f32 + 1.0, pos.y),
            Terrain,
        ));
    }
    for river_hex in non_river_hexes {
        let pos = layout.hex_to_world_pos(river_hex);
        let Some(hex_height) = tile_heightmap.get(&river_hex) else {
            continue;
        };
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(green.clone()),
            Transform::from_xyz(pos.x, *hex_height as f32 + 1.0, pos.y),
            Terrain,
        ));
    }
}

fn setup_lighting(mut commands: Commands) {
    commands.insert_resource(AmbientLight {
        color: bevy::color::palettes::css::GHOST_WHITE.into(),
        brightness: 2_000.0,
    });
}

#[derive(Debug, Component)]
struct Player;

#[derive(Debug, Component)]
struct WorldModelCamera;

fn move_player(input: Res<ButtonInput<KeyCode>>, mut player: Query<&mut Transform, With<Player>>) {
    let Ok(mut transform) = player.get_single_mut() else {
        return;
    };
    // let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
    // jjjj
    let translation = transform.translation;

    let step = 0.35;
    if input.pressed(KeyCode::KeyW) {
        transform.translation = Vec3 {
            z: translation.z - step,
            ..translation
        };
    }
    let rotate_step = 0.001;
    if input.pressed(KeyCode::ShiftLeft) {
        transform.translation = Vec3 {
            y: translation.y - step / 2.,
            ..translation
        };
    }
    if input.pressed(KeyCode::Space) {
        transform.translation = Vec3 {
            y: translation.y + step / 2.,
            ..translation
        };
    }
    if input.pressed(KeyCode::KeyE) {
        transform.rotate_x(rotate_step);
    }
    if input.pressed(KeyCode::KeyQ) {
        transform.rotate_x(-rotate_step);
    }
    if input.pressed(KeyCode::KeyZ) {
        transform.rotate_y(-rotate_step);
    }
    if input.pressed(KeyCode::KeyX) {
        transform.rotate_y(rotate_step);
    }
    if input.pressed(KeyCode::KeyD) {
        transform.translation = Vec3 {
            x: translation.x + step,
            ..translation
        };
    }
    if input.pressed(KeyCode::KeyA) {
        transform.translation = Vec3 {
            x: translation.x - step,
            ..translation
        };
    }
    if input.pressed(KeyCode::KeyS) {
        transform.translation = Vec3 {
            z: translation.z + step,
            ..translation
        };
    }
}

fn setup_camera(mut commands: Commands) {
    commands
        // .spawn((
        //     Player,
        //     Transform::from_xyz(4., 700.0, 430.0),
        //     Visibility::default(),
        // ))
        .spawn((
            Player,
            Transform::from_xyz(4., 200.0, 130.0),
            Visibility::default(),
        ))
        .with_children(|parent| {
            parent.spawn((WorldModelCamera,));

            // Spawn view model camera.
            parent.spawn((
                Camera3d::default(),
                Camera {
                    // Bump the order to render on top of the world model.
                    order: 1,

                    ..default()
                },
                Transform::from_xyz(10., 30., 10.).looking_to(
                    Vec3 {
                        x: 0.0,
                        y: -0.6,
                        z: -0.3,
                    },
                    Vec3::Y,
                ),
            ));
        });
}

#[derive(Component)]
struct Terrain;

fn toggle_wireframe(
    mut commands: Commands,
    landscapes_wireframes: Query<Entity, (With<Terrain>, With<Wireframe>)>,
    landscapes: Query<Entity, (With<Terrain>, Without<Wireframe>)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Enter) {
        println!("ENTER!");
        println!("{}", landscapes.iter().len());
        println!("{}", landscapes_wireframes.iter().len());
        for terrain in &landscapes {
            commands.entity(terrain).insert(Wireframe);
        }
        for terrain in &landscapes_wireframes {
            commands.entity(terrain).remove::<Wireframe>();
        }
    }
}
