use bevy::{
    asset::RenderAssetUsages,
    pbr::ExtendedMaterial,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
    utils::hashbrown::HashMap,
};
use fast_poisson::Poisson2D;
use geo::{coord, Contains, Coord, LineString};
use noise::{BasicMulti, MultiFractal, NoiseFn, SuperSimplex};
use rand::Rng;
use spade::{handles::VoronoiVertex, DelaunayTriangulation, Point2, Triangulation};
use uuid::Uuid;

use std::hash::{Hash, Hasher};

use bevy::utils::HashSet;
use hexx::{hex, EdgeDirection, Hex, HexLayout};

use crate::{
    art::CustomMaterial,
    math::{generate_subdivided_hexagon, sdf_dis},
    Terrain, HEX_RADIUS, MAP_SIZE,
};

pub mod erosion;

#[derive(Default, Clone)]
pub struct Tile {
    pub vertices: Vec<HexVertex>,
    pub hex: Hex,
    pub position: [f32; 3],
    pub chunk_index: usize,
    pub region: Uuid,
}

impl Tile {
    pub fn height(&self) -> f32 {
        self.position[1]
    }
}

#[derive(Default, Clone, Copy)]
pub struct HexVertex {
    pub index: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl HexVertex {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, index: 0 }
    }

    pub fn as_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

const CMP_SCALE: f32 = 100.0;

impl Eq for HexVertex {}

impl PartialEq for HexVertex {
    fn eq(&self, other: &Self) -> bool {
        let a: (i32, i32) = ((self.x * CMP_SCALE) as i32, (self.z * CMP_SCALE) as i32);
        let b: (i32, i32) = ((other.x * CMP_SCALE) as i32, (other.z * CMP_SCALE) as i32);

        a == b
    }
}

impl Hash for HexVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a: (i32, i32) = ((self.x * CMP_SCALE) as i32, (self.z * CMP_SCALE) as i32);
        a.hash(state);
    }
}

// WORLD
// Store tiles
// Utils for updating terrain
#[derive(Default)]
pub struct World {
    tiles: HashSet<Hex>,
}

// REGION
// Keep track of own tiles coords
// utils for updating region
#[derive(Default, Clone)]
pub struct Region {
    tiles: HashSet<Hex>,
}

#[derive(Default)]
pub struct Chunk {
    pub id: u32,
    pub vertex_set: HashSet<HexVertex>,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}
#[derive(Default)]
struct VRegion {
    id: Uuid,
    center: Vec2,
    vertices: HashSet<(HashF32, HashF32)>,
    edges: Vec<(Vec2, Vec2)>,
    indices: Vec<u32>,
    tiles: HashSet<Hex>,
}
#[derive(Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy)]
pub(crate) struct HashF32 {
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

fn get_map_hexes() -> Vec<Hex> {
    let rect_hexes: Vec<Hex> = hexx::shapes::pointy_rectangle(MAP_SIZE).collect();
    rect_hexes
}

// Generate Voronoi regions
// Claims hex tiles for each regions
// Use simplex noise for terrain generation
// stitch region borderes
pub(crate) fn generate_map(
    mut commands: Commands,
    mut custom_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,

    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Using Hexx crate for hexagon tile
    // Hexagons have pointy top (towards -Z direction)
    let layout = HexLayout::pointy().with_hex_size(1.0);
    let layout_tiles: Vec<hex::Hex> = hexx::shapes::pointy_rectangle(MAP_SIZE).collect();

    // State for every tile
    // - vertex heights
    // - position
    let mut tiles: HashMap<Hex, Tile> = HashMap::new();
    for t in &layout_tiles {
        tiles.insert(*t, Tile::default());
    }

    // Get world boundry min and max corners from first and last hex
    // in hexx layout
    let min_hex = layout.hex_to_world_pos(*layout_tiles.iter().next().unwrap());
    let max_hex = layout.hex_to_world_pos(*layout_tiles.iter().last().unwrap());
    let world_width = max_hex.x - min_hex.x;
    let world_height = max_hex.y - min_hex.y;
    let world_corners = vec![
        min_hex,
        Vec2::new(min_hex.x, max_hex.y),
        max_hex,
        Vec2::new(max_hex.x, min_hex.y),
    ];
    let world_boundries = vec![
        [world_corners[0], world_corners[1]],
        [world_corners[1], world_corners[2]],
        [world_corners[2], world_corners[3]],
        [world_corners[3], world_corners[0]],
    ];

    // Voronoi Region Generation
    // 1. Generate randomly distributed points with minimum
    //    separation between points
    // 2. Extract Voronoi Regions fro:m delaunay triangulation
    let mut voronoi_regions = create_voronoi_regions(
        min_hex,
        max_hex,
        world_width,
        world_height,
        world_boundries,
        world_corners.clone(),
    );

    // Assign tiles to Voronoi Regions ////
    let mut shapes = Vec::new();
    for (_i, reg) in voronoi_regions.iter().enumerate() {
        let linestring: Vec<Coord> = reg
            .edges
            .clone()
            .into_iter()
            .map(|(e1, _)| to_coord(e1))
            .collect();
        let polygon = geo::Polygon::new(LineString::new(linestring), vec![]);
        shapes.push(polygon);
    }
    let mut tile_set: HashSet<Hex> = HashSet::new();
    for t in &layout_tiles {
        tile_set.insert(*t);
    }

    for t in tile_set {
        let pos = layout.hex_to_world_pos(t);
        let pt = geo::Point::new(pos.x as f64, pos.y as f64);
        // println!("-------------");
        for (i, reg) in voronoi_regions.iter_mut().enumerate() {
            let shape = &shapes[i];
            if shape.contains(&pt) {
                reg.tiles.insert(t);
                if let Some(mt) = tiles.get_mut(&t) {
                    mt.region = reg.id;
                }
                break;
            }
        }
    }

    //// Draw World Boundry ////
    let bounding_rect_vertices: Vec<[f32; 3]> = world_corners
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

    //// Generate hexagon tile vertices ////
    let radius = HEX_RADIUS;
    let subdivisions = 0;
    let (hex_template_positions, _uvs, hex_template_indices, _hex_vertex_weights) =
        generate_subdivided_hexagon(radius.into(), subdivisions);

    //// Procedural terrain generation ////
    let mut rng = rand::thread_rng();
    let seed = rng.gen_range(0_u32..=1000000);
    let noise = BasicMulti::<SuperSimplex>::new(seed)
        .set_octaves(6)
        .set_frequency(0.008);
    let amp = 3.0;

    // let mut chunks = Vec::new();
    let mut chunks = HashMap::new();
    for reg in voronoi_regions.iter() {
        chunks.insert(reg.id, Chunk::default());
    }
    for region in voronoi_regions {
        for hex in region.tiles {
            if let Some(t) = tiles.get_mut(&hex) {
                t.region = region.id;
            }
        }
    }

    for (hex, tile) in tiles.iter_mut() {
        if let Some(tile_chunk) = chunks.get_mut(&tile.region) {
            let pos = layout.hex_to_world_pos(*hex);
            tile.position = [pos.x, 0.0, pos.y];
            let tile_height = (noise.get([pos.x as f64, pos.y as f64]) + 0.3) * amp;
            let mut index_map: Vec<u32> = Vec::new();
            for vertex in &hex_template_positions {
                let x1 = vertex[0] + pos.x;
                let y1 = vertex[2] + pos.y;
                let mut hex_vertex =
                    HexVertex::new(vertex[0] + pos.x, tile_height as f32, vertex[2] + pos.y);
                let index = tile_chunk.vertices.len();
                hex_vertex.index = index;
                tile.vertices.push(hex_vertex);
                tile_chunk.vertices.push(hex_vertex.as_array());
                index_map.push(hex_vertex.index as u32);
                // tile_chunk.vertices.push(hex_vertex.as_array());
            }
            for ind in &hex_template_indices {
                tile_chunk.indices.push(index_map[*ind as usize]);
            }
        }
        //
    }

    let mut neighbors = HashSet::new();
    let mut near_border = HashMap::new();

    for (hex, tile) in tiles.iter() {
        let mut regions = HashSet::new();
        regions.insert(tile.region);
        for neighbor in hex.all_neighbors() {
            if let Some(n) = tiles.get(&neighbor) {
                regions.insert(n.region);
            }
        }
        if regions.len() > 1 {
            neighbors.insert(*hex);
            for nearby in hex.range(10) {
                let existing = near_border.entry(nearby).or_insert(100.0);
                let dis = nearby.distance_to(*hex) as f64;
                if dis < *existing {
                    *existing = dis.max(1.0);
                }
            }
        }
    }

    for (hex, tile) in tiles.iter() {
        if neighbors.contains(hex) {
            if let Some(chunk_ref) = chunks.get_mut(&tile.region) {
                for v in tile.vertices.iter() {
                    chunk_ref.vertices[v.index][1] = 5.0;
                }
            }
        }
        if let Some(val) = near_border.get(hex) {
            if let Some(chunk_ref) = chunks.get_mut(&tile.region) {
                for v in tile.vertices.iter() {
                    chunk_ref.vertices[v.index][1] = (30.0 / (*val).powf(1.3) as f32).min(8.0);
                }
            }
        }
    }

    // test random tile height manipulation
    // for _ in 0..20 {
    //     let rand_x = rng.gen_range(-100..100);
    //     let rand_y = rng.gen_range(-100..100);
    //     let rand_hex = hex(rand_x, rand_y);
    //     if let Some(t) = tiles.get_mut(&rand_hex) {
    //         if let Some(chunk_ref) = chunks.get_mut(&t.region) {
    //             for v in t.vertices.iter() {
    //                 chunk_ref.vertices[v.index][1] = 20.0;
    //             }
    //         }
    //     }
    // }

    /*
    for region in voronoi_regions {
        let mut chunk = Chunk::default();
        // let amp = rng.gen_range(1_f64..30.0);
        // noise1 = noise1.set_frequency(rng.gen_range(0.003_f64..0.03));
        let sdfs = region.edges;
        for hex in region.tiles {
            let pos = layout.hex_to_world_pos(hex);
            let tile_height = (noise.get([pos.x as f64, pos.y as f64]) + 0.3) * amp;
            let tile_height = tile_height.max(0.01);
            let mut index_map: Vec<u32> = Vec::new();
            for vertex in &hex_template_positions {
                let x1 = vertex[0] + pos.x;
                let y1 = vertex[2] + pos.y;
                let dis = sdf_dis(&sdfs, Vec2::new(x1, y1));
                let f1 = (1.0 / (1.0 + dis.abs().powf(1.5) * 0.005)) as f64;
                let f2 = 1.0 - f1;
                let mut h = noise.get([x1 as f64, y1 as f64]);
                let sign = if h < 0.0 { -1.0 } else { 1.0 };
                h = h.powf(2.0) * sign * 2.0;
                let ampf = amp * f2;
                h = ampf * h + 0.5 * f1;
                h += ampf / 5.0;
                if h < 0.0 {
                    h /= 4.0;
                }
                h += 0.5;
                // let mut hex_vertex = HexVertex::new(vertex[0] + pos.x, h as f32, vertex[2] + pos.y);
                let mut hex_vertex =
                    HexVertex::new(vertex[0] + pos.x, tile_height as f32, vertex[2] + pos.y);

                if !crate::SHARE_VERTICES || !chunk.vertex_set.contains(&hex_vertex) {
                    let index = chunk.vertices.len();
                    hex_vertex.index = index;
                    chunk.vertex_set.insert(hex_vertex);
                    chunk.vertices.push(hex_vertex.as_array());
                } else {
                    if let Some(vi) = chunk.vertex_set.get(&hex_vertex) {
                        hex_vertex = *vi;
                    }
                }

                // if let Some(vi) = chunk.vertex_set.get(&hex_vertex) {
                //     hex_vertex = *vi;
                // } else {
                //     let index = chunk.vertices.len();
                //     hex_vertex.index = index;
                //     chunk.vertex_set.insert(hex_vertex);
                //     chunk.vertices.push(hex_vertex.as_array());
                // }
                // tile.vertices.push(hex_vertex);
                index_map.push(hex_vertex.index as u32);
            }
            for ind in &hex_template_indices {
                chunk.indices.push(index_map[*ind as usize]);
            }
        }
        // println!("REGION {j} has {} tiles", { reg.tiles.len() });
        chunks.push(chunk);
    }
    */

    // cube
    let cube_size = 0.8;
    let cube_color = materials.add(Color::srgb(0.0, 0.5, 1.0));
    let cube = meshes.add(Cuboid::new(cube_size, cube_size, cube_size));
    // let cube_tuple = (Mesh3d(cube.clone()), MeshMaterial3d(cube_color.clone()));

    // Draw mountain splines
    for _ in 0..10 {
        let mut mountain_range = Vec::new();
        let rand_x = rng.gen_range(MAP_SIZE[0]..MAP_SIZE[1]);
        let rand_y = rng.gen_range(MAP_SIZE[0]..MAP_SIZE[1]);
        // let rand_y = rng.gen_range(-100..100);
        let mut cursor_hex = hex(rand_x, rand_y);
        let mut direction = 0.0;
        let mut mountain_height = 10.0;

        for _ in 0..200 {
            let alter_height = rng.gen_range(-2.0_f32..2.0);
            mountain_height += alter_height;
            let pos = layout.hex_to_world_pos(cursor_hex);
            let pos3 = Vec3::new(pos.x, mountain_height, pos.y);
            mountain_range.push(pos3);

            let alter_course = rng.gen_range(-0.4_f32..0.4);
            direction += alter_course;

            let dir = EdgeDirection::from_pointy_angle(direction);
            let next_neighbor = cursor_hex.neighbor(dir);
            cursor_hex = next_neighbor;
        }

        let spline_positions: Vec<[f32; 3]> = mountain_range
            .clone()
            .into_iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();

        let mut indices: Vec<u32> = Vec::new();

        for i in 0..(spline_positions.len() - 1) {
            let next = (i + 1) % spline_positions.len();
            let p = spline_positions[i];
            indices.push(i as u32);
            indices.push(next as u32);
            // commands.spawn((
            //     Mesh3d(cube.clone()),
            //     MeshMaterial3d(cube_color.clone()),
            //     Transform::from_xyz(p[0], p[1], p[2]),
            // ));
        }
        let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, spline_positions)
            .with_inserted_indices(Indices::U32(indices));
        // commands.spawn((
        //     Mesh3d(meshes.add(mesh)),
        //     MeshMaterial3d(materials.add(Color::srgb(1.0, 0.0, 0.0))),
        // ));

        // modify tiles around mountain
        let mut nearby_hexes = HashSet::new();
        for node in &mountain_range {
            let hex = layout.world_pos_to_hex(Vec2::new(node.x, node.z));
            nearby_hexes.insert(hex);

            for nearby in hex.range(40) {
                nearby_hexes.insert(nearby);
            }
        }

        for hex in nearby_hexes {
            if let Some(t) = tiles.get_mut(&hex) {
                if let Some(chunk_ref) = chunks.get_mut(&t.region) {
                    let mut min_dis = 100000.0;
                    let mut min_height = 10000.0;

                    for n in &mountain_range {
                        let dis = ((t.position[0] - n[0]).powi(2) + (t.position[2] - n[2]).powi(2));
                        if dis < min_dis {
                            min_dis = dis;
                            min_height = n[1];
                        }
                    }

                    for v in t.vertices.iter() {
                        let mut factor = 1.0 / (1.0 + min_dis.powf(1.1) * 0.09);
                        chunk_ref.vertices[v.index][1] += min_height * factor / 4.00;
                    }
                }
            }
        }
        // for
    }
    for (_, chunk) in chunks {
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
        // if region.center.y.abs() > 200.0 {
        //     rand_g = 0.05;
        //     modc = (region.center.y.abs() - 200.0) as f32 / 300.00;
        // }
        let shader_mat = custom_materials.add(ExtendedMaterial {
            base: StandardMaterial::default(),
            extension: CustomMaterial {
                color: LinearRgba::new(0.05, rand_g, 0.05, 1.0),
                mod_color: LinearRgba::new(modc, modc, modc, 1.0),
            },
        });
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(shader_mat.clone()),
            Terrain,
        ));
    }
}

fn create_voronoi_regions(
    min_hex: Vec2,
    max_hex: Vec2,
    world_width: f32,
    world_height: f32,
    world_boundries: Vec<[Vec2; 2]>,
    world_corners: Vec<Vec2>,
) -> Vec<VRegion> {
    let min_separation = 100.0;
    let points =
        Poisson2D::new().with_dimensions([world_width as f64, world_height as f64], min_separation);
    let mut triangulation: DelaunayTriangulation<_> = DelaunayTriangulation::new();
    for p in points {
        let x: f32 = p[0] as f32 + min_hex.x;
        let y: f32 = p[1] as f32 + min_hex.y;
        triangulation
            .insert(Point2::new(x as f64, y as f64))
            .unwrap();
    }

    // check if point is within world
    fn is_inner(pt2: Point2<f64>, corners: (Vec2, Vec2)) -> bool {
        let x = pt2.x as f32;
        let y = pt2.y as f32;
        x > corners.0.x && x < corners.1.x && y > corners.0.y && y < corners.1.y
    }

    let corner_pts = (min_hex, max_hex);
    let mut voronoi_regions = Vec::new();

    for face in triangulation.voronoi_faces() {
        let mut vregion = VRegion::default();
        vregion.id = Uuid::new_v4();
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

                    for l in &world_boundries {
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
                [VoronoiVertex::Inner(from), VoronoiVertex::Outer(edge)]
                | [VoronoiVertex::Outer(edge), VoronoiVertex::Inner(from)] => {
                    let from_pos = from.circumcenter();
                    let edge_dir = edge.direction_vector();
                    if !is_inner(from_pos, corner_pts) {
                        continue;
                    }
                    let mut to_pos2 = None;

                    let mut k = 0;
                    for l in &world_boundries {
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
                [VoronoiVertex::Outer(_), VoronoiVertex::Outer(_)] => {}
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
                (0, 1) => corner = Some(world_corners[1]),
                (1, 2) => corner = Some(world_corners[2]),
                (2, 3) => corner = Some(world_corners[3]),
                (0, 3) => corner = Some(world_corners[0]),
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

        voronoi_regions.push(vregion);
    }

    voronoi_regions
}
