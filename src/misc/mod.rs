use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use async_std::sync::RwLock;
use bevy::ecs::world::CommandQueue;
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, AsyncComputeTaskPool, Task};
use bevy::utils::HashMap;
use bevy::{color::Color, prelude::Resource, time::Timer};
use hexx::{hex, HexLayout};
use rand::Rng;

use crate::math::generate_subdivided_hexagon;
use crate::terrain::{Chunk, Tile};

#[derive(Resource)]
pub struct EventTimer {
    pub field1: Timer,
}
struct OverlayColor;

impl OverlayColor {
    const RED: Color = Color::srgb(1.0, 0.0, 0.0);
    const GREEN: Color = Color::srgb(0.0, 1.0, 0.0);
}
fn start_background(
    mut commands: Commands,
    data: ResMut<MapData>,
    mesh_query: Query<&Mesh3d, With<MeshMarker>>,
) {
    let thread_pool = AsyncComputeTaskPool::get();
    let changed_len = data.0.write_blocking().changed.len();
    // let changed_len = map.changed.len();
    // let altered_coords: Vec<(i32, i32)> = map.changed.drain(..).collect();

    // let a = map.changed.pop;
    if changed_len > 0 {
        println!("BACKGROUND CHANGED LEN: {}", changed_len);
        let entity = commands.spawn_empty().id();
        let layout = HexLayout::pointy().with_hex_size(1.0);
        let d = data.0.clone();
        let task = thread_pool.spawn(async move {
            let mut map = d.write().await;
            let altered_coords: Vec<(i32, i32)> = map.changed.drain(..).collect();
            let duration = Duration::from_secs_f32(rand::thread_rng().gen_range(0.05..5.0));
            let radius = 1.0;
            let subdivisions = 2;
            let (hex_template_positions, _uvs, hex_template_indices, hex_vertex_weights) =
                generate_subdivided_hexagon(radius.into(), subdivisions);
            for coord in altered_coords {
                let hex = hex(coord.0, coord.1);
                let neighbors = hex.all_neighbors();
                let mut neighbor_heights: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                let mut h = 0.0;
                if let Some(tile) = map.tiles.get(&(hex.x, hex.y)) {
                    h = tile.position[1];
                }
                for (i, neighbor) in neighbors.into_iter().enumerate() {
                    let mut nh = h;

                    if let Some(neighbor_tile) = map.tiles.get(&(neighbor.x, neighbor.y)) {
                        nh = neighbor_tile.position[1];
                    }
                    neighbor_heights[i] = nh as f64;
                }
                let mut hex_point_heights: [f64; 7] = [0.0; 7];
                for i in 0..6 {
                    // Corner, avg height of three hexes
                    let prev = (i + 5) % 6;
                    hex_point_heights[i] =
                        (neighbor_heights[prev] + neighbor_heights[i] + h as f64) / 3.;
                }
                if let Some(tile) = map.tiles.get_mut(&(hex.x, hex.y)) {
                    for i in 0..tile.vertices.len() {
                        let mut vh = 0.0;
                        for j in 0..6 {
                            vh += hex_point_heights[j] * hex_vertex_weights[i][j];
                        }
                        // tile.vertices[i].y = vh as f32;
                        tile.vertices[i].y = h;
                    }
                }
            }
            let mut tiles: Vec<Tile> = Vec::new();

            for (_, t) in &map.tiles {
                tiles.push(t.clone());
            }

            for tile in tiles {
                for v in &tile.vertices {
                    map.chunks[tile.chunk_index].vertices[v.index][1] = v.y;
                }
            }

            // // Pretend this is a time-intensive function. :)
            // async_std::task::sleep(duration).await;
            let mut command_queue = CommandQueue::default();

            command_queue.push(move |world: &mut World| {
                // let query_handle = {
                //     let mut system_state = SystemState::<(Resource<MapData>)>::new(world);
                //     let (query_handle) = system_state.get_mut(world);
                //     query_handle.clone()
                // };
                world.entity_mut(entity).remove::<BackgroundTask>();
            });

            command_queue
        });

        commands.entity(entity).insert(BackgroundTask(task));
    }
}
fn interval_task(time: Res<Time>, mut timer: ResMut<EventTimer>, data: ResMut<MapData>) {
    if timer.field1.tick(time.delta()).just_finished() {
        let mut rng = rand::thread_rng();
        let counts = rng.gen_range(0_usize..5);
        let mut map = data.0.write_blocking();
        for _ in 0..0 {
            let layout = HexLayout::pointy().with_hex_size(1.0);
            let x = rng.gen_range(-20_i32..20);
            let y = rng.gen_range(-20_i32..20);
            let coord = (x, y);
            // let mut map = data.0.write().unwrap();
            let Some(tile) = map.tiles.get_mut(&coord) else {
                return;
            };

            tile.position[1] = 10.0;
            map.changed.push_back(coord);
        }
    }
}
fn receive_background(
    mut commands: Commands,
    mut background_tasks: Query<&mut BackgroundTask>,
    data: ResMut<MapData>,
    mut meshes: ResMut<Assets<Mesh>>,
    mesh_query: Query<(&Mesh3d, &MeshMarker)>,
) {
    for mut task in &mut background_tasks {
        if let Some(mut commands_queue) = block_on(future::poll_once(&mut task.0)) {
            commands.append(&mut commands_queue);
            let map = data.0.read_blocking();
            for (handle, marker) in &mesh_query {
                let mesh = meshes.get_mut(handle).unwrap();
                let id = marker.0 as usize;
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, map.chunks[id].vertices.clone());
            }
        }
    }
}
#[derive(Resource, Default, Deref, Clone)]
pub struct MapData(pub Arc<RwLock<Map>>);

#[derive(Default)]
pub struct Map {
    pub chunks: Vec<Chunk>,
    pub tiles: HashMap<(i32, i32), Tile>,
    pub changed: VecDeque<(i32, i32)>,
}

#[derive(Component)]
struct BackgroundTask(Task<CommandQueue>);

#[derive(Component)]
struct MeshMarker(u32);
