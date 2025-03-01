use bevy::{prelude::*, utils::HashMap};
use std::{collections::VecDeque, f32::consts::PI, thread};

use hexx::HexLayout;
use rand::Rng;

use crate::{misc::EventTimer, MapData, MAP_SIZE};

use super::Tile;

#[derive(Default)]
pub struct Drop {
    position: Vec2,
    speed: Vec2,
    age: i32,
    volume: f32,
    sediment: f32,
}

impl Drop {
    pub fn new(age: i32, volume: f32, sediment: f32) -> Self {
        Self {
            age,
            volume,
            sediment,
            ..Default::default()
        }
    }
}

// #[derive(Default)]
// pub struct Cell {
//     height: i32,
//     height: i32,
// }

#[derive(Default)]
pub struct WorldTerrain {
    size: (i32, i32, i32, i32), // -x, x, -z, z
    pub tiles: HashMap<(i32, i32), Tile>,
}

impl WorldTerrain {
    pub fn set_size(&mut self, size: (i32, i32, i32, i32)) {
        self.size = size;
    }
    pub fn set_heightmap(&mut self, tiles: HashMap<(i32, i32), Tile>) {
        self.tiles = tiles;
    }
    pub fn erode(&mut self, cycles: usize) {
        let mut rng = rand::thread_rng();
        let layout = HexLayout::pointy().with_hex_size(1.0);
        let evaporation_rate = 0.001;
        let deposition_rate = 0.1;
        let min_volume = 0.01;
        let max_age = 500;
        let gravity = 1.0;
        let mut total_change = 0.0;
        let total_change2 = 0.0;
        for i in 0..cycles {
            // println!("cycle {i}");
            let start_x = rng.gen_range(self.size.0..self.size.1);
            let start_z = rng.gen_range(self.size.2..self.size.3);
            // let start_x = rng.gen_range(-5..5);
            // let start_z = rng.gen_range(-5..5);
            let pos = (start_x, start_z);

            let Some(starting_tile) = self.tiles.get(&pos) else {
                continue;
            };

            if starting_tile.height() < 0.1 {
                continue;
            }

            let mut drop = Drop::new(0, 1.0, 0.0);
            drop.position = Vec2::new(starting_tile.position[0], starting_tile.position[2]);

            loop {
                let p = drop.position;
                let hex = layout.world_pos_to_hex(p);
                let cell_p = (hex.x, hex.y);
                let mut cell_hex = None;
                let mut cell_height = 0.0;
                let mut cell_position = Vec3::default();
                if let Some(cell) = self.tiles.get_mut(&cell_p) {
                    cell_hex = Some(cell.hex);
                    cell_height = cell.height();
                    cell_position = Vec3::from_array(cell.position);
                } else {
                    break;
                }

                let Some(hex) = cell_hex else { break };
                let mut speed_diff = Vec2::default();
                let mut max_hdiff: f32 = 0.0;
                let mut normal = Vec3::default();
                // neighbors
                // println!("CELL HEIGHT IS: {}", cell_height);
                // println!(
                //     "({}, {}, {})",
                //     cell_position[0], cell_position[1], cell_position[2]
                // );
                let mut i = 0;
                for n in hex.all_neighbors() {
                    if let Some(n) = self.tiles.get(&(n.x, n.y)) {
                        let angle: f32 = i as f32 * PI / 3.0; // 60 degrees in radians
                        let height = n.height();
                        // println!("NEIGH {i} IS: {:?}", n.position);
                        let h_diff = cell_height - height;

                        normal += (1.0 / 6.0)
                            * Vec3::new(angle.cos() * h_diff, 1.0, angle.sin() * h_diff)
                                .normalize();
                        max_hdiff = max_hdiff.max(h_diff.abs());
                        let pos = n.position;
                        // println!("({}, {}, {})", n.position[0], n.position[1], n.position[2]);

                        speed_diff += Vec2::new(pos[0] - p.x, pos[2] - p.y) * h_diff;
                    }
                    i += 1;
                }

                // normal = normal.normalize();
                // println!("NORMAL IS: ({}, {}, {})", normal[0], normal[1], normal[2]);
                // break;

                drop.speed += Vec2::new(normal.x, normal.z) / (drop.volume);
                drop.position += drop.speed.normalize();
                drop.speed *= 0.95;

                let next_hex = layout.world_pos_to_hex(drop.position);
                let next_cell_p = (next_hex.x, next_hex.y);
                let next_height;
                if let Some(cell) = self.tiles.get_mut(&next_cell_p) {
                    next_height = cell.height();
                } else {
                    break;
                }
                let Some(cell) = self.tiles.get_mut(&cell_p) else {
                    break;
                };
                // let hdiff2 = nex
                let mut max_sediment =
                    drop.volume * drop.speed.length() * (cell_height - next_height);
                max_sediment = max_sediment.max(0.0);

                let sdiff = max_sediment - drop.sediment;
                // sdiff *= 2.0;
                drop.sediment += deposition_rate * sdiff;
                let change = drop.volume * deposition_rate * sdiff;
                total_change += change;
                cell.position[1] -= change;
                drop.volume *= 1.0 - evaporation_rate;
                drop.age += 1;
                if drop.volume < min_volume {
                    cell.position[1] += drop.sediment;
                    break;
                }
                if drop.age > max_age {
                    cell.position[1] += drop.sediment;
                    break;
                }
            }

            // let start_z = rng.gen_range()
            // self.
        }
        println!("TOTAL CHANGE: {total_change}");
    }
}

pub fn erode(time: Res<Time>, mut timer: ResMut<EventTimer>, data: ResMut<MapData>) {
    if time.elapsed().as_secs() > 7 {
        return;
    }
    if timer.field1.tick(time.delta()).just_finished() {
        let mut tiles = HashMap::new();
        {
            let map = data.0.read_blocking();
            tiles = map.tiles.clone();
        }

        let handler = thread::spawn(|| {
            // thread code
            let mut world_terrain = WorldTerrain::default();
            let s = MAP_SIZE;
            world_terrain.set_size((s[0], s[1], s[2], s[3]));
            world_terrain.set_heightmap(tiles);
            println!("ERODING");
            world_terrain.erode(90_000);
            println!("ERODING DONE");
            return world_terrain.tiles;
        });

        let tiles = handler.join().unwrap();
        let mut map = data.0.write_blocking();
        let mut altered = Vec::new();
        for (pos, _) in &tiles {
            altered.push(*pos);
        }
        map.tiles = tiles;
        map.changed = VecDeque::from(altered);
    }
}
