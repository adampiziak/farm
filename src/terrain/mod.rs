use std::hash::{Hash, Hasher};

use bevy::utils::HashSet;
use hexx::Hex;

pub mod erosion;

#[derive(Default, Clone)]
pub struct Tile {
    pub vertices: Vec<HexVertex>,
    pub hex: Hex,
    pub position: [f32; 3],
    pub chunk_index: usize,
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

#[derive(Default)]
pub struct Chunk {
    pub id: u32,
    pub vertex_set: HashSet<HexVertex>,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}
