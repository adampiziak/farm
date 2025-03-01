use bevy::prelude::*;

#[derive(Component)]
pub struct Terrain;

#[derive(Component)]
pub struct MainTerrain;

pub fn update_terrain() {
    println!("hey!")
}

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_terrain);
    }
}
