use art::CustomMaterial;
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

pub const HEX_RADIUS: f32 = 1.0;
pub const MAP_SIZE: [i32; 4] = [-200, 200, -200, 200];
pub const SHARE_VERTICES: bool = false;
// const MAP_SIZE: [i32; 4] = [-300, 300, -300, 300];
// const MAP_SIZE: [i32; 4] = [-400, 400, -400, 400];
// const MAP_SIZE: [i32; 4] = [-100, 100, -100, 100];
// const MAP_SIZE: [i32; 4] = [-150, 150, -150, 150];
// const MAP_SIZE: [i32; 4] = [-50, 50, -50, 50];

mod art;
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
        .add_systems(Startup, terrain::generate_map)
        .add_systems(Startup, setup_lighting)
        .add_systems(Update, move_player)
        .add_systems(Startup, setup_camera)
        .add_systems(Update, toggle_wireframe)
        .run();
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
