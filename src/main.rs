use art::CustomMaterial;
use bevy::prelude::*;

use bevy::dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin};
use bevy::pbr::ExtendedMaterial;
use bevy::{
    pbr::wireframe::{Wireframe, WireframePlugin},
    render::{
        settings::{WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
    text::FontSmoothing,
};
use misc::MapData;

use noisy_bevy::NoisyShaderPlugin;

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
