use art::{CustomMaterial, MyExtension};
use bevy::color::palettes::css::{ANTIQUE_WHITE, GHOST_WHITE, RED, WHITE};
use bevy::image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
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
use rand::{thread_rng, Rng};
use terrain::TreeThingy;

pub const HEX_RADIUS: f32 = 1.0;
// pub const MAP_SIZE: [i32; 4] = [-200, 200, -200, 200];
// pub const MAP_SIZE: [i32; 4] = [-100, 100, -100, 100];
pub const SHARE_VERTICES: bool = false;
// const MAP_SIZE: [i32; 4] = [-300, 300, -300, 300];
// pub const MAP_SIZE: [i32; 4] = [-400, 400, -400, 400];
// const MAP_SIZE: [i32; 4] = [-150, 150, -150, 150];
// const MAP_SIZE: [i32; 4] = [-50, 50, -50, 50];
// const MAP_SIZE: [i32; 4] = [-30, 30, -30, 30];
const MAP_SIZE: [i32; 4] = [-20, 20, -20, 20];
// const MAP_SIZE: [i32; 4] = [-80, 80, -80, 80];

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
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, MyExtension>,
        >::default())
        .add_systems(Startup, terrain::generate_map)
        .add_systems(Startup, setup_lighting)
        // .add_systems(Startup, setup_cube)
        .add_systems(Update, move_player)
        .add_systems(Startup, setup_camera)
        .add_systems(Update, toggle_wireframe)
        .add_systems(Update, tree_visible)
        .run();
}

fn setup_cube(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    asset_server: Res<AssetServer>,
) {
    let normal_handle = asset_server.load_with_settings(
        "textures/cube_normal.png",
        // The normal map texture is in linear color space. Lighting won't look correct
        // if `is_srgb` is `true`, which is the default.
        |settings: &mut ImageLoaderSettings| settings.is_srgb = false,
    );
    let parallax_depth_scale = 0.2;
    let max_parallax_layer_count = ops::exp2(5.0);
    let parallax_mapping_method = ParallaxMappingMethod::Occlusion;

    let parallax_material = materials.add(StandardMaterial {
        perceptual_roughness: 0.4,
        base_color_texture: Some(asset_server.load("textures/cube_color.png")),
        normal_map_texture: Some(normal_handle),
        // The depth map is a grayscale texture where black is the highest level and
        // white the lowest.
        // depth_map: Some(asset_server.load("textures/parallax_example/cube_depth.png")),
        // depth_map: Some(asset_server.load("textures/mountain_displacement.png")),
        depth_map: Some(asset_server.load_with_settings(
            // "textures/grass01.png",
            "textures/cube_depth.png",
            |s: &mut _| {
                *s = ImageLoaderSettings {
                    sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                        // rewriting mode to repeat image,
                        address_mode_u: ImageAddressMode::Repeat,
                        address_mode_v: ImageAddressMode::Repeat,
                        ..default()
                    }),
                    ..default()
                }
            },
        )),
        parallax_depth_scale,
        parallax_mapping_method,
        max_parallax_layer_count,
        ..default()
    });
    commands.spawn((
        Mesh3d(
            meshes.add(
                // NOTE: for normal maps and depth maps to work, the mesh
                // needs tangents generated.
                Mesh::from(Cuboid::default())
                    .with_generated_tangents()
                    .unwrap(),
            ),
        ),
        MeshMaterial3d(parallax_material.clone()),
        Transform::from_xyz(0.0, 20.0, 0.0),
    ));
}

/// When everything is ready, un-hide the game map
fn tree_visible(
    mut query: Query<(&mut Visibility, &TreeThingy)>,
    player: Query<&Transform, With<Player>>,
) {
    // let mut rng = thread_rng();
    // let mut i = 0;
    let rad = 62.0;
    if let Ok(ply) = player.get_single() {
        let mut player_pos = ply.translation.xz();
        player_pos[1] -= rad * 1.0;
        player_pos[0] += 10.0;

        // ply.translation
        for (mut vis, thingy) in &mut query {
            if thingy.position.xz().distance(player_pos) < rad {
                // if thingy
                // let num = rng.gen_range(0_u32..100);

                *vis = Visibility::Visible;
            } else {
                *vis = Visibility::Hidden;
            }
        }
    }
    // let num = rng.gen_range(0_u32..100);
    // let mut vis_map = query.get_many_mut(entities)
}

fn setup_lighting(mut commands: Commands) {
    // commands.insert_resource(AmbientLight {
    //     color: bevy::color::palettes::css::NAVAJO_WHITE.into(),

    //     brightness: 100.0,
    // });
    // commands.spawn((
    //     PointLight {
    //         intensity: 100_000.0,
    //         color: WHITE.into(),
    //         shadows_enabled: true,
    //         range: 100.0,

    //         ..default()
    //     },
    //     Transform::from_xyz(0.0, 40.0, 0.0),
    // ));
    // commands.spawn((
    //     PointLight {
    //         intensity: 1_000_000.0,
    //         color: WHITE.into(),
    //         shadows_enabled: true,
    //         range: 100.0,
    //         ..default()
    //     },
    //     Transform::from_xyz(1.0, 10.0, 0.0),
    // ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,

            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 300.0, 0.0).looking_to(
            Vec3 {
                x: -0.2,
                y: -0.16,
                z: 0.2,
            },
            Vec3::Y,
        ),
    ));
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

    let step = 0.55;
    if input.pressed(KeyCode::KeyW) {
        transform.translation = Vec3 {
            z: translation.z - step,
            ..translation
        };
    }
    let rotate_step = 0.01;
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
            Transform::from_xyz(4., 30.0, 100.0),
            Visibility::default(),
        ))
        .with_children(|parent| {
            // parent.spawn((WorldModelCamera,));

            // Spawn view model camera.
            parent.spawn((
                Camera3d::default(),
                Transform::from_xyz(10., 30., 10.).looking_to(
                    Vec3 {
                        x: 0.0,
                        y: -0.2,
                        z: -0.9,
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
