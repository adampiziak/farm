use bevy::{
    pbr::MaterialExtension,
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
        },
    },
};
pub const SHADER_ASSET_PATH: &str = "hexagon_shader.wgsl";
pub const EXTENDED_SHADER_ASSET_PATH: &str = "extended_material.wgsl";

// This struct defines the data that will be passed to your shader
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct CustomMaterial {
    #[uniform(100)]
    pub(crate) color: LinearRgba,
    #[uniform(101)]
    pub(crate) mod_color: LinearRgba,
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

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct MyExtension {
    // We need to ensure that the bindings of the base material and the extension do not conflict,
    // so we start from binding slot 100, leaving slots 0-99 for the base material.
    #[uniform(100)]
    pub quantize_steps: u32,
    #[uniform(101)]
    pub(crate) color: LinearRgba,
    #[texture(102)]
    #[sampler(103)]
    pub color_texture: Option<Handle<Image>>,
    #[texture(104)]
    #[sampler(105)]
    pub color2_texture: Option<Handle<Image>>,
    #[texture(106)]
    #[sampler(107)]
    pub mountain_texture: Option<Handle<Image>>,
    #[texture(108)]
    #[sampler(109)]
    pub mountain_normals: Option<Handle<Image>>,
}

impl MaterialExtension for MyExtension {
    fn fragment_shader() -> ShaderRef {
        EXTENDED_SHADER_ASSET_PATH.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        EXTENDED_SHADER_ASSET_PATH.into()
    }
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
}
