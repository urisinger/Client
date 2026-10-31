use crate::renderer::camera::CameraUniform;
use crate::renderer::chunk::atlas::TextureAtlas;
use crate::renderer::chunk::buffer::{CHUNK_VERTEX_BUFFERS, COMPACT_CHUNK_VERTEX_BUFFERS};
use crate::renderer::chunk::mesher::TerrainFormat;
use crate::renderer::pipelines::lightmap::Lightmap;
use crate::renderer::{rhi, shader, util};

/// Set 1: the atlas, its sampler and the sprite-rectangle buffer the fragment
/// shaders resolve sprite UVs with.
const ATLAS_LAYOUT: &[rhi::BindGroupLayoutEntry] = &[
    util::FRAGMENT_TEXTURE[0],
    util::FRAGMENT_TEXTURE[1],
    rhi::BindGroupLayoutEntry::storage(rhi::ShaderStages::FRAGMENT),
];

pub struct ChunkPipeline {
    pub pipeline_solid: rhi::backend::RenderPipeline,
    pub pipeline_cutout: rhi::backend::RenderPipeline,
    pub water_pipeline: rhi::backend::RenderPipeline,
    lod_pipelines: [rhi::backend::RenderPipeline; 3],
    compact_lod_pipelines: [rhi::backend::RenderPipeline; 3],
    pub mesh_layout: rhi::backend::BindGroupLayout,
    /// Compact opaque terrain: no discard, early-Z. Drawn first
    /// (front-to-back).
    pub compact_pipeline_solid: rhi::backend::RenderPipeline,
    /// Compact cutout terrain: alpha-test discard. Drawn after solid.
    pub compact_pipeline_cutout: rhi::backend::RenderPipeline,
    /// Compact translucent water: alpha blending on, depth write off. Shares
    /// the terrain pipelines' layout and bind groups.
    pub compact_water_pipeline: rhi::backend::RenderPipeline,
    pub pipeline_layout: rhi::backend::PipelineLayout,
    compact_pipeline_layout: rhi::backend::PipelineLayout,
    pub camera_layout: rhi::backend::BindGroupLayout,
    pub atlas_layout: rhi::backend::BindGroupLayout,
    pub camera_sets: Vec<rhi::backend::BindGroup>,
    pub atlas_set: rhi::backend::BindGroup,
    /// Set 2, which [`Lightmap`] owns.
    lightmap_set: rhi::backend::BindGroup,
    camera_buffers: Vec<rhi::backend::Buffer>,
}

impl ChunkPipeline {
    pub fn new(
        device: &rhi::backend::Device,
        targets: rhi::backend::PipelineTargets,
        atlas: &TextureAtlas,
        lightmap: &Lightmap,
    ) -> Self {
        let camera_layout = rhi::backend::create_bind_group_layout(device, util::VERTEX_UNIFORM);
        let atlas_layout = rhi::backend::create_bind_group_layout(device, ATLAS_LAYOUT);
        let mesh_layout = rhi::backend::create_bind_group_layout(
            device,
            &[
                rhi::BindGroupLayoutEntry::storage(rhi::ShaderStages::VERTEX),
                rhi::BindGroupLayoutEntry::storage(rhi::ShaderStages::VERTEX),
            ],
        );
        // Water pushes its section origin + fade per draw (water.vert); the
        // indirect passes read them per-instance and ignore the range.
        let pipeline_layout = rhi::backend::create_pipeline_layout(
            device,
            &[camera_layout, atlas_layout, lightmap.layout],
            Some(rhi::PushConstants {
                visibility: rhi::ShaderStages::VERTEX,
                size: 16,
            }),
        );
        let compact_pipeline_layout = rhi::backend::create_pipeline_layout(
            device,
            &[camera_layout, atlas_layout, lightmap.layout, mesh_layout],
            Some(rhi::PushConstants {
                visibility: rhi::ShaderStages::VERTEX,
                size: 32,
            }),
        );

        let [
            compact_pipeline_solid,
            compact_pipeline_cutout,
            compact_water_pipeline,
        ] = create_compact_pipelines(device, targets, compact_pipeline_layout, false);
        let compact_lod_pipelines =
            create_compact_pipelines(device, targets, compact_pipeline_layout, true);
        let [pipeline_solid, pipeline_cutout, water_pipeline] =
            create_pipelines(device, targets, pipeline_layout, false);
        let lod_pipelines = create_pipelines(device, targets, pipeline_layout, true);

        let (camera_sets, camera_buffers) = util::create_uniform_sets(
            device,
            camera_layout,
            size_of::<CameraUniform>() as u64,
            "camera_uniform",
        );

        let mut pipeline = Self {
            mesh_layout,
            pipeline_solid,
            pipeline_cutout,
            water_pipeline,
            lod_pipelines,
            compact_lod_pipelines,
            compact_pipeline_solid,
            compact_pipeline_cutout,
            compact_water_pipeline,
            pipeline_layout,
            compact_pipeline_layout,
            camera_layout,
            atlas_layout,
            camera_sets,
            atlas_set: rhi::backend::BindGroup::default(),
            lightmap_set: lightmap.group,
            camera_buffers,
        };
        pipeline.rebind_atlas(device, atlas);
        pipeline
    }

    pub fn update_camera(&mut self, frame: usize, uniform: &CameraUniform) {
        let bytes = bytemuck::bytes_of(uniform);
        self.camera_buffers[frame].write(bytes);
    }

    pub fn rebind_atlas(&mut self, device: &rhi::backend::Device, atlas: &TextureAtlas) {
        let [texture, sampler] = util::texture_binding(atlas.texture.handle(), atlas.sampler);
        self.atlas_set.replace(
            device,
            self.atlas_layout,
            &[
                texture,
                sampler,
                (
                    2,
                    rhi::backend::BindingResource::storage(
                        &atlas.sprite_rects,
                        atlas.sprite_rects_bytes(),
                    ),
                ),
            ],
        );
    }

    pub fn bind(
        &self,
        pass: &mut rhi::backend::RenderPassEncoder,
        frame: usize,
        cutout: bool,
        format: TerrainFormat,
        has_lods: bool,
    ) {
        if format == TerrainFormat::Vanilla {
            if has_lods {
                self.bind_with(pass, self.lod_pipelines[usize::from(cutout)], frame);
            } else {
                self.bind_moving(pass, frame, cutout);
            }
            return;
        }
        let pipeline = if has_lods {
            self.compact_lod_pipelines[usize::from(cutout)]
        } else if cutout {
            self.compact_pipeline_cutout
        } else {
            self.compact_pipeline_solid
        };
        self.bind_with(pass, pipeline, frame);
    }

    /// Bind the translucent water pipeline (same bind groups as `bind`).
    pub fn bind_water(
        &self,
        pass: &mut rhi::backend::RenderPassEncoder,
        frame: usize,
        format: TerrainFormat,
        has_lods: bool,
    ) {
        self.bind_with(
            pass,
            if format == TerrainFormat::Vanilla {
                if has_lods {
                    self.lod_pipelines[2]
                } else {
                    self.water_pipeline
                }
            } else if has_lods {
                self.compact_lod_pipelines[2]
            } else {
                self.compact_water_pipeline
            },
            frame,
        );
    }

    pub fn bind_moving(
        &self,
        pass: &mut rhi::backend::RenderPassEncoder,
        frame: usize,
        cutout: bool,
    ) {
        self.bind_with(
            pass,
            if cutout {
                self.pipeline_cutout
            } else {
                self.pipeline_solid
            },
            frame,
        );
    }

    fn bind_with(
        &self,
        pass: &mut rhi::backend::RenderPassEncoder,
        pipeline: rhi::backend::RenderPipeline,
        frame: usize,
    ) {
        pass.set_pipeline(&pipeline);
        pass.set_bind_groups(
            0,
            &[self.camera_sets[frame], self.atlas_set, self.lightmap_set],
        );
    }

    pub fn destroy(&mut self, device: &rhi::backend::Device) {
        util::destroy_buffers(device, &mut self.camera_buffers);

        self.compact_pipeline_solid.destroy(device);
        self.compact_pipeline_cutout.destroy(device);
        self.compact_water_pipeline.destroy(device);
        self.pipeline_solid.destroy(device);
        self.pipeline_cutout.destroy(device);
        self.water_pipeline.destroy(device);
        for pipeline in &mut self.lod_pipelines {
            pipeline.destroy(device);
        }
        for pipeline in &mut self.compact_lod_pipelines {
            pipeline.destroy(device);
        }
        self.pipeline_layout.destroy(device);
        self.compact_pipeline_layout.destroy(device);
        util::destroy_bind_groups(device, self.camera_sets.drain(..));
        self.atlas_set.destroy(device);
        self.camera_layout.destroy(device);
        self.atlas_layout.destroy(device);
        self.mesh_layout.destroy(device);
    }
}

/// Builds the original vertex-input chunk pipelines: `solid` (no discard,
/// early-Z), `cutout` (chunk.frag, alpha-test discard) and translucent
/// `water`. Water reads only the packed vertices; the terrain pipelines also
/// read the per-instance meta.
fn create_pipelines(
    device: &rhi::backend::Device,
    targets: rhi::backend::PipelineTargets,
    layout: rhi::backend::PipelineLayout,
    lod: bool,
) -> [rhi::backend::RenderPipeline; 3] {
    let (vertex, water_vertex, fragments): (&[u8], &[u8], [&[u8]; 3]) = if lod {
        (
            shader::include_shader!("lod_chunk.vert"),
            shader::include_shader!("lod_water.vert"),
            [
                shader::include_shader!("lod_chunk_solid.frag"),
                shader::include_shader!("lod_chunk.frag"),
                shader::include_shader!("lod_water.frag"),
            ],
        )
    } else {
        (
            shader::include_shader!("chunk.vert"),
            shader::include_shader!("water.vert"),
            [
                shader::include_shader!("chunk_solid.frag"),
                shader::include_shader!("chunk.frag"),
                shader::include_shader!("water.frag"),
            ],
        )
    };
    let terrain = |label, fragment| rhi::RenderPipelineDesc {
        cull_mode: rhi::CullMode::Back,
        depth: Some(rhi::DepthState::WRITE),
        ..rhi::RenderPipelineDesc::new(label, vertex, fragment, &CHUNK_VERTEX_BUFFERS)
    };
    [
        terrain("solid chunk", fragments[0]),
        terrain("cutout chunk", fragments[1]),
        rhi::RenderPipelineDesc {
            vertex_shader: water_vertex,
            vertex_buffers: &CHUNK_VERTEX_BUFFERS[..1],
            depth: Some(rhi::DepthState::READ_ONLY),
            blend: Some(rhi::BlendState::TRANSLUCENT),
            ..terrain("water", fragments[2])
        },
    ]
    .map(|desc| rhi::backend::create_render_pipeline(device, targets, layout, &desc))
}

fn create_compact_pipelines(
    device: &rhi::backend::Device,
    targets: rhi::backend::PipelineTargets,
    layout: rhi::backend::PipelineLayout,
    lod: bool,
) -> [rhi::backend::RenderPipeline; 3] {
    let fragments: [&[u8]; 3] = if lod {
        [
            shader::include_shader!("lod_chunk_solid.frag"),
            shader::include_shader!("lod_chunk.frag"),
            shader::include_shader!("lod_water.frag"),
        ]
    } else {
        [
            shader::include_shader!("chunk_solid.frag"),
            shader::include_shader!("chunk.frag"),
            shader::include_shader!("water.frag"),
        ]
    };
    let terrain = |label, fragment_shader| rhi::RenderPipelineDesc {
        cull_mode: rhi::CullMode::Back,
        depth: Some(rhi::DepthState::WRITE),
        ..rhi::RenderPipelineDesc::new(
            label,
            shader::include_shader!("compact_chunk.vert"),
            fragment_shader,
            &COMPACT_CHUNK_VERTEX_BUFFERS,
        )
    };
    let water = rhi::RenderPipelineDesc {
        vertex_shader: shader::include_shader!("compact_water.vert"),
        vertex_buffers: &COMPACT_CHUNK_VERTEX_BUFFERS[..1],
        depth: Some(rhi::DepthState::READ_ONLY),
        blend: Some(rhi::BlendState::TRANSLUCENT),
        ..terrain("compact water", fragments[2])
    };
    [
        terrain("compact solid chunk", fragments[0]),
        terrain("compact cutout chunk", fragments[1]),
        water,
    ]
    .map(|desc| rhi::backend::create_render_pipeline(device, targets, layout, &desc))
}

#[cfg(test)]
mod tests {
    use crate::renderer::shader;

    fn op_count(bytes: &[u8], opcode: u16) -> usize {
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let mut cursor = 5; // SPIR-V header
        let mut count = 0;
        while cursor < words.len() {
            let instruction = words[cursor];
            count += usize::from(instruction as u16 == opcode);
            let word_count = (instruction >> 16) as usize;
            assert!(word_count > 0);
            cursor += word_count;
        }
        count
    }

    #[test]
    fn original_fragment_shaders_do_not_pay_for_lod_sampling() {
        let original: [&[u8]; 3] = [
            shader::include_shader!("chunk.frag"),
            shader::include_shader!("chunk_solid.frag"),
            shader::include_shader!("water.frag"),
        ];
        let lod: [&[u8]; 3] = [
            shader::include_shader!("lod_chunk.frag"),
            shader::include_shader!("lod_chunk_solid.frag"),
            shader::include_shader!("lod_water.frag"),
        ];
        for (original, lod) in original.into_iter().zip(lod) {
            // OpImageSampleExplicitLod, OpDPdx, OpDPdy are only needed by
            // repeating LOD textures. Check the compiled shader, not its source.
            for opcode in [88, 207, 208] {
                assert_eq!(op_count(original, opcode), 0);
                assert!(op_count(lod, opcode) > 0);
            }
        }
    }
}
