//! Shared block/face records derived from master's SectionCuboid and
//! FaceRecord. LOD belongs to the draw batch, not to an individual cuboid.

pub const CUBOIDS_PER_WINDOW: u16 = 682;
pub const MAX_LOD: u32 = 4;

/// Low word first; shaders can decode this without 64-bit integer support.
/// Bits 52..63 are reserved and always zero. Tint is RGB UNORM8.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockRecord {
    pub words: [u32; 2],
}

impl BlockRecord {
    pub fn new(position: [u8; 3], global_id: u16, tint: [u8; 3]) -> Self {
        assert!(position.into_iter().all(|axis| axis < 16));
        let packed = u64::from(position[0])
            | (u64::from(position[1]) << 4)
            | (u64::from(position[2]) << 8)
            | (u64::from(global_id) << 12)
            | (u64::from(tint[0]) << 28)
            | (u64::from(tint[1]) << 36)
            | (u64::from(tint[2]) << 44);
        Self {
            words: [packed as u32, (packed >> 32) as u32],
        }
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn position(self) -> [u8; 3] {
        std::array::from_fn(|axis| ((self.words[0] >> (axis * 4)) & 15) as u8)
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn global_id(self) -> u16 {
        ((self.words[0] >> 12) & 0xffff) as u16
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn tint(self) -> [u8; 3] {
        [
            ((self.words[0] >> 28) | (self.words[1] << 4)) as u8,
            (self.words[1] >> 4) as u8,
            (self.words[1] >> 12) as u8,
        ]
    }
}

/// A window-local cuboid/face reference followed by four UNORM5 shades.
/// Direction order: down, up, north, south, west, east.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FaceRecord {
    pub packed: u32,
}

impl FaceRecord {
    pub fn new(cuboid: u16, direction: u8, shades: [u8; 4]) -> Self {
        assert!(cuboid < CUBOIDS_PER_WINDOW);
        assert!(direction < 6);
        assert!(shades.into_iter().all(|shade| shade < 32));
        let mut packed = u32::from(cuboid) * 6 + u32::from(direction);
        for (corner, shade) in shades.into_iter().enumerate() {
            packed |= u32::from(shade) << (12 + corner * 5);
        }
        Self { packed }
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn cuboid(self) -> u16 {
        ((self.packed & 0xfff) / 6) as u16
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn direction(self) -> u8 {
        ((self.packed & 0xfff) % 6) as u8
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn shades(self) -> [u8; 4] {
        std::array::from_fn(|corner| ((self.packed >> (12 + corner * 5)) & 31) as u8)
    }
}

/// One entry per fluid cuboid. Corner order is low-X/low-Z, low-X/high-Z,
/// high-X/high-Z, high-X/low-Z. Upper 16 bits are reserved and zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FluidHeights {
    pub packed: u32,
}

impl FluidHeights {
    pub fn new(heights: [f32; 4]) -> Self {
        let mut packed = 0;
        for (corner, height) in heights.into_iter().enumerate() {
            assert!(height.is_finite());
            packed |= ((height.clamp(0.0, 1.0) * 15.0).round() as u32) << (corner * 4);
        }
        Self { packed }
    }

    #[cfg(any(test, feature = "cpu-bench"))]
    #[allow(dead_code)]
    pub fn heights(self) -> [f32; 4] {
        std::array::from_fn(|corner| ((self.packed >> (corner * 4)) & 15) as f32 / 15.0)
    }
}

/// The batch's scale; section-local block origins are never scaled.
pub fn lod_scale(lod: u32) -> u32 {
    assert!(lod <= MAX_LOD);
    1 << lod
}

/// Shared geometry/material data. LOD batches use only UVs and sprite IDs;
/// cube positions are generated in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalFace {
    pub positions: [[f32; 4]; 4],
    pub uvs: [[f32; 4]; 4],
    pub material: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalCuboid {
    pub faces: [GlobalFace; 6],
}

/// Master's immutable baked-element table. Constructed once per asset/atlas
/// load, shared by the renderer and mesh workers; never rebuilt from a mesh.
pub struct GlobalCuboidTable {
    pub data: Vec<GlobalCuboid>,
    ids: std::collections::HashMap<u32, u16>,
    cubes: std::collections::HashMap<String, u16>,
    overlays: std::collections::HashMap<String, u16>,
    pub missing_id: u16,
    fluids: [[[u16; 2]; 2]; 2],
}

impl GlobalCuboidTable {
    pub(crate) fn build(
        assets: &crate::world::block::render_assets::BlockRenderAssets,
        atlas: &super::atlas::AtlasUVMap,
    ) -> Self {
        use bytemuck::Zeroable;

        use crate::world::block::model::Direction;
        let mut data = Vec::new();
        let mut dedup = std::collections::HashMap::new();
        let mut intern = |cuboid: GlobalCuboid| {
            let key = bytemuck::bytes_of(&cuboid);
            if let Some(&id) = dedup.get(key) {
                return id;
            }
            let id = u16::try_from(data.len()).expect("baked cuboid table exceeds u16 ID space");
            data.push(cuboid);
            dedup.insert(key.to_vec(), id);
            id
        };
        let mut baked = std::collections::BTreeMap::new();
        for quad in assets.all_baked_quads() {
            let cuboid = baked
                .entry(quad.cuboid_uid)
                .or_insert_with(GlobalCuboid::zeroed);
            let face = &mut cuboid.faces[quad.direction as usize];
            face.positions = quad.positions.map(|p| [p[0], p[1], p[2], 0.0]);
            face.uvs = quad
                .uvs
                .map(|uv| [uv[0].clamp(0.0, 1.0), uv[1].clamp(0.0, 1.0), 0.0, 0.0]);
            face.material[0] = u32::from(atlas.region(quad.texture).sprite);
        }
        let ids = baked
            .into_iter()
            .map(|(uid, geometry)| (uid, intern(geometry)))
            .collect();
        let cube = |sprites: [u16; 6]| {
            let mut cuboid = GlobalCuboid::zeroed();
            for direction in Direction::VALUES {
                let (positions, uvs) = super::mesher::cube_face_geometry(direction);
                cuboid.faces[direction as usize] = GlobalFace {
                    positions: positions.map(|p| [p[0], p[1], p[2], 0.0]),
                    uvs: uvs.map(|uv| [uv[0], uv[1], 0.0, 0.0]),
                    material: [u32::from(sprites[direction as usize]), 0, 0, 0],
                };
            }
            cuboid
        };
        let missing_id = intern(cube([atlas.missing_region().sprite; 6]));
        let mut cubes = std::collections::HashMap::new();
        let mut overlays = std::collections::HashMap::new();
        for (name, textures) in assets.all_face_textures() {
            let sprites = [
                &textures.bottom,
                &textures.top,
                &textures.north,
                &textures.south,
                &textures.west,
                &textures.east,
            ]
            .map(|texture| atlas.get_region(texture).sprite);
            cubes.insert(name.to_owned(), intern(cube(sprites)));
            if let Some(overlay) = &textures.side_overlay {
                overlays.insert(
                    name.to_owned(),
                    intern(cube([atlas.get_region(overlay).sprite; 6])),
                );
            }
        }
        // Only fixed fluid templates are needed: ordinary/backward top winding
        // and the two bottom inset cases. Surface heights stay in mesh data.
        let mut fluids = [[[0; 2]; 2]; 2];
        for (kind_fluids, texture) in fluids.iter_mut().zip(["water_still", "lava_still"]) {
            for (inset, inset_fluids) in kind_fluids.iter_mut().enumerate() {
                for (backward, fluid_id) in inset_fluids.iter_mut().enumerate() {
                    let mut geometry = cube([atlas.get_region(texture).sprite; 6]);
                    for direction in Direction::VALUES {
                        let face = &mut geometry.faces[direction as usize];
                        for (corner, p) in face.positions.iter_mut().enumerate() {
                            let top = direction == Direction::Up
                                || (direction != Direction::Down && (corner == 0 || corner == 3));
                            p[3] = if top { 1.0 } else { 0.0 };
                            if !top {
                                p[1] = if inset != 0 {
                                    super::mesher::FLUID_INSET
                                } else {
                                    0.0
                                };
                            }
                            match direction {
                                Direction::North => p[2] = super::mesher::FLUID_INSET,
                                Direction::South => p[2] = 1.0 - super::mesher::FLUID_INSET,
                                Direction::West => p[0] = super::mesher::FLUID_INSET,
                                Direction::East => p[0] = 1.0 - super::mesher::FLUID_INSET,
                                _ => {}
                            }
                        }
                    }
                    if backward != 0 {
                        let top = geometry.faces[Direction::Up as usize];
                        geometry.faces[Direction::Down as usize] = GlobalFace {
                            positions: [0, 3, 2, 1].map(|i| top.positions[i]),
                            uvs: [0, 3, 2, 1].map(|i| top.uvs[i]),
                            material: top.material,
                        };
                    }
                    *fluid_id = intern(geometry);
                }
            }
        }
        Self {
            data,
            ids,
            cubes,
            overlays,
            missing_id,
            fluids,
        }
    }

    pub fn id(&self, uid: u32) -> u16 {
        self.ids[&uid]
    }
    pub fn cube_id(&self, name: &str) -> u16 {
        self.cubes.get(name).copied().unwrap_or(self.missing_id)
    }
    pub fn overlay_id(&self, name: &str) -> u16 {
        self.overlays.get(name).copied().unwrap_or(self.missing_id)
    }
    pub fn fluid_id(&self, lava: bool, inset: bool, backward: bool) -> u16 {
        self.fluids[usize::from(lava)][usize::from(inset)][usize::from(backward)]
    }
}

/// All cuboid IDs already reference the shared baked table.
#[derive(Default)]
pub struct PackedMesh {
    pub blocks: Vec<BlockRecord>,
    pub faces: Vec<FaceRecord>,
    /// One batch per reference window. All faces in a batch share its LOD.
    pub batches: Vec<FaceBatch>,
    pub heights: Vec<FluidHeights>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FaceBatch {
    pub first_face: u32,
    pub face_count: u32,
    pub first_block: u32,
    pub lod: u32,
    /// Fluid height-array offset; for solids, model position offset Z bits.
    pub first_height: u32,
    pub fluid: u32,
    /// Solids' model position offset X/Y bits; zero for fluids.
    pub offset_xy: [u32; 2],
}

impl PackedMesh {
    /// Fourteen low bits address the four corners of at most 4092 faces in
    /// a 682-cuboid window. High bits select its batch directly in the shader.
    /// CPU mesh indices stay linear for geometry hashes and CPU benchmarks.
    pub fn draw_indices(&self, opaque: &[u32], water: &[u32]) -> (Vec<u32>, Vec<u32>) {
        assert!(self.batches.len() <= 65536);
        let mut face_vertices = vec![0u32; self.faces.len()];
        for (batch_index, batch) in self.batches.iter().enumerate() {
            assert!(batch.face_count <= u32::from(CUBOIDS_PER_WINDOW) * 6);
            for face in 0..batch.face_count {
                face_vertices[(batch.first_face + face) as usize] =
                    ((batch_index as u32) << 14) | (face << 2);
            }
        }
        let encode = |indices: &[u32]| {
            indices
                .iter()
                .map(|&vertex| face_vertices[(vertex / 4) as usize] | (vertex & 3))
                .collect()
        };
        (encode(opaque), encode(water))
    }

    /// Header and encoded batch offsets are u32-word offsets relative to this
    /// section's base. Resolve stream offsets here to avoid dependent header
    /// reads for every GPU vertex. CPU batches retain their element indices.
    pub fn encode(&self) -> Vec<u8> {
        let mut words = vec![0u32; 8];
        words[0] = self.batches.len() as u32;
        words[1] = words.len() as u32;
        words.extend_from_slice(bytemuck::cast_slice(&self.batches));
        words[2] = words.len() as u32;
        for block in &self.blocks {
            words.extend_from_slice(&block.words);
        }
        words[3] = words.len() as u32;
        words.extend(self.faces.iter().map(|face| face.packed));
        words[4] = words.len() as u32;
        words.extend(self.heights.iter().map(|height| height.packed));
        for (index, batch) in self.batches.iter().enumerate() {
            let offset = 8 + index * 8;
            words[offset] = words[3] + batch.first_face;
            words[offset + 2] = words[2] + batch.first_block * 2;
            if batch.fluid != 0 {
                words[offset + 4] = words[4] + batch.first_height;
            }
        }
        // The arena's legacy vertex base uses a 16-byte stride. Padding keeps
        // that base exact while all actual geometry is fetched from storage.
        words.resize(words.len().next_multiple_of(4), 0);
        bytemuck::cast_slice(&words).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_offsets_and_baked_ids_match_the_gpu_layout() {
        assert_eq!(size_of::<GlobalFace>(), 144);
        assert_eq!(size_of::<GlobalCuboid>(), 864);
        assert_eq!(size_of::<FaceBatch>(), 32);
        let mesh = PackedMesh {
            blocks: vec![BlockRecord::new([3, 4, 5], 65535, [17, 34, 255])],
            faces: vec![FaceRecord::new(0, 1, [1, 7, 19, 31])],
            batches: vec![FaceBatch {
                first_face: 0,
                face_count: 1,
                first_block: 0,
                lod: 2,
                first_height: 0,
                fluid: 1,
                offset_xy: [0; 2],
            }],
            heights: vec![FluidHeights::new([1.0; 4])],
        };
        let bytes = mesh.encode();
        assert_eq!(bytes.len() % 16, 0);
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_ne_bytes(*word))
            .collect();
        assert_eq!(&words[..8], &[1, 8, 16, 18, 19, 0, 0, 0]);
        assert_eq!(&words[8..16], &[18, 1, 16, 2, 19, 1, 0, 0]);
        let block = BlockRecord {
            words: [words[16], words[17]],
        };
        assert_eq!(block.position(), [3, 4, 5]);
        assert_eq!(block.global_id(), 65535);
        assert_eq!(block.tint(), [17, 34, 255]);
        assert_eq!(block.words[1] >> 20, 0);
        assert_eq!(words[18], mesh.faces[0].packed);
        assert_eq!(words[19], 0xffff);
    }

    #[test]
    fn batch_stream_pointers_rebase_and_keep_solid_model_offsets() {
        let z_bits = (-0.125f32).to_bits();
        let mesh = PackedMesh {
            blocks: vec![
                BlockRecord::new([1, 2, 3], 7, [255; 3]),
                BlockRecord::new([4, 5, 6], 11, [255; 3]),
            ],
            faces: vec![
                FaceRecord::new(0, 0, [31; 4]),
                FaceRecord::new(0, 3, [31; 4]),
            ],
            batches: vec![
                FaceBatch {
                    first_face: 0,
                    face_count: 1,
                    first_block: 0,
                    lod: 1,
                    first_height: 0,
                    fluid: 1,
                    offset_xy: [0; 2],
                },
                FaceBatch {
                    first_face: 1,
                    face_count: 1,
                    first_block: 1,
                    lod: 0,
                    first_height: z_bits,
                    fluid: 0,
                    offset_xy: [0; 2],
                },
            ],
            heights: vec![FluidHeights::new([1.0; 4])],
        };
        // A section can start anywhere in the shared arena. Batch pointers
        // must follow that base while solid offset bits stay uninterpreted.
        let base = 12;
        let mut words = vec![0u32; base];
        words.extend(
            mesh.encode()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_ne_bytes(*word)),
        );
        for index in 0..2 {
            let batch = base + 8 + index * 8;
            let face = words[base + words[batch] as usize];
            assert_eq!(face, mesh.faces[index].packed);
            let block_word = base + words[batch + 2] as usize;
            let block = BlockRecord {
                words: [words[block_word], words[block_word + 1]],
            };
            assert_eq!(block.global_id(), [7, 11][index]);
        }
        assert_eq!(words[base + words[base + 8 + 4] as usize], 0xffff);
        assert_eq!(words[base + 16 + 4], z_bits);
        assert_eq!(mesh.batches[1].first_face, 1);
        assert_eq!(mesh.batches[1].first_block, 1);
    }

    #[test]
    fn block_layout_matches_shader_words_and_reserves_top_bits() {
        assert_eq!(size_of::<BlockRecord>(), 8);
        let record = BlockRecord::new([15, 7, 3], 0xabcd, [0x12, 0x34, 0xff]);
        assert_eq!(record.position(), [15, 7, 3]);
        assert_eq!(record.global_id(), 0xabcd);
        assert_eq!(record.tint(), [0x12, 0x34, 0xff]);
        assert_eq!(record.words[0], 0x2abcd37f);
        assert_eq!(record.words[1], 0x000ff341);
        assert_eq!(record.words[1] >> 20, 0);
    }

    #[test]
    fn all_window_references_and_directions_are_distinct() {
        assert_eq!(size_of::<FaceRecord>(), 4);
        for cuboid in 0..CUBOIDS_PER_WINDOW {
            for direction in 0..6 {
                let record = FaceRecord::new(cuboid, direction, [0, 1, 16, 31]);
                assert_eq!(record.cuboid(), cuboid);
                assert_eq!(record.direction(), direction);
                assert_eq!(record.shades(), [0, 1, 16, 31]);
                assert!(record.packed & 0xfff < 4092);
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_reference_outside_window() {
        FaceRecord::new(CUBOIDS_PER_WINDOW, 0, [0; 4]);
    }

    #[test]
    fn fluid_height_precision_is_unchanged() {
        assert_eq!(size_of::<FluidHeights>(), 4);
        let record = FluidHeights::new([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
        assert_eq!(record.packed, 0xfa50);
        assert_eq!(record.heights(), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
        for lod in 0..=MAX_LOD {
            assert_eq!(lod_scale(lod), 1 << lod);
        }
    }
}
