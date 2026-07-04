use azalea_block::BlockState;
use azalea_core::position::{
    ChunkPos, ChunkSectionBiomePos, ChunkSectionBlockPos, ChunkSectionPos,
};
use azalea_registry::data::Biome;
use azalea_world::Chunk;
use parking_lot::{RwLock, RwLockReadGuard};

use crate::world::chunk::{ChunkLightData, ChunkStore};

pub struct LocalSection {
    pub blocks: [[[BlockState; 18]; 18]; 18],
    pub biomes: [[[Biome; 6]; 6]; 6],
    pub light: [[[u8; 18]; 18]; 18],
}

impl LocalSection {
    pub fn new(chunk_store: &ChunkStore,  spos: ChunkSectionPos) -> Self {
        let mut section = Self {
            blocks: [[[BlockState::AIR; 18]; 18]; 18],
            biomes: [[[Biome::default(); 6]; 6]; 6],
            light: [[[0; 18]; 18]; 18],
        };

        section.build(chunk_store, spos);

        section
    }

    pub fn new_boxed(
        chunk_store: &ChunkStore,
        spos: ChunkSectionPos,
    ) -> Box<Self> {
        let mut section = Box::new(Self {
            blocks: [[[BlockState::AIR; 18]; 18]; 18],
            biomes: [[[Biome::default(); 6]; 6]; 6],
            light: [[[0; 18]; 18]; 18],
        });

        section.build(chunk_store,  spos);

        section
    }

    #[inline]
    fn build(&mut self, chunk_store: &ChunkStore, spos: ChunkSectionPos) {
        let min_y = chunk_store.min_y();
        let base_y = spos.y - min_y.div_euclid(16);

        let arcs: [[Option<std::sync::Arc<RwLock<Chunk>>>; 3]; 3] = std::array::from_fn(|x| {
            std::array::from_fn(|z| {
                let pos = ChunkPos {
                    x: spos.x + (x as i32) - 1,
                    z: spos.z + (z as i32) - 1,
                };
                chunk_store.get_chunk(&pos)
            })
        });

        let light_grid: [[Option<&ChunkLightData>; 3]; 3] = std::array::from_fn(|x| {
            std::array::from_fn(|z| {
                let pos = ChunkPos {
                    x: spos.x + (x as i32) - 1,
                    z: spos.z + (z as i32) - 1,
                };
                chunk_store.light_data.get(&(pos.x, pos.z))
            })
        });

        let guards: [[Option<RwLockReadGuard<Chunk>>; 3]; 3] =
            std::array::from_fn(|x| std::array::from_fn(|z| arcs[x][z].as_ref().map(|c| c.read())));

        for lx in -1i32..17 {
            for ly in -1i32..17 {
                for lz in -1i32..17 {
                    let chunk = guards[(lx.div_euclid(16) + 1) as usize]
                        [(lz.div_euclid(16) + 1) as usize]
                        .as_deref();
                    let section_y = base_y + ly.div_euclid(16);
                    self.blocks[(lx + 1) as usize][(ly + 1) as usize][(lz + 1) as usize] = chunk
                        .and_then(|c| {
                            if section_y >= 0 && section_y < c.sections.len() as i32 {
                                c.sections.get(section_y as usize)
                            } else {
                                None
                            }
                        })
                        .map(|s| {
                            s.get_block_state(ChunkSectionBlockPos {
                                x: lx.rem_euclid(16) as u8,
                                y: ly.rem_euclid(16) as u8,
                                z: lz.rem_euclid(16) as u8,
                            })
                        }).unwrap_or_default();
                }
            }
        }

        for bx in -1i32..5 {
            for by in -1i32..5 {
                for bz in -1i32..5 {
                    let chunk = guards[(bx.div_euclid(4) + 1) as usize]
                        [(bz.div_euclid(4) + 1) as usize]
                        .as_deref();
                    let section_y = base_y + by.div_euclid(4);
                    self.biomes[(bx + 1) as usize][(by + 1) as usize][(bz + 1) as usize] = chunk
                        .and_then(|c| {
                            if section_y >= 0 && section_y < c.sections.len() as i32 {
                                c.sections.get(section_y as usize)
                            } else {
                                None
                            }
                        })
                        .map(|s| {
                            s.get_biome(ChunkSectionBiomePos {
                                x: bx.rem_euclid(4) as u8,
                                y: by.rem_euclid(4) as u8,
                                z: bz.rem_euclid(4) as u8,
                            })
                        })
                        .unwrap_or_default();
                }
            }
        }

        for lx in -1i32..17 {
            for ly in -1i32..17 {
                for lz in -1i32..17 {
                    let cx_off = lx.div_euclid(16);
                    let cz_off = lz.div_euclid(16);
                    let ax = (cx_off + 1) as usize;
                    let az = (cz_off + 1) as usize;

                    self.light[(lx + 1) as usize][(ly + 1) as usize][(lz + 1) as usize] =
                        if let Some(ld) = &light_grid[ax][az] {
                            let local_x = lx.rem_euclid(16);
                            let abs_y = spos.y * 16 + ly;
                            let local_z = lz.rem_euclid(16);

                            ld.get_sky_light(local_x, abs_y, local_z)
                                .max(ld.get_block_light(local_x, abs_y, local_z))
                        } else {
                            15
                        }
                }
            }
        }
    }
    /// Gets a block state at local coordinates (-1..17).
    #[inline]
    pub fn get_block_state(&self, x: i32, y: i32, z: i32) -> BlockState {
        let ix = x + 1;
        let iy = y + 1;
        let iz = z + 1;
        if (ix as u32) < 18 && (iy as u32) < 18 && (iz as u32) < 18 {
            self.blocks[ix as usize][iy as usize][iz as usize]
        } else {
            BlockState::AIR
        }
    }

    /// Gets light data at local block coordinates (-1..17).
    #[inline]
    pub fn get_light(&self, x: i32, y: i32, z: i32) -> u8 {
        let ix = x + 1;
        let iy = y + 1;
        let iz = z + 1;
        if (ix as u32) < 18 && (iy as u32) < 18 && (iz as u32) < 18 {
            self.light[ix as usize][iy as usize][iz as usize]
        } else {
            0
        }
    }

    /// Gets a biome at local block coordinates (-1..17).
    #[inline]
    pub fn get_biome(&self, x: i32, y: i32, z: i32) -> Biome {
        let bx = x.div_euclid(4) + 1;
        let by = y.div_euclid(4) + 1;
        let bz = z.div_euclid(4) + 1;
        if (bx as u32) < 6 && (by as u32) < 6 && (bz as u32) < 6 {
            self.biomes[bx as usize][by as usize][bz as usize]
        } else {
            Biome::default()
        }
    }
}

