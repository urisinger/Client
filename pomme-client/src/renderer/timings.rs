use super::rhi;
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timestamp {
    FrameStart = 0,
    FrameEnd,
    CullStart,
    CullEnd,
    GuiBakeStart,
    GuiBakeEnd,
    TerrainStart,
    TerrainEnd,
    EntitiesStart,
    EntitiesEnd,
    TranslucentStart,
    TranslucentEnd,
    UiStart,
    UiEnd,
    Count, // Automatically tracks the total number of timestamps needed
}
#[derive(Debug, Default, Clone, Copy)]
pub struct RenderTimings {
    // CPU waits retained for benchmark reporting and frame pacing.
    pub pace_ms: f32,
    pub frame_ms: f32,
    pub fence_ms: f32,
    pub acquire_ms: f32,
    pub cull_ms: f32,
    pub draw_ms: f32,
    pub present_ms: f32,
    pub ticks: [u64; Timestamp::Count as usize],
    pub timestamp_period: f32,
    /// Modulus mask from the queue's `timestamp_valid_bits`; deltas wrap
    /// within this width, so subtract wrapping and mask instead of saturating.
    pub timestamp_mask: u64,
}
impl RenderTimings {
    pub fn duration(&self, start: Timestamp, end: Timestamp) -> f32 {
        let diff_ticks =
            self.ticks[end as usize].wrapping_sub(self.ticks[start as usize]) & self.timestamp_mask;
        (diff_ticks as f64 * self.timestamp_period as f64 / 1_000_000.0) as f32
    }
    pub fn frame_ms(&self) -> f32 {
        self.duration(Timestamp::FrameStart, Timestamp::FrameEnd)
    }
    pub fn cull_ms(&self) -> f32 {
        self.duration(Timestamp::CullStart, Timestamp::CullEnd)
    }
    pub fn gui_bake_ms(&self) -> f32 {
        self.duration(Timestamp::GuiBakeStart, Timestamp::GuiBakeEnd)
    }
    pub fn terrain_ms(&self) -> f32 {
        self.duration(Timestamp::TerrainStart, Timestamp::TerrainEnd)
    }
    pub fn entities_ms(&self) -> f32 {
        self.duration(Timestamp::EntitiesStart, Timestamp::EntitiesEnd)
    }
    pub fn translucent_ms(&self) -> f32 {
        self.duration(Timestamp::TranslucentStart, Timestamp::TranslucentEnd)
    }
    pub fn ui_ms(&self) -> f32 {
        self.duration(Timestamp::UiStart, Timestamp::UiEnd)
    }
}
pub struct Timer {
    cmd: rhi::backend::CommandEncoder,
    pool: Option<rhi::backend::TimestampQueries>,
}
impl Timer {
    pub fn new(
        cmd: rhi::backend::CommandEncoder,
        pool: Option<rhi::backend::TimestampQueries>,
    ) -> Self {
        Self { cmd, pool }
    }
    pub fn write(&self, point: Timestamp) {
        if let Some(pool) = self.pool {
            self.cmd.write_timestamp(pool, point as u32);
        }
    }
    pub fn zero_duration(&self, start: Timestamp, end: Timestamp) {
        self.write(start);
        self.write(end);
    }
    /// Returns a drop-guard that automatically writes the end timestamp when it
    /// goes out of scope. Both stamps use BottomOfPipe: a TopOfPipe start
    /// stamps before preceding work drains, overlapping the scopes.
    pub fn scope<'a>(&'a self, start: Timestamp, end: Timestamp) -> TimerScope<'a> {
        self.write(start);
        TimerScope { timer: self, end }
    }
}
pub struct TimerScope<'a> {
    timer: &'a Timer,
    end: Timestamp,
}
impl TimerScope<'_> {
    pub fn end(self) {}
}
impl<'a> Drop for TimerScope<'a> {
    fn drop(&mut self) {
        self.timer.write(self.end);
    }
}
