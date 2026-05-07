use std::sync::Arc;

#[cfg(feature = "cpu-profiling")]
use std::collections::HashMap;
#[cfg(feature = "cpu-profiling")]
use std::sync::Mutex;
#[cfg(feature = "cpu-profiling")]
use std::time::Instant;

#[cfg(feature = "cpu-profiling")]
pub struct Profiler {
    cpu_records: Mutex<HashMap<&'static str, Vec<f32>>>,
}

#[cfg(feature = "cpu-profiling")]
impl Profiler {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cpu_records: Mutex::new(HashMap::new()),
        })
    }

    pub fn scope(self: &Arc<Self>, name: &'static str) -> CpuScopeGuard {
        CpuScopeGuard {
            name,
            start: Instant::now(),
            profiler: Arc::clone(self),
        }
    }

    // `scope_cond` removed; profiling is gated by compile-time `cpu-profiling` feature

    fn record(&self, name: &'static str, elapsed: f32) {
        let mut lock = self.cpu_records.lock().unwrap();
        lock.entry(name).or_default().push(elapsed);
    }

    /// Print a short summary (averages) via logging for each scope.
    pub fn print_summary(&self) {
        let lock = self.cpu_records.lock().unwrap();
        if lock.is_empty() {
            return;
        }
        log::info!("--- CPU profiling summary (last values) ---");
        for (k, v) in lock.iter() {
            let sum: f32 = v.iter().copied().sum();
            let avg = sum / (v.len() as f32);
            log::info!(
                "{:<32} avg: {:>7.3} ms (samples {})",
                k,
                avg * 1000.0,
                v.len()
            );
        }
    }
}

#[cfg(feature = "cpu-profiling")]
pub struct CpuScopeGuard {
    name: &'static str,
    start: Instant,
    profiler: Arc<Profiler>,
}

#[cfg(feature = "cpu-profiling")]
impl Drop for CpuScopeGuard {
    fn drop(&mut self) {
        let elapsed = (Instant::now() - self.start).as_secs_f32();
        self.profiler.record(self.name, elapsed);
    }
}

// No-op profiler when feature is not enabled
#[cfg(not(feature = "cpu-profiling"))]
pub struct Profiler;

#[cfg(not(feature = "cpu-profiling"))]
impl Profiler {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
    pub fn scope(self: &Arc<Self>, _name: &'static str) -> DummyGuard {
        DummyGuard
    }
    // `scope_cond` removed; no-op profiler only exposes `scope` in this mode
    pub fn print_summary(&self) {}
}

#[cfg(not(feature = "cpu-profiling"))]
pub struct DummyGuard;

#[cfg(not(feature = "cpu-profiling"))]
impl Drop for DummyGuard {
    fn drop(&mut self) {}
}
