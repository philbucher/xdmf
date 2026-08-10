//! Allocation-counting global allocator for the M2 performance benchmarks
//! (`plans/02_performance.md` part A). Kept separate from `mesh.rs` so
//! `examples/bench_cfd.rs` — which reports wall time and bytes, not allocation counts — can
//! include the mesh generator alone without the allocator's `pub` items becoming unreachable
//! dead code in that binary.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

/// A `System`-wrapping allocator that counts calls to `alloc`/`alloc_zeroed`/`realloc`, so the
/// per-`write_data` allocation count (the metric `README.md` asks for) is visible to the benches.
///
/// Counts allocations for the whole process, criterion's own included, so it is only meaningful
/// as a delta around a narrow region via [`CountingAllocator::count`], not as an absolute number.
pub struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards unchanged to `System`, which is itself a valid `GlobalAlloc`;
// the only addition is a non-synchronizing counter bump around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

impl CountingAllocator {
    /// Runs `f`, returning its result together with the number of allocations observed while it
    /// ran.
    ///
    /// Not thread-safe against other allocating work on other threads (the counter is global,
    /// per the struct docs); the benches that use this run single-threaded, which criterion does
    /// by default for a single benchmark iteration.
    pub fn count<T>(f: impl FnOnce() -> T) -> (T, usize) {
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        let result = f();
        let after = ALLOCATIONS.load(Ordering::Relaxed);
        (result, after - before)
    }
}
