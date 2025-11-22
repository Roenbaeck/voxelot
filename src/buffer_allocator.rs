//! Slab allocator for managing regions within large GPU buffers
//!
//! This module provides a free-list based allocator for subdividing large
//! GPU buffers into smaller chunks without requiring individual buffer allocations.

use std::fmt;

/// Error type for allocation failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationError {
    /// Not enough contiguous space available
    OutOfMemory,
    /// Requested size is larger than total buffer capacity
    SizeTooLarge,
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationError::OutOfMemory => write!(f, "Out of memory: no suitable free region"),
            AllocationError::SizeTooLarge => write!(f, "Allocation size exceeds buffer capacity"),
        }
    }
}

impl std::error::Error for AllocationError {}

/// Represents a free region in the buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRegion {
    offset: u64,
    size: u64,
}

/// Slab allocator for managing regions within a fixed-size buffer
///
/// Uses a sorted free-list with first-fit allocation strategy.
/// Coalesces adjacent free regions on deallocation.
pub struct SlabAllocator {
    /// Total size of the buffer being managed
    total_size: u64,
    /// Sorted list of free regions (by offset)
    free_list: Vec<FreeRegion>,
    /// Number of currently allocated regions
    allocated_count: usize,
    /// Total bytes currently allocated
    allocated_bytes: u64,
}

impl SlabAllocator {
    /// Create a new allocator for a buffer of the given size
    pub fn new(total_size: u64) -> Self {
        Self {
            total_size,
            free_list: vec![FreeRegion {
                offset: 0,
                size: total_size,
            }],
            allocated_count: 0,
            allocated_bytes: 0,
        }
    }

    /// Allocate a region of at least `size` bytes
    ///
    /// Returns the byte offset of the allocated region on success.
    /// Uses first-fit strategy: finds the first free region large enough.
    pub fn allocate(&mut self, size: u64) -> Result<u64, AllocationError> {
        if size == 0 {
            return Ok(0); // Allow zero-size allocations (used for empty meshes)
        }

        if size > self.total_size {
            return Err(AllocationError::SizeTooLarge);
        }

        // Find first free region that fits
        for i in 0..self.free_list.len() {
            let region = self.free_list[i];
            if region.size >= size {
                let offset = region.offset;

                // Split the region: remove allocated portion from free list
                if region.size == size {
                    // Exact fit - remove entire region
                    self.free_list.remove(i);
                } else {
                    // Partial fit - shrink the free region
                    self.free_list[i] = FreeRegion {
                        offset: region.offset + size,
                        size: region.size - size,
                    };
                }

                self.allocated_count += 1;
                self.allocated_bytes += size;

                return Ok(offset);
            }
        }

        Err(AllocationError::OutOfMemory)
    }

    /// Allocate a region of at least `size` bytes with `align` alignment.
    /// Returns a byte offset aligned to `align`.
    pub fn allocate_aligned(&mut self, size: u64, align: u64) -> Result<u64, AllocationError> {
        if size == 0 {
            return Ok(0);
        }
        if align == 0 {
            return self.allocate(size);
        }
        if size > self.total_size {
            return Err(AllocationError::SizeTooLarge);
        }

        for i in 0..self.free_list.len() {
            let region = self.free_list[i];
            // Align the offset within the region
            let mut aligned_offset = region.offset;
            if aligned_offset % align != 0 {
                aligned_offset += align - (aligned_offset % align);
            }
            let end = region.offset + region.size;
            if aligned_offset + size <= end {
                // We can allocate here. Remove or adjust free regions.
                // Remove the current region and replace with prefix/suffix as needed.
                self.free_list.remove(i);
                // prefix
                if aligned_offset > region.offset {
                    self.free_list.insert(i, FreeRegion { offset: region.offset, size: aligned_offset - region.offset });
                }
                // suffix
                if aligned_offset + size < end {
                    let suffix_offset = aligned_offset + size;
                    let suffix_size = end - suffix_offset;
                    let insert_idx = self.free_list.binary_search_by_key(&suffix_offset, |r| r.offset).unwrap_or_else(|p| p);
                    self.free_list.insert(insert_idx, FreeRegion { offset: suffix_offset, size: suffix_size });
                }

                self.allocated_count += 1;
                self.allocated_bytes += size;
                return Ok(aligned_offset);
            }
        }

        Err(AllocationError::OutOfMemory)
    }

    /// Free a previously allocated region
    ///
    /// # Arguments
    /// * `offset` - The offset returned by `allocate()`
    /// * `size` - The size that was allocated
    ///
    /// This will coalesce adjacent free regions to reduce fragmentation.
    pub fn free(&mut self, offset: u64, size: u64) {
        if size == 0 {
            return; // Nothing to free
        }

        // Create new free region
        let new_region = FreeRegion { offset, size };

        // Find insertion point (free_list is sorted by offset)
        let insert_pos = self
            .free_list
            .binary_search_by_key(&offset, |r| r.offset)
            .unwrap_or_else(|pos| pos);

        // Insert the region
        self.free_list.insert(insert_pos, new_region);

        // Coalesce adjacent regions
        self.coalesce_at(insert_pos);

        self.allocated_count = self.allocated_count.saturating_sub(1);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
    }

    /// Coalesce free regions adjacent to the one at `index`
    fn coalesce_at(&mut self, index: usize) {
        // Coalesce with next region
        while index < self.free_list.len() - 1 {
            let current = self.free_list[index];
            let next = self.free_list[index + 1];

            if current.offset + current.size == next.offset {
                // Adjacent - merge them
                self.free_list[index].size += next.size;
                self.free_list.remove(index + 1);
            } else {
                break;
            }
        }

        // Coalesce with previous region
        if index > 0 {
            let prev_idx = index - 1;
            let prev = self.free_list[prev_idx];
            let current = self.free_list[index];

            if prev.offset + prev.size == current.offset {
                // Adjacent - merge them
                self.free_list[prev_idx].size += current.size;
                self.free_list.remove(index);
            }
        }
    }

    /// Get the total size of the buffer
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Get the number of currently allocated regions
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Get the total bytes currently allocated
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Get the total bytes available (free)
    pub fn available_bytes(&self) -> u64 {
        self.free_list.iter().map(|r| r.size).sum()
    }

    /// Get the largest contiguous free region size
    pub fn largest_free_region(&self) -> u64 {
        self.free_list.iter().map(|r| r.size).max().unwrap_or(0)
    }

    /// Get fragmentation metric (0.0 = no fragmentation, 1.0 = highly fragmented)
    ///
    /// Calculated as: 1.0 - (largest_free / total_free)
    /// If there's only one free region, fragmentation is 0.
    pub fn fragmentation(&self) -> f32 {
        let total_free = self.available_bytes();
        if total_free == 0 {
            return 0.0; // Completely allocated
        }

        let largest = self.largest_free_region();
        if largest == total_free {
            return 0.0; // All free space is contiguous
        }

        1.0 - (largest as f32 / total_free as f32)
    }

    /// Get the number of free regions
    pub fn free_region_count(&self) -> usize {
        self.free_list.len()
    }

    /// Reset the allocator to its initial state (all free)
    pub fn reset(&mut self) {
        self.free_list.clear();
        self.free_list.push(FreeRegion {
            offset: 0,
            size: self.total_size,
        });
        self.allocated_count = 0;
        self.allocated_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = SlabAllocator::new(1024);

        // Allocate some regions
        let offset1 = allocator.allocate(100).unwrap();
        assert_eq!(offset1, 0);
        assert_eq!(allocator.allocated_count(), 1);
        assert_eq!(allocator.allocated_bytes(), 100);

        let offset2 = allocator.allocate(200).unwrap();
        assert_eq!(offset2, 100);
        assert_eq!(allocator.allocated_count(), 2);

        let offset3 = allocator.allocate(300).unwrap();
        assert_eq!(offset3, 300);

        // Free middle region
        allocator.free(offset2, 200);
        assert_eq!(allocator.allocated_count(), 2);
        assert_eq!(allocator.allocated_bytes(), 400);

        // Allocate smaller region - should reuse freed space
        let offset4 = allocator.allocate(50).unwrap();
        assert_eq!(offset4, 100); // Reuses freed space
    }

    #[test]
    fn test_coalescing() {
        let mut allocator = SlabAllocator::new(1000);

        let o1 = allocator.allocate(100).unwrap();
        let o2 = allocator.allocate(100).unwrap();
        let o3 = allocator.allocate(100).unwrap();

        // Free in reverse order - should coalesce
        allocator.free(o3, 100);
        allocator.free(o2, 100);
        allocator.free(o1, 100);

        // Should have single free region now
        assert_eq!(allocator.free_region_count(), 1);
        assert_eq!(allocator.largest_free_region(), 1000);
    }

    #[test]
    fn test_out_of_memory() {
        let mut allocator = SlabAllocator::new(100);

        allocator.allocate(50).unwrap();
        allocator.allocate(30).unwrap();

        // Only 20 bytes left
        assert!(matches!(
            allocator.allocate(50),
            Err(AllocationError::OutOfMemory)
        ));
    }

    #[test]
    fn test_size_too_large() {
        let mut allocator = SlabAllocator::new(100);

        assert!(matches!(
            allocator.allocate(200),
            Err(AllocationError::SizeTooLarge)
        ));
    }

    #[test]
    fn test_fragmentation() {
        let mut allocator = SlabAllocator::new(1000);

        // Initially no fragmentation
        assert_eq!(allocator.fragmentation(), 0.0);

        // Allocate some regions
        let o1 = allocator.allocate(100).unwrap();
        let _o2 = allocator.allocate(100).unwrap();
        let o3 = allocator.allocate(100).unwrap();

        // Free non-adjacent regions - creates fragmentation
        allocator.free(o1, 100);
        allocator.free(o3, 100);

        // Should have 2 free regions (fragmented)
        assert!(allocator.fragmentation() > 0.0);
        assert_eq!(allocator.free_region_count(), 2);
    }

    #[test]
    fn test_zero_size_allocation() {
        let mut allocator = SlabAllocator::new(100);

        let offset = allocator.allocate(0).unwrap();
        assert_eq!(offset, 0);

        allocator.free(0, 0); // Should not panic
        assert_eq!(allocator.allocated_count(), 0);
    }
}
