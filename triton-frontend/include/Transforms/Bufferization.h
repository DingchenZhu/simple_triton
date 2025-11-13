// include/Transforms/Bufferization.h
#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace triton {
namespace transforms {

// Bufferization options
struct TritonBufferizationOptions {
    // Memory space mappings
    enum class MemorySpace {
        Global = 0,      // Global memory
        Shared = 3,      // Shared memory
        Local = 5,       // Local/register memory
        Constant = 4     // Constant memory
    };
    
    // Allocation strategy
    enum class AllocationStrategy {
        Static,          // Static allocation
        Dynamic,         // Dynamic allocation
        StackBased,      // Stack-based allocation
        PoolBased        // Memory pool based
    };
    
    MemorySpace defaultMemorySpace = MemorySpace::Global;
    AllocationStrategy allocationStrategy = AllocationStrategy::Dynamic;
    bool allowReturnAllocs = false;
    bool hoistAllocations = true;
    bool promoteBufferToStack = true;
    bool analyzeAliasing = true;
    size_t alignmentBits = 128;  // GPU memory alignment
    size_t sharedMemoryLimit = 49152;  // 48KB shared memory limit
};

// Forward declarations
std::unique_ptr<mlir::Pass> createTritonBufferizationPass(
    const TritonBufferizationOptions& options = TritonBufferizationOptions());
std::unique_ptr<mlir::Pass> createBufferDeallocationPass();
std::unique_ptr<mlir::Pass> createBufferHoistingPass();
std::unique_ptr<mlir::Pass> createBufferReusePass();
std::unique_ptr<mlir::Pass> createInPlaceBufferizationPass();

} // namespace transforms
} // namespace triton