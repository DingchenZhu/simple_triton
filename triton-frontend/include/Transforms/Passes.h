// include/Transforms/Passes.h
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
class PassManager;
} // namespace mlir

namespace triton {
namespace transforms {

// Pass creation functions
std::unique_ptr<mlir::Pass> createCommonSubexpressionEliminationPass();
std::unique_ptr<mlir::Pass> createDeadCodeEliminationPass();
std::unique_ptr<mlir::Pass> createOperatorFusionPass();
std::unique_ptr<mlir::Pass> createRedundantLoadEliminationPass();
std::unique_ptr<mlir::Pass> createLoopFusionPass();
std::unique_ptr<mlir::Pass> createMemoryCoalescingPass();
std::unique_ptr<mlir::Pass> createConstantFoldingPass();
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();

// Register all passes
void registerTritonOptimizationPasses();

// Add all optimization passes to a pass manager
void addTritonOptimizationPipeline(mlir::PassManager& pm, int optimizationLevel = 2);

} // namespace transforms
} // namespace triton