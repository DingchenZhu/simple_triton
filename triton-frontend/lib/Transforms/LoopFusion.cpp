// lib/Transforms/LoopFusion.cpp
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/Dominance.h"

namespace triton {
namespace transforms {

namespace {

class LoopFusionPass : public mlir::PassWrapper<LoopFusionPass,
                                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopFusionPass)
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Collect all loops
        llvm::SmallVector<mlir::scf::ForOp, 16> loops;
        func.walk([&](mlir::scf::ForOp forOp) {
            loops.push_back(forOp);
        });
        
        // Try to fuse adjacent loops
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < loops.size() && !changed; ++i) {
                for (size_t j = i + 1; j < loops.size() && !changed; ++j) {
                    if (canFuse(loops[i], loops[j])) {
                        fuseLoops(loops[i], loops[j]);
                        loops.erase(loops.begin() + j);
                        changed = true;
                    }
                }
            }
        }
    }
    
    StringRef getArgument() const final { return "triton-loop-fusion"; }
    StringRef getDescription() const final { 
        return "Loop Fusion Optimization"; 
    }
    
private:
    bool canFuse(mlir::scf::ForOp loop1, mlir::scf::ForOp loop2) {
        // Check if loops are adjacent
        if (!areAdjacent(loop1, loop2)) {
            return false;
        }
        
        // Check if loops have same bounds
        if (!haveSameBounds(loop1, loop2)) {
            return false;
        }
        
        // Check for dependencies
        if (hasDataDependency(loop1, loop2)) {
            return false;
        }
        
        return true;
    }
    
    bool areAdjacent(mlir::scf::ForOp loop1, mlir::scf::ForOp loop2) {
        // Check if loops are in the same block
        if (loop1->getBlock() != loop2->getBlock()) {
            return false;
        }
        
        // Check if loop2 immediately follows loop1
        auto* next = loop1->getNextNode();
        while (next && next != loop2) {
            // Skip operations that don't prevent fusion
            if (!llvm::isa<mlir::arith::ConstantOp>(next)) {
                return false;
            }
            next = next->getNextNode();
        }
        
        return next == loop2;
    }
    
    bool haveSameBounds(mlir::scf::ForOp loop1, mlir::scf::ForOp loop2) {
        return loop1.getLowerBound() == loop2.getLowerBound() &&
               loop1.getUpperBound() == loop2.getUpperBound() &&
               loop1.getStep() == loop2.getStep();
    }
    
    bool hasDataDependency(mlir::scf::ForOp loop1, mlir::scf::ForOp loop2) {
        // Collect values defined in loop1
        llvm::SmallPtrSet<mlir::Value, 16> loop1Defs;
        loop1.walk([&](mlir::Operation* op) {
            for (auto result : op->getResults()) {
                loop1Defs.insert(result);
            }
        });
        
        // Check if loop2 uses any values from loop1
        bool hasUse = false;
        loop2.walk([&](mlir::Operation* op) {
            for (auto operand : op->getOperands()) {
                if (loop1Defs.count(operand)) {
                    hasUse = true;
                    return;
                }
            }
        });
        
        return hasUse;
    }
    
    void fuseLoops(mlir::scf::ForOp loop1, mlir::scf::ForOp loop2) {
        mlir::OpBuilder builder(loop1);
        
        // Create new fused loop
        auto fusedLoop = builder.create<mlir::scf::ForOp>(
            loop1.getLoc(),
            loop1.getLowerBound(),
            loop1.getUpperBound(),
            loop1.getStep()
        );
        
        // Move operations from both loops
        builder.setInsertionPointToStart(fusedLoop.getBody());
        
        // Copy operations from loop1
        for (auto& op : loop1.getBody()->without_terminator()) {
            builder.clone(op);
        }
        
        // Copy operations from loop2
        for (auto& op : loop2.getBody()->without_terminator()) {
            builder.clone(op);
        }
        
        // Add yield
        builder.create<mlir::scf::YieldOp>(fusedLoop.getLoc());
        
        // Erase original loops
        loop1.erase();
        loop2.erase();
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLoopFusionPass() {
    return std::make_unique<LoopFusionPass>();
}

} // namespace transforms
} // namespace triton