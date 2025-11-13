// lib/Transforms/DeadCodeElimination.cpp
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace triton {
namespace transforms {

namespace {

class DCEPass : public mlir::PassWrapper<DCEPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DCEPass)
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Perform multiple iterations until no more dead code is found
        bool changed = true;
        while (changed) {
            changed = false;
            
            // Collect live operations
            llvm::SmallPtrSet<mlir::Operation*, 32> liveOps;
            markLiveOperations(func, liveOps);
            
            // Remove dead operations
            changed |= removeDeadOperations(func, liveOps);
            
            // Remove dead blocks
            changed |= removeDeadBlocks(func);
        }
    }
    
    StringRef getArgument() const final { return "triton-dce"; }
    StringRef getDescription() const final { 
        return "Dead Code Elimination"; 
    }
    
private:
    void markLiveOperations(mlir::func::FuncOp func,
                           llvm::SmallPtrSet<mlir::Operation*, 32>& liveOps) {
        llvm::SetVector<mlir::Operation*> worklist;
        
        // Mark operations with side effects as live
        func.walk([&](mlir::Operation* op) {
            if (hasSideEffects(op) || isTerminator(op)) {
                worklist.insert(op);
                liveOps.insert(op);
            }
        });
        
        // Mark function returns as live
        func.walk([&](mlir::func::ReturnOp returnOp) {
            worklist.insert(returnOp.getOperation());
            liveOps.insert(returnOp.getOperation());
        });
        
        // Propagate liveness
        while (!worklist.empty()) {
            mlir::Operation* op = worklist.pop_back_val();
            
            // Mark operands as live
            for (auto operand : op->getOperands()) {
                if (auto defOp = operand.getDefiningOp()) {
                    if (liveOps.insert(defOp).second) {
                        worklist.insert(defOp);
                    }
                }
            }
            
            // For block arguments, mark predecessor terminators as live
            if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                mlir::Block* block = blockArg.getOwner();
                for (auto* pred : block->getPredecessors()) {
                    auto* terminator = pred->getTerminator();
                    if (liveOps.insert(terminator).second) {
                        worklist.insert(terminator);
                    }
                }
            }
        }
    }
    
    bool removeDeadOperations(mlir::func::FuncOp func,
                             const llvm::SmallPtrSet<mlir::Operation*, 32>& liveOps) {
        llvm::SmallVector<mlir::Operation*, 16> toErase;
        
        func.walk([&](mlir::Operation* op) {
            if (!liveOps.count(op) && !op->isKnownTerminator()) {
                toErase.push_back(op);
            }
        });
        
        // Erase dead operations in reverse order
        for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
            (*it)->erase();
        }
        
        return !toErase.empty();
    }
    
    bool removeDeadBlocks(mlir::func::FuncOp func) {
        bool changed = false;
        mlir::DominanceInfo domInfo(func);
        llvm::SmallVector<mlir::Block*, 8> toErase;
        
        for (auto& block : func) {
            // Skip entry block
            if (&block == &func.front()) {
                continue;
            }
            
            // Check if block is unreachable
            if (block.hasNoPredecessors() || 
                !domInfo.isReachableFromEntry(&block)) {
                toErase.push_back(&block);
            }
        }
        
        for (auto* block : toErase) {
            block->erase();
            changed = true;
        }
        
        return changed;
    }
    
    bool hasSideEffects(mlir::Operation* op) {
        return op->hasEffect<mlir::MemoryEffects::Write>() ||
               op->hasEffect<mlir::MemoryEffects::Allocate>() ||
               op->hasEffect<mlir::MemoryEffects::Free>() ||
               op->hasTrait<mlir::OpTrait::IsTerminator>();
    }
    
    bool isTerminator(mlir::Operation* op) {
        return op->hasTrait<mlir::OpTrait::IsTerminator>();
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createDeadCodeEliminationPass() {
    return std::make_unique<DCEPass>();
}

} // namespace transforms
} // namespace triton