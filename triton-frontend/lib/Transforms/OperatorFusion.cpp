// lib/Transforms/OperatorFusion.cpp
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/TritonDialect.h"
#include <queue>

namespace triton {
namespace transforms {

namespace {

// Fusion pattern for elementwise operations
class ElementwiseFusionPattern : public mlir::OpRewritePattern<mlir::Operation> {
public:
    using OpRewritePattern::OpRewritePattern;
    
    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                       mlir::PatternRewriter& rewriter) const override {
        // Check if this is an elementwise operation
        if (!isElementwise(op)) {
            return mlir::failure();
        }
        
        // Find fusible chain of operations
        llvm::SmallVector<mlir::Operation*, 8> fusionChain;
        if (!buildFusionChain(op, fusionChain)) {
            return mlir::failure();
        }
        
        // Fuse operations
        return fuseOperations(fusionChain, rewriter);
    }
    
private:
    bool isElementwise(mlir::Operation* op) const {
        // Check if operation is elementwise
        // This includes arithmetic ops, comparisons, etc.
        return op->hasTrait<mlir::OpTrait::Elementwise>() ||
               llvm::isa<mlir::arith::AddFOp, mlir::arith::MulFOp,
                        mlir::arith::SubFOp, mlir::arith::DivFOp>(op);
    }
    
    bool buildFusionChain(mlir::Operation* root,
                         llvm::SmallVector<mlir::Operation*, 8>& chain) const {
        std::queue<mlir::Operation*> worklist;
        llvm::SmallPtrSet<mlir::Operation*, 8> visited;
        
        worklist.push(root);
        visited.insert(root);
        
        while (!worklist.empty()) {
            auto* op = worklist.front();
            worklist.pop();
            chain.push_back(op);
            
            // Check users
            for (auto user : op->getUsers()) {
                if (isElementwise(user) && !visited.count(user)) {
                    // Check if all operands are in the fusion chain or external
                    bool canFuse = true;
                    for (auto operand : user->getOperands()) {
                        if (auto defOp = operand.getDefiningOp()) {
                            if (!visited.count(defOp) && defOp != op) {
                                canFuse = false;
                                break;
                            }
                        }
                    }
                    
                    if (canFuse) {
                        worklist.push(user);
                        visited.insert(user);
                    }
                }
            }
            
            // Check operands
            for (auto operand : op->getOperands()) {
                if (auto defOp = operand.getDefiningOp()) {
                    if (isElementwise(defOp) && !visited.count(defOp)) {
                        // Check if this op has single use
                        if (defOp->hasOneUse()) {
                            worklist.push(defOp);
                            visited.insert(defOp);
                        }
                    }
                }
            }
        }
        
        return chain.size() > 1;
    }
    
    mlir::LogicalResult fuseOperations(
        const llvm::SmallVector<mlir::Operation*, 8>& chain,
        mlir::PatternRewriter& rewriter) const {
        
        // Create fused operation
        auto loc = chain.front()->getLoc();
        
        // Collect inputs and outputs
        llvm::SmallVector<mlir::Value, 8> inputs;
        llvm::SmallVector<mlir::Value, 8> outputs;
        llvm::SmallPtrSet<mlir::Value, 8> internalValues;
        
        for (auto* op : chain) {
            for (auto result : op->getResults()) {
                internalValues.insert(result);
            }
        }
        
        for (auto* op : chain) {
            for (auto operand : op->getOperands()) {
                if (!internalValues.count(operand)) {
                    inputs.push_back(operand);
                }
            }
            for (auto result : op->getResults()) {
                bool isOutput = false;
                for (auto user : result.getUsers()) {
                    if (std::find(chain.begin(), chain.end(), user) == chain.end()) {
                        isOutput = true;
                        break;
                    }
                }
                if (isOutput) {
                    outputs.push_back(result);
                }
            }
        }
        
        // Create custom fused operation
        auto fusedOp = rewriter.create<FusedKernelOp>(
            loc, 
            mlir::TypeRange(llvm::SmallVector<mlir::Type, 4>(
                outputs.begin(), outputs.end(),
                [](mlir::Value v) { return v.getType(); }
            )),
            inputs
        );
        
        // Replace uses
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i].replaceAllUsesWith(fusedOp.getResult(i));
        }
        
        // Erase original operations
        for (auto* op : llvm::reverse(chain)) {
            rewriter.eraseOp(op);
        }
        
        return mlir::success();
    }
};

// Pattern for load-compute-store fusion
class LoadComputeStoreFusionPattern : public mlir::OpRewritePattern<triton::StoreOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    
    mlir::LogicalResult matchAndRewrite(triton::StoreOp storeOp,
                                       mlir::PatternRewriter& rewriter) const override {
        auto value = storeOp.getValue();
        
        // Check if value comes from a compute operation
        auto computeOp = value.getDefiningOp();
        if (!computeOp || !isCompute(computeOp)) {
            return mlir::failure();
        }
        
        // Check if compute operation uses loads
        llvm::SmallVector<triton::LoadOp, 4> loads;
        for (auto operand : computeOp->getOperands()) {
            if (auto loadOp = operand.getDefiningOp<triton::LoadOp>()) {
                loads.push_back(loadOp);
            }
        }
        
        if (loads.empty()) {
            return mlir::failure();
        }
        
        // Create fused operation
        auto loc = storeOp.getLoc();
        auto fusedOp = rewriter.create<FusedLoadComputeStoreOp>(
            loc,
            storeOp.getPtr(),
            llvm::SmallVector<mlir::Value, 4>(
                loads.begin(), loads.end(),
                [](triton::LoadOp load) { return load.getPtr(); }
            ),
            computeOp->getName().getIdentifier(),
            computeOp->getAttrs()
        );
        
        // Replace store operation
        rewriter.eraseOp(storeOp);
        
        // Remove compute operation if it has no other uses
        if (computeOp->use_empty()) {
            rewriter.eraseOp(computeOp);
        }
        
        // Remove load operations if they have no other uses
        for (auto loadOp : loads) {
            if (loadOp->use_empty()) {
                rewriter.eraseOp(loadOp);
            }
        }
        
        return mlir::success();
    }
    
private:
    bool isCompute(mlir::Operation* op) const {
        return llvm::isa<mlir::arith::AddFOp, mlir::arith::MulFOp,
                        mlir::arith::SubFOp, mlir::arith::DivFOp>(op);
    }
};

// Fusion pass
class OperatorFusionPass : public mlir::PassWrapper<OperatorFusionPass, 
                                                     mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperatorFusionPass)
    
    void runOnOperation() override {
        auto func = getOperation();
        mlir::MLIRContext* context = &getContext();
        
        mlir::RewritePatternSet patterns(context);
        patterns.add<ElementwiseFusionPattern>(context);
        patterns.add<LoadComputeStoreFusionPattern>(context);
        
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getArgument() const final { return "triton-operator-fusion"; }
    StringRef getDescription() const final { 
        return "Operator Fusion Optimization"; 
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createOperatorFusionPass() {
    return std::make_unique<OperatorFusionPass>();
}

} // namespace transforms
} // namespace triton