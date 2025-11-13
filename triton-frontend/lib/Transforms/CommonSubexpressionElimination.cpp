// lib/Transforms/CommonSubexpressionElimination.cpp
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <unordered_map>

namespace triton {
namespace transforms {

namespace {

// Hash function for MLIR Values
struct ValueHash {
    std::size_t operator()(const mlir::Value& v) const {
        return std::hash<void*>()(v.getAsOpaquePointer());
    }
};

// Expression signature for CSE
struct ExpressionSignature {
    mlir::StringRef opName;
    llvm::SmallVector<mlir::Value, 4> operands;
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    mlir::Type resultType;
    
    bool operator==(const ExpressionSignature& other) const {
        return opName == other.opName &&
               operands == other.operands &&
               attributes == other.attributes &&
               resultType == other.resultType;
    }
};

struct ExpressionSignatureHash {
    std::size_t operator()(const ExpressionSignature& sig) const {
        std::size_t hash = std::hash<std::string>()(sig.opName.str());
        for (auto operand : sig.operands) {
            hash ^= std::hash<void*>()(operand.getAsOpaquePointer()) << 1;
        }
        for (auto attr : sig.attributes) {
            hash ^= std::hash<void*>()(attr.getAsOpaquePointer()) << 2;
        }
        hash ^= std::hash<void*>()(sig.resultType.getAsOpaquePointer()) << 3;
        return hash;
    }
};

class CSEPass : public mlir::PassWrapper<CSEPass, mlir::OperationPass<mlir::func::FuncOp>> {
private:
    using ExpressionMap = std::unordered_map<ExpressionSignature, 
                                             mlir::Value, 
                                             ExpressionSignatureHash>;
    
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSEPass)
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Process each block in the function
        func.walk([this](mlir::Block* block) {
            eliminateCSEInBlock(block);
        });
    }
    
    StringRef getArgument() const final { return "triton-cse"; }
    StringRef getDescription() const final { 
        return "Common Subexpression Elimination"; 
    }
    
private:
    void eliminateCSEInBlock(mlir::Block* block) {
        ExpressionMap expressionMap;
        llvm::SmallVector<mlir::Operation*, 16> toErase;
        
        for (auto& op : *block) {
            // Skip operations with side effects
            if (op.hasEffect<mlir::MemoryEffects::Write>() ||
                op.hasEffect<mlir::MemoryEffects::Allocate>() ||
                op.hasEffect<mlir::MemoryEffects::Free>()) {
                continue;
            }
            
            // Skip operations without results
            if (op.getNumResults() == 0) {
                continue;
            }
            
            // Create expression signature
            ExpressionSignature signature;
            signature.opName = op.getName().getStringRef();
            signature.operands.append(op.operands().begin(), op.operands().end());
            
            for (auto attr : op.getAttrs()) {
                signature.attributes.push_back(attr.getValue());
            }
            
            if (op.getNumResults() > 0) {
                signature.resultType = op.getResult(0).getType();
            }
            
            // Check if expression already exists
            auto it = expressionMap.find(signature);
            if (it != expressionMap.end()) {
                // Replace all uses with existing value
                op.getResult(0).replaceAllUsesWith(it->second);
                toErase.push_back(&op);
            } else {
                // Add to expression map
                expressionMap[signature] = op.getResult(0);
            }
        }
        
        // Erase redundant operations
        for (auto* op : toErase) {
            op->erase();
        }
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createCommonSubexpressionEliminationPass() {
    return std::make_unique<CSEPass>();
}

} // namespace transforms
} // namespace triton