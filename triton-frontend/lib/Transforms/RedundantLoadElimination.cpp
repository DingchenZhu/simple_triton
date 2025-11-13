// lib/Transforms/RedundantLoadElimination.cpp
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "Dialect/TritonDialect.h"
#include "llvm/ADT/DenseMap.h"

namespace triton {
namespace transforms {

namespace {

class RedundantLoadEliminationPass : 
    public mlir::PassWrapper<RedundantLoadEliminationPass, 
                            mlir::OperationPass<mlir::func::FuncOp>> {
private:
    struct LoadInfo {
        mlir::Value ptr;
        mlir::Value mask;
        mlir::Value other;
        
        bool operator==(const LoadInfo& other) const {
            return ptr == other.ptr && 
                   mask == other.mask && 
                   this->other == other.other;
        }
    };
    
    struct LoadInfoHash {
        std::size_t operator()(const LoadInfo& info) const {
            std::size_t hash = std::hash<void*>()(info.ptr.getAsOpaquePointer());
            if (info.mask) {
                hash ^= std::hash<void*>()(info.mask.getAsOpaquePointer()) << 1;
            }
            if (info.other) {
                hash ^= std::hash<void*>()(info.other.getAsOpaquePointer()) << 2;
            }
            return hash;
        }
    };
    
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RedundantLoadEliminationPass)
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Process each block
        func.walk([this](mlir::Block* block) {
            eliminateRedundantLoadsInBlock(block);
        });
    }
    
    StringRef getArgument() const final { 
        return "triton-redundant-load-elimination"; 
    }
    StringRef getDescription() const final { 
        return "Redundant Load Elimination"; 
    }
    
private:
    void eliminateRedundantLoadsInBlock(mlir::Block* block) {
        std::unordered_map<LoadInfo, mlir::Value, LoadInfoHash> loadMap;
        llvm::SmallVector<mlir::Operation*, 16> toErase;
        mlir::AliasAnalysis aliasAnalysis(block->getParentOp());
        
        for (auto& op : *block) {
            if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(op)) {
                LoadInfo info{
                    loadOp.getPtr(),
                    loadOp.getMask(),
                    loadOp.getOther()
                };
                
                // Check if we've seen this load before
                auto it = loadMap.find(info);
                if (it != loadMap.end()) {
                    // Check if there's no intervening store
                    if (!hasInterveningStore(it->second.getDefiningOp(), 
                                           &op, aliasAnalysis)) {
                        // Replace redundant load
                        loadOp.getResult().replaceAllUsesWith(it->second);
                        toErase.push_back(&op);
                        continue;
                    }
                }
                
                // Add to load map
                loadMap[info] = loadOp.getResult();
            } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
                // Invalidate loads that may alias with this store
                invalidateAliasedLoads(storeOp, loadMap, aliasAnalysis);
            }
        }
        
        // Erase redundant loads
        for (auto* op : toErase) {
            op->erase();
        }
    }
    
    bool hasInterveningStore(mlir::Operation* load1,
                            mlir::Operation* load2,
                            mlir::AliasAnalysis& aliasAnalysis) {
        // Check if there's a store between load1 and load2 that may alias
        auto* block = load1->getBlock();
        bool foundLoad1 = false;
        
        for (auto& op : *block) {
            if (&op == load1) {
                foundLoad1 = true;
                continue;
            }
            
            if (&op == load2) {
                break;
            }
            
            if (foundLoad1) {
                if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
                    auto load1Op = llvm::cast<triton::LoadOp>(*load1);
                    if (mayAlias(load1Op.getPtr(), storeOp.getPtr(), aliasAnalysis)) {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    void invalidateAliasedLoads(
        triton::StoreOp storeOp,
        std::unordered_map<LoadInfo, mlir::Value, LoadInfoHash>& loadMap,
        mlir::AliasAnalysis& aliasAnalysis) {
        
        auto it = loadMap.begin();
        while (it != loadMap.end()) {
            if (mayAlias(it->first.ptr, storeOp.getPtr(), aliasAnalysis)) {
                it = loadMap.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    bool mayAlias(mlir::Value ptr1, mlir::Value ptr2,
                 mlir::AliasAnalysis& aliasAnalysis) {
        auto result = aliasAnalysis.alias(ptr1, ptr2);
        return result.isNo() ? false : true;
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createRedundantLoadEliminationPass() {
    return std::make_unique<RedundantLoadEliminationPass>();
}

} // namespace transforms
} // namespace triton