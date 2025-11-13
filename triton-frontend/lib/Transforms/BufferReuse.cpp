// lib/Transforms/BufferReuse.cpp
#include "Transforms/Bufferization.h"
#include "mlir/Analysis/Liveness.h"

namespace triton {
namespace transforms {

namespace {

class BufferReusePass :
    public mlir::PassWrapper<BufferReusePass,
                            mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferReusePass)
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Perform liveness analysis
        mlir::Liveness liveness(func);
        
        // Build allocation graph
        AllocationGraph allocGraph(func, liveness);
        allocGraph.build();
        
        // Find reuse opportunities
        ReuseAnalyzer reuseAnalyzer(allocGraph);
        auto reusePairs = reuseAnalyzer.findReusableBuffers();
        
        // Apply reuse transformations
        applyBufferReuse(reusePairs);
    }
    
    StringRef getArgument() const final { 
        return "triton-buffer-reuse"; 
    }
    StringRef getDescription() const final { 
        return "Buffer reuse optimization"; 
    }
    
private:
    class AllocationGraph {
    public:
        struct AllocInfo {
            mlir::memref::AllocOp allocOp;
            mlir::Operation* firstUse;
            mlir::Operation* lastUse;
            int64_t size;
            unsigned memorySpace;
            bool canReuse;
        };
        
        AllocationGraph(mlir::func::FuncOp func, mlir::Liveness& liveness)
            : func(func), liveness(liveness) {}
        
        void build() {
            // Collect all allocations
            func.walk([this](mlir::memref::AllocOp allocOp) {
                analyzeAllocation(allocOp);
            });
            
            // Build interference graph
            buildInterferenceGraph();
        }
        
        llvm::SmallVector<AllocInfo, 16> allocations;
        llvm::DenseSet<std::pair<mlir::memref::AllocOp, 
                                 mlir::memref::AllocOp>> interference;
        
    private:
        mlir::func::FuncOp func;
        mlir::Liveness& liveness;
        
        void analyzeAllocation(mlir::memref::AllocOp allocOp) {
            AllocInfo info;
            info.allocOp = allocOp;
            info.size = getAllocationSize(allocOp);
            info.memorySpace = allocOp.getType().getMemorySpaceAsInt();
            info.canReuse = !hasEscapingUse(allocOp);
            
            // Find first and last use
            findUseRange(allocOp, info.firstUse, info.lastUse);
            
            allocations.push_back(info);
        }
        
        int64_t getAllocationSize(mlir::memref::AllocOp allocOp) {
            auto type = allocOp.getType();
            int64_t size = 1;
            
            for (auto dim : type.getShape()) {
                if (dim == mlir::ShapedType::kDynamic) {
                    return -1; // Dynamic size
                }
                size *= dim;
            }
            
            size *= type.getElementTypeBitWidth() / 8;
            return size;
        }
        
        bool hasEscapingUse(mlir::memref::AllocOp allocOp) {
            for (auto user : allocOp.getResult().getUsers()) {
                if (llvm::isa<mlir::func::ReturnOp>(user) ||
                    llvm::isa<mlir::func::CallOp>(user)) {
                    return true;
                }
            }
            return false;
        }
        
        void findUseRange(mlir::memref::AllocOp allocOp,
                         mlir::Operation*& firstUse,
                         mlir::Operation*& lastUse) {
            firstUse = nullptr;
            lastUse = nullptr;
            
            for (auto& use : allocOp.getResult().getUses()) {
                auto* user = use.getOwner();
                if (!firstUse || user->isBeforeInBlock(firstUse)) {
                    firstUse = user;
                }
                if (!lastUse || lastUse->isBeforeInBlock(user)) {
                    lastUse = user;
                }
            }
        }
        
        void buildInterferenceGraph() {
            for (size_t i = 0; i < allocations.size(); ++i) {
                for (size_t j = i + 1; j < allocations.size(); ++j) {
                    if (interferes(allocations[i], allocations[j])) {
                        interference.insert({allocations[i].allocOp,
                                           allocations[j].allocOp});
                    }
                }
            }
        }
        
        bool interferes(const AllocInfo& a, const AllocInfo& b) {
            // Check if lifetimes overlap
            if (!a.firstUse || !a.lastUse || !b.firstUse || !b.lastUse) {
                return true; // Conservative
            }
            
            // Check if one's lifetime contains the other's start
            if (liveness.isLiveAt(a.allocOp.getResult(), b.firstUse) ||
                liveness.isLiveAt(b.allocOp.getResult(), a.firstUse)) {
                return true;
            }
            
            return false;
        }
    };
    
    class ReuseAnalyzer {
    public:
        explicit ReuseAnalyzer(AllocationGraph& allocGraph)
            : allocGraph(allocGraph) {}
        
        llvm::SmallVector<std::pair<mlir::memref::AllocOp, 
                                    mlir::memref::AllocOp>, 8>
        findReusableBuffers() {
            llvm::SmallVector<std::pair<mlir::memref::AllocOp,
                                       mlir::memref::AllocOp>, 8> reusePairs;
            
            for (size_t i = 0; i < allocGraph.allocations.size(); ++i) {
                for (size_t j = i + 1; j < allocGraph.allocations.size(); ++j) {
                    if (canReuse(allocGraph.allocations[i],
                               allocGraph.allocations[j])) {
                        reusePairs.push_back({allocGraph.allocations[i].allocOp,
                                            allocGraph.allocations[j].allocOp});
                    }
                }
            }
            
            return reusePairs;
        }
        
    private:
        AllocationGraph& allocGraph;
        
        bool canReuse(const AllocationGraph::AllocInfo& a,
                     const AllocationGraph::AllocInfo& b) {
            // Check if buffers can be reused
            
            // 1. Must not interfere
            if (allocGraph.interference.count({a.allocOp, b.allocOp}) ||
                allocGraph.interference.count({b.allocOp, a.allocOp})) {
                return false;
            }
            
            // 2. Must have compatible sizes
            if (a.size != b.size || a.size == -1) {
                return false;
            }
            
            // 3. Must be in same memory space
            if (a.memorySpace != b.memorySpace) {
                return false;
            }
            
            // 4. Must be reusable
            if (!a.canReuse || !b.canReuse) {
                return false;
            }
            
            return true;
        }
    };
    
    void applyBufferReuse(
        const llvm::SmallVector<std::pair<mlir::memref::AllocOp,
                                         mlir::memref::AllocOp>, 8>& reusePairs) {
        for (auto [alloc1, alloc2] : reusePairs) {
            // Reuse alloc1's buffer for alloc2
            alloc2.getResult().replaceAllUsesWith(alloc1.getResult());
            
            // Remove alloc2 and its dealloc
            alloc2.erase();
            
            // Find and remove corresponding dealloc
            for (auto user : alloc2.getResult().getUsers()) {
                if (auto deallocOp = llvm::dyn_cast<mlir::memref::DeallocOp>(user)) {
                    deallocOp.erase();
                    break;
                }
            }
        }
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createBufferReusePass() {
    return std::make_unique<BufferReusePass>();
}

} // namespace transforms
} // namespace triton