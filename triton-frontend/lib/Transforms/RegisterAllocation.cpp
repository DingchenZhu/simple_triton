// lib/Transforms/RegisterAllocation.cpp
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/Liveness.h"

namespace triton {
namespace transforms {

namespace {

class RegisterAllocationPass :
    public mlir::PassWrapper<RegisterAllocationPass,
                            mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegisterAllocationPass)
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Perform liveness analysis
        mlir::Liveness liveness(func);
        
        // Build interference graph
        InterferenceGraph interferenceGraph(func, liveness);
        
        // Perform register allocation
        RegisterAllocator allocator(interferenceGraph);
        allocator.allocate();
        
        // Apply allocation results
        applyAllocation(func, allocator);
        
        // Spill if necessary
        handleSpills(func, allocator);
    }
    
    StringRef getArgument() const final { 
        return "triton-register-allocation"; 
    }
    StringRef getDescription() const final { 
        return "Register Allocation Optimization"; 
    }
    
private:
    class InterferenceGraph {
    public:
        InterferenceGraph(mlir::func::FuncOp func, mlir::Liveness& liveness) {
            buildGraph(func, liveness);
        }
        
        bool interferes(mlir::Value a, mlir::Value b) const {
            return interference.count({a, b}) || interference.count({b, a});
        }
        
        llvm::SmallVector<mlir::Value, 16> values;
        llvm::DenseSet<std::pair<mlir::Value, mlir::Value>> interference;
        
    private:
        void buildGraph(mlir::func::FuncOp func, mlir::Liveness& liveness) {
            // Collect all values
            func.walk([this](mlir::Operation* op) {
                for (auto result : op->getResults()) {
                    values.push_back(result);
                }
            });
            
            // Build interference edges
            for (size_t i = 0; i < values.size(); ++i) {
                for (size_t j = i + 1; j < values.size(); ++j) {
                    if (liveness.isLiveAt(values[i], values[j].getDefiningOp()) ||
                        liveness.isLiveAt(values[j], values[i].getDefiningOp())) {
                        interference.insert({values[i], values[j]});
                    }
                }
            }
        }
    };
    
    class RegisterAllocator {
    public:
        explicit RegisterAllocator(InterferenceGraph& graph) 
            : graph(graph), maxRegisters(255) {} // GPU register file size
        
        void allocate() {
            // Use graph coloring algorithm
            graphColoring();
        }
        
        int getRegister(mlir::Value value) const {
            auto it = allocation.find(value);
            return it != allocation.end() ? it->second : -1;
        }
        
        llvm::DenseMap<mlir::Value, int> allocation;
        llvm::SmallVector<mlir::Value, 8> spilled;
        
    private:
        InterferenceGraph& graph;
        int maxRegisters;
        
        void graphColoring() {
            // Simplified graph coloring implementation
            llvm::SmallVector<mlir::Value, 16> worklist = graph.values;
            
            // Sort by degree (number of interferences)
            llvm::sort(worklist, [this](mlir::Value a, mlir::Value b) {
                return getDegree(a) > getDegree(b);
            });
            
            // Allocate registers
            for (auto value : worklist) {
                int reg = findAvailableRegister(value);
                if (reg >= 0) {
                    allocation[value] = reg;
                } else {
                    spilled.push_back(value);
                }
            }
        }
        
        int getDegree(mlir::Value value) {
            int degree = 0;
            for (auto other : graph.values) {
                if (graph.interferes(value, other)) {
                    degree++;
                }
            }
            return degree;
        }
        
        int findAvailableRegister(mlir::Value value) {
            llvm::BitVector used(maxRegisters);
            
            for (auto other : graph.values) {
                if (graph.interferes(value, other)) {
                    auto it = allocation.find(other);
                    if (it != allocation.end()) {
                        used.set(it->second);
                    }
                }
            }
            
            for (int i = 0; i < maxRegisters; ++i) {
                if (!used[i]) {
                    return i;
                }
            }
            
            return -1; // Need to spill
        }
    };
    
    void applyAllocation(mlir::func::FuncOp func, RegisterAllocator& allocator) {
        // Apply register allocation decisions
        func.walk([&](mlir::Operation* op) {
            for (auto result : op->getResults()) {
                int reg = allocator.getRegister(result);
                if (reg >= 0) {
                    // Set register attribute
                    op->setAttr("register", 
                               mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(op->getContext(), 32),
                                   reg));
                }
            }
        });
    }
    
    void handleSpills(mlir::func::FuncOp func, RegisterAllocator& allocator) {
        // Handle spilled values by inserting loads/stores
        for (auto value : allocator.spilled) {
            insertSpillCode(value);
        }
    }
    
    void insertSpillCode(mlir::Value value) {
        mlir::OpBuilder builder(value.getContext());
        auto op = value.getDefiningOp();
        
        // Allocate spill slot in local memory
        builder.setInsertionPoint(op);
        auto spillSlot = builder.create<mlir::memref::AllocaOp>(
            op->getLoc(),
            mlir::MemRefType::get({1}, value.getType())
        );
        
        // Store after definition
        builder.setInsertionPointAfter(op);
        builder.create<mlir::memref::StoreOp>(
            op->getLoc(), value, spillSlot, 
            mlir::ValueRange{builder.create<mlir::arith::ConstantIndexOp>(
                op->getLoc(), 0)}
        );
        
        // Load before each use
        for (auto& use : llvm::make_early_inc_range(value.getUses())) {
            builder.setInsertionPoint(use.getOwner());
            auto load = builder.create<mlir::memref::LoadOp>(
                use.getOwner()->getLoc(), spillSlot,
                mlir::ValueRange{builder.create<mlir::arith::ConstantIndexOp>(
                    use.getOwner()->getLoc(), 0)}
            );
            use.set(load.getResult());
        }
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createRegisterAllocationPass() {
    return std::make_unique<RegisterAllocationPass>();
}

} // namespace transforms
} // namespace triton