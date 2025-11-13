// lib/Transforms/PassPipeline.cpp (更新版)
void addTritonBufferizationPipeline(mlir::PassManager& pm,
                                   const TritonBufferizationOptions& options) {
    // Pre-bufferization optimizations
    pm.addPass(createInPlaceBufferizationPass());
    
    // Main bufferization
    pm.addPass(createTritonBufferizationPass(options));
    
    // Post-bufferization optimizations
    pm.addPass(createBufferHoistingPass());
    pm.addPass(createBufferReusePass());
    pm.addPass(createBufferDeallocationPass());
    
    // Cleanup
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createDeadCodeEliminationPass());
}
void addTritonOptimizationPipeline(mlir::PassManager& pm, int optimizationLevel) {
    // Level 0: No optimization
    if (optimizationLevel == 0) {
        return;
    }
    if (optimizationLevel >= 1) {
        TritonBufferizationOptions bufferOpts;
        bufferOpts.hoistAllocations = (optimizationLevel >= 2);
        bufferOpts.promoteBufferToStack = (optimizationLevel >= 2);
        
        addTritonBufferizationPipeline(pm, bufferOpts);
    }
    // Level 1: Basic optimizations
    if (optimizationLevel >= 1) {
        // Basic cleanup
        pm.addPass(createDeadCodeEliminationPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createCommonSubexpressionEliminationPass());
        pm.addPass(createConstantFoldingPass());
        pm.addPass(createAlgebraicSimplificationPass());
        pm.addPass(createDeadCodeEliminationPass());
    }
    
    // Level 2: Advanced optimizations
    if (optimizationLevel >= 2) {
        // Memory optimizations
        pm.addPass(createRedundantLoadEliminationPass());
        pm.addPass(createMemoryCoalescingPass());
        pm.addPass(createSharedMemoryOptimizationPass());
        
        // Loop optimizations
        pm.addPass(createLoopFusionPass());
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
        pm.addPass(createVectorizationPass());
        
        // Operator fusion
        pm.addPass(createOperatorFusionPass());
        
        // Layout optimization
        pm.addPass(createLayoutOptimizationPass());
        
        // Clean up
        pm.addPass(createCommonSubexpressionEliminationPass());
        pm.addPass(createDeadCodeEliminationPass());
    }
    
    // Level 3: Aggressive optimizations
    if (optimizationLevel >= 3) {
        // Tensor core mapping
        pm.addPass(createTensorCoreMappingPass());
        
        // Aggressive fusion
        pm.addPass(createOperatorFusionPass());
        
        // Loop transformations
        pm.addPass(mlir::createLoopUnrollPass());
        pm.addPass(createLoopFusionPass());
        pm.addPass(createVectorizationPass());
        
        // Register allocation
        pm.addPass(createRegisterAllocationPass());
        
        // Re-run layout optimization after transformations
        pm.addPass(createLayoutOptimizationPass());
        
        // Re-run memory optimizations
        pm.addPass(createSharedMemoryOptimizationPass());
        pm.addPass(createMemoryCoalescingPass());
        
        // Final cleanup
        pm.addPass(createCommonSubexpressionEliminationPass());
        pm.addPass(createDeadCodeEliminationPass());
        pm.addPass(mlir::createCanonicalizerPass());
    }
    
    // GPU-specific passes (always applied for GPU targets)
    pm.addPass(createTensorCoreMappingPass());
    pm.addPass(createSharedMemoryOptimizationPass());
    pm.addPass(createRegisterAllocationPass());
}