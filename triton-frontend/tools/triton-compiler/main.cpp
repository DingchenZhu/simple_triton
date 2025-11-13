// tools/triton-compiler/main.cpp
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "CodeGen/MLIRGenerator.h"
#include "Dialect/TritonDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input triton file>"),
    cl::init("-"),
    cl::value_desc("filename"));

static cl::opt<std::string> outputFilename(
    "o",
    cl::desc("Output filename"),
    cl::value_desc("filename"),
    cl::init("-"));

static cl::opt<bool> emitMLIR(
    "emit-mlir",
    cl::desc("Emit MLIR IR"),
    cl::init(false));

static cl::opt<bool> emitLLVM(
    "emit-llvm",
    cl::desc("Emit LLVM IR"),
    cl::init(false));

static cl::opt<bool> optimize(
    "O",
    cl::desc("Enable optimizations"),
    cl::init(false));

int main(int argc, char** argv) {
    InitLLVM y(argc, argv);
    
    // Parse command line options
    cl::ParseCommandLineOptions(argc, argv, "Triton Compiler\n");
    
    // Read input file
    std::string sourceCode;
    if (inputFilename == "-") {
        sourceCode = std::string(
            std::istreambuf_iterator<char>(std::cin),
            std::istreambuf_iterator<char>()
        );
    } else {
        std::ifstream file(inputFilename);
        if (!file) {
            llvm::errs() << "Error: Cannot open input file: " << inputFilename << "\n";
            return 1;
        }
        sourceCode = std::string(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );
    }
    
    // Initialize MLIR context
    MLIRContext context;
    context.getOrLoadDialect<triton::TritonDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<arith::ArithmeticDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    
    // Lexical analysis
    triton::Lexer lexer(sourceCode);
    auto tokens = lexer.tokenize();
    
    // Syntax analysis
    triton::Parser parser(tokens);
    auto ast = parser.parse();
    if (!ast) {
        llvm::errs() << "Error: Failed to parse input\n";
        return 1;
    }
    
    // Generate MLIR
    triton::MLIRGenerator generator(&context);
    auto module = generator.generate(ast.get());
    
    if (!module) {
        llvm::errs() << "Error: Failed to generate MLIR\n";
        return 1;
    }
    
    // Verify the module
    if (failed(verify(module))) {
        llvm::errs() << "Error: Module verification failed\n";
        module.dump();
        return 1;
    }
    
    // Set up pass manager
    PassManager pm(&context);
    
    if (optimize) {
        // Add optimization passes
        pm.addPass(createInlinerPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createLoopFusionPass());
        pm.addPass(createAffineLoopUnrollPass());
    }
    
    // Run passes
    if (failed(pm.run(module))) {
        llvm::errs() << "Error: Pass manager failed\n";
        return 1;
    }
    
    // Output generation
    std::error_code ec;
    auto output = std::make_unique<ToolOutputFile>(outputFilename, ec, sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Error: Cannot open output file: " << ec.message() << "\n";
        return 1;
    }
    
    if (emitMLIR) {
        module.print(output->os());
    } else if (emitLLVM) {
        // Lower to LLVM IR
        registerLLVMDialectTranslation(context);
        
        // Add lowering passes
        PassManager loweringPM(&context);
        loweringPM.addPass(createConvertToLLVMPass());
        
        if (failed(loweringPM.run(module))) {
            llvm::errs() << "Error: Failed to lower to LLVM\n";
            return 1;
        }
        
        // Translate to LLVM IR
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule) {
            llvm::errs() << "Error: Failed to translate to LLVM IR\n";
            return 1;
        }
        
        llvmModule->print(output->os(), nullptr);
    } else {
        // Default: emit MLIR
        module.print(output->os());
    }
    
    output->keep();
    return 0;
}