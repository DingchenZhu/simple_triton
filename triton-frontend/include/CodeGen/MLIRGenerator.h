// include/CodeGen/MLIRGenerator.h
#pragma once
#include "Parser/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace triton {

class MLIRGenerator : public ast::ASTVisitor {
private:
    mlir::MLIRContext* context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    // Symbol table for variable lookup
    llvm::StringMap<mlir::Value> symbolTable;
    
    // Current insertion point
    mlir::Block* currentBlock;
    
    // Generated values
    mlir::Value currentValue;
    
public:
    explicit MLIRGenerator(mlir::MLIRContext* ctx);
    
    mlir::ModuleOp generate(ast::Program* program);
    
    // Visitor implementations
    void visit(ast::Type* node) override;
    void visit(ast::BinaryOp* node) override;
    void visit(ast::UnaryOp* node) override;
    void visit(ast::CallExpr* node) override;
    void visit(ast::Identifier* node) override;
    void visit(ast::Literal* node) override;
    void visit(ast::IndexExpr* node) override;
    void visit(ast::LoadExpr* node) override;
    void visit(ast::StoreExpr* node) override;
    void visit(ast::ArangeExpr* node) override;
    void visit(ast::ProgramIdExpr* node) override;
    void visit(ast::ExprStmt* node) override;
    void visit(ast::AssignStmt* node) override;
    void visit(ast::IfStmt* node) override;
    void visit(ast::ForStmt* node) override;
    void visit(ast::ReturnStmt* node) override;
    void visit(ast::Function* node) override;
    void visit(ast::Program* node) override;
    
private:
    mlir::Type convertType(ast::Type* type);
    mlir::Value getValue(ast::Expr* expr);
    mlir::FuncOp generateFunction(ast::Function* func);
    mlir::FuncOp generateKernel(ast::Function* kernel);
};

} // namespace triton