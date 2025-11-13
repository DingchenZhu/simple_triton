// include/Parser/AST.h
#pragma once
#include <memory>
#include <vector>
#include <string>
#include <variant>

namespace triton {
namespace ast {

// Forward declarations
class ASTVisitor;

// Base AST Node
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(ASTVisitor* visitor) = 0;
};

// Type nodes
class Type : public ASTNode {
public:
    enum class TypeKind {
        Float32, Float16, Int32, Int64, Bool,
        Pointer, Tensor
    };
    
    TypeKind kind;
    std::vector<int64_t> shape;  // For tensor types
    std::shared_ptr<Type> elementType;  // For pointer/tensor types
    
    void accept(ASTVisitor* visitor) override;
};

// Expression nodes
class Expr : public ASTNode {
public:
    std::shared_ptr<Type> type;
};

class BinaryOp : public Expr {
public:
    enum class OpKind {
        Add, Sub, Mul, Div, Mod,
        EQ, NE, LT, GT, LE, GE,
        And, Or
    };
    
    OpKind op;
    std::shared_ptr<Expr> left;
    std::shared_ptr<Expr> right;
    
    void accept(ASTVisitor* visitor) override;
};

class UnaryOp : public Expr {
public:
    enum class OpKind { Neg, Not };
    
    OpKind op;
    std::shared_ptr<Expr> operand;
    
    void accept(ASTVisitor* visitor) override;
};

class CallExpr : public Expr {
public:
    std::string callee;
    std::vector<std::shared_ptr<Expr>> args;
    
    void accept(ASTVisitor* visitor) override;
};

class Identifier : public Expr {
public:
    std::string name;
    
    explicit Identifier(const std::string& n) : name(n) {}
    void accept(ASTVisitor* visitor) override;
};

class Literal : public Expr {
public:
    std::variant<int64_t, double, bool> value;
    
    void accept(ASTVisitor* visitor) override;
};

class IndexExpr : public Expr {
public:
    std::shared_ptr<Expr> base;
    std::vector<std::shared_ptr<Expr>> indices;
    
    void accept(ASTVisitor* visitor) override;
};

// Triton-specific expressions
class LoadExpr : public Expr {
public:
    std::shared_ptr<Expr> ptr;
    std::shared_ptr<Expr> mask;
    std::shared_ptr<Expr> other;  // default value
    
    void accept(ASTVisitor* visitor) override;
};

class StoreExpr : public Expr {
public:
    std::shared_ptr<Expr> ptr;
    std::shared_ptr<Expr> value;
    std::shared_ptr<Expr> mask;
    
    void accept(ASTVisitor* visitor) override;
};

class ArangeExpr : public Expr {
public:
    std::shared_ptr<Expr> start;
    std::shared_ptr<Expr> end;
    
    void accept(ASTVisitor* visitor) override;
};

class ProgramIdExpr : public Expr {
public:
    int axis;  // 0, 1, or 2 for x, y, z
    
    void accept(ASTVisitor* visitor) override;
};

// Statement nodes
class Stmt : public ASTNode {};

class ExprStmt : public Stmt {
public:
    std::shared_ptr<Expr> expr;
    
    void accept(ASTVisitor* visitor) override;
};

class AssignStmt : public Stmt {
public:
    std::shared_ptr<Identifier> target;
    std::shared_ptr<Expr> value;
    
    void accept(ASTVisitor* visitor) override;
};

class IfStmt : public Stmt {
public:
    std::shared_ptr<Expr> condition;
    std::vector<std::shared_ptr<Stmt>> thenBlock;
    std::vector<std::shared_ptr<Stmt>> elseBlock;
    
    void accept(ASTVisitor* visitor) override;
};

class ForStmt : public Stmt {
public:
    std::shared_ptr<Identifier> var;
    std::shared_ptr<Expr> start;
    std::shared_ptr<Expr> end;
    std::shared_ptr<Expr> step;
    std::vector<std::shared_ptr<Stmt>> body;
    
    void accept(ASTVisitor* visitor) override;
};

class ReturnStmt : public Stmt {
public:
    std::shared_ptr<Expr> value;
    
    void accept(ASTVisitor* visitor) override;
};

// Function and Kernel nodes
class Parameter {
public:
    std::string name;
    std::shared_ptr<Type> type;
    bool isConstexpr;
};

class Function : public ASTNode {
public:
    std::string name;
    std::vector<Parameter> params;
    std::shared_ptr<Type> returnType;
    std::vector<std::shared_ptr<Stmt>> body;
    bool isKernel;
    
    // Kernel metadata
    struct LaunchConfig {
        std::vector<std::shared_ptr<Expr>> gridDim;
        std::vector<std::shared_ptr<Expr>> blockDim;
    } launchConfig;
    
    void accept(ASTVisitor* visitor) override;
};

class Program : public ASTNode {
public:
    std::vector<std::shared_ptr<Function>> functions;
    
    void accept(ASTVisitor* visitor) override;
};

// Visitor pattern for AST traversal
class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;
    
    virtual void visit(Type* node) = 0;
    virtual void visit(BinaryOp* node) = 0;
    virtual void visit(UnaryOp* node) = 0;
    virtual void visit(CallExpr* node) = 0;
    virtual void visit(Identifier* node) = 0;
    virtual void visit(Literal* node) = 0;
    virtual void visit(IndexExpr* node) = 0;
    virtual void visit(LoadExpr* node) = 0;
    virtual void visit(StoreExpr* node) = 0;
    virtual void visit(ArangeExpr* node) = 0;
    virtual void visit(ProgramIdExpr* node) = 0;
    virtual void visit(ExprStmt* node) = 0;
    virtual void visit(AssignStmt* node) = 0;
    virtual void visit(IfStmt* node) = 0;
    virtual void visit(ForStmt* node) = 0;
    virtual void visit(ReturnStmt* node) = 0;
    virtual void visit(Function* node) = 0;
    virtual void visit(Program* node) = 0;
};

} // namespace ast
} // namespace triton