// include/Dialect/TritonDialect.h
#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace triton {

class TritonDialect : public Dialect {
public:
    explicit TritonDialect(MLIRContext* context);
    static StringRef getDialectNamespace() { return "tt"; }
    
    // Parse/Print methods for custom types
    Type parseType(DialectAsmParser& parser) const override;
    void printType(Type type, DialectAsmPrinter& printer) const override;
};

// Triton-specific types
class PointerType : public Type::TypeBase<PointerType, Type, TypeStorage> {
public:
    using Base::Base;
    
    static PointerType get(MLIRContext* context, Type pointeeType, 
                          unsigned addressSpace = 1);
    
    Type getPointeeType() const;
    unsigned getAddressSpace() const;
};

// Triton operations
class LoadOp : public Op<LoadOp, OpTrait::OneResult, 
                         OpTrait::OneOperand,
                         OpTrait::MemoryEffect<MemoryEffects::Read>> {
public:
    using Op::Op;
    
    static StringRef getOperationName() { return "tt.load"; }
    
    static void build(OpBuilder& builder, OperationState& state,
                     Type resultType, Value ptr, Value mask = nullptr,
                     Value other = nullptr);
    
    LogicalResult verify();
};

class StoreOp : public Op<StoreOp, OpTrait::ZeroResult,
                         OpTrait::AtLeastNOperands<2>::Impl,
                         OpTrait::MemoryEffect<MemoryEffects::Write>> {
public:
    using Op::Op;
    
    static StringRef getOperationName() { return "tt.store"; }
    
    static void build(OpBuilder& builder, OperationState& state,
                     Value ptr, Value value, Value mask = nullptr);
    
    LogicalResult verify();
};

class ArangeOp : public Op<ArangeOp, OpTrait::OneResult,
                          OpTrait::AtLeastNOperands<1>::Impl> {
public:
    using Op::Op;
    
    static StringRef getOperationName() { return "tt.arange"; }
    
    static void build(OpBuilder& builder, OperationState& state,
                     Type resultType, Value start, Value end);
    
    LogicalResult verify();
};

class GetProgramIdOp : public Op<GetProgramIdOp, OpTrait::OneResult,
                                OpTrait::ZeroOperands> {
public:
    using Op::Op;
    
    static StringRef getOperationName() { return "tt.get_program_id"; }
    
    static void build(OpBuilder& builder, OperationState& state,
                     Type resultType, int32_t axis);
    
    int32_t getAxis();
    LogicalResult verify();
};

class DotOp : public Op<DotOp, OpTrait::OneResult,
                       OpTrait::NOperands<2>::Impl> {
public:
    using Op::Op;
    
    static StringRef getOperationName() { return "tt.dot"; }
    
    static void build(OpBuilder& builder, OperationState& state,
                     Type resultType, Value lhs, Value rhs);
    
    LogicalResult verify();
};

// Register all operations
void registerTritonOps();

} // namespace triton
} // namespace mlir