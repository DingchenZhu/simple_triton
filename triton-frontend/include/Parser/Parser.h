// include/Parser/Parser.h
#pragma once
#include "Lexer/Lexer.h"
#include "Parser/AST.h"
#include <memory>
#include <vector>

namespace triton {

class Parser {
private:
    std::vector<Token> tokens;
    size_t current;
    
public:
    explicit Parser(const std::vector<Token>& toks);
    std::shared_ptr<ast::Program> parse();
    
private:
    // Helper functions
    Token peek();
    Token previous();
    Token advance();
    bool check(TokenType type);
    bool match(std::initializer_list<TokenType> types);
    Token consume(TokenType type, const std::string& message);
    bool isAtEnd();
    
    // Parsing functions
    std::shared_ptr<ast::Function> parseFunction();
    std::shared_ptr<ast::Function> parseKernel();
    std::vector<ast::Parameter> parseParameters();
    std::shared_ptr<ast::Type> parseType();
    
    // Statement parsing
    std::shared_ptr<ast::Stmt> parseStatement();
    std::shared_ptr<ast::Stmt> parseExprStatement();
    std::shared_ptr<ast::Stmt> parseAssignStatement();
    std::shared_ptr<ast::Stmt> parseIfStatement();
    std::shared_ptr<ast::Stmt> parseForStatement();
    std::shared_ptr<ast::Stmt> parseReturnStatement();
    
    // Expression parsing
    std::shared_ptr<ast::Expr> parseExpression();
    std::shared_ptr<ast::Expr> parseAssignment();
    std::shared_ptr<ast::Expr> parseLogicalOr();
    std::shared_ptr<ast::Expr> parseLogicalAnd();
    std::shared_ptr<ast::Expr> parseEquality();
    std::shared_ptr<ast::Expr> parseComparison();
    std::shared_ptr<ast::Expr> parseTerm();
    std::shared_ptr<ast::Expr> parseFactor();
    std::shared_ptr<ast::Expr> parseUnary();
    std::shared_ptr<ast::Expr> parsePostfix();
    std::shared_ptr<ast::Expr> parsePrimary();
    
    // Triton-specific parsing
    std::shared_ptr<ast::Expr> parseTritonBuiltin();
    std::shared_ptr<ast::LoadExpr> parseLoad();
    std::shared_ptr<ast::StoreExpr> parseStore();
    std::shared_ptr<ast::ArangeExpr> parseArange();
    std::shared_ptr<ast::ProgramIdExpr> parseProgramId();
};

} // namespace triton