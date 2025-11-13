// include/Lexer/Lexer.h
#pragma once
#include <string>
#include <vector>
#include <memory>

namespace triton {

enum class TokenType {
    // Keywords
    KW_DEF,
    KW_KERNEL,
    KW_FOR,
    KW_IF,
    KW_ELSE,
    KW_RETURN,
    KW_IMPORT,
    
    // Triton specific
    KW_TL,           // tl namespace
    KW_LOAD,         // tl.load
    KW_STORE,        // tl.store
    KW_ARANGE,       // tl.arange
    KW_PROGRAM_ID,   // tl.program_id
    KW_CONSTEXPR,    // tl.constexpr
    
    // Types
    KW_FLOAT32,
    KW_FLOAT16,
    KW_INT32,
    KW_INT64,
    KW_BOOL,
    
    // Operators
    OP_PLUS,
    OP_MINUS,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    OP_ASSIGN,
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_GT,
    OP_LE,
    OP_GE,
    
    // Delimiters
    LEFT_PAREN,
    RIGHT_PAREN,
    LEFT_BRACKET,
    RIGHT_BRACKET,
    LEFT_BRACE,
    RIGHT_BRACE,
    COMMA,
    COLON,
    SEMICOLON,
    DOT,
    ARROW,
    
    // Literals
    IDENTIFIER,
    INTEGER_LITERAL,
    FLOAT_LITERAL,
    STRING_LITERAL,
    
    // Special
    DECORATOR,
    COMMENT,
    NEWLINE,
    INDENT,
    DEDENT,
    EOF_TOKEN,
    ERROR
};

struct Token {
    TokenType type;
    std::string value;
    size_t line;
    size_t column;
    
    Token(TokenType t, const std::string& v, size_t l, size_t c)
        : type(t), value(v), line(l), column(c) {}
};

class Lexer {
private:
    std::string source;
    size_t current;
    size_t line;
    size_t column;
    std::vector<Token> tokens;
    
public:
    explicit Lexer(const std::string& src);
    std::vector<Token> tokenize();
    
private:
    char peek();
    char advance();
    void skipWhitespace();
    void skipComment();
    Token scanToken();
    Token scanIdentifier();
    Token scanNumber();
    Token scanString();
    bool isAtEnd();
};

} // namespace triton