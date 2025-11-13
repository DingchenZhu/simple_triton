// lib/Lexer/Lexer.cpp
#include "Lexer/Lexer.h"
#include <unordered_map>
#include <cctype>

namespace triton {

static std::unordered_map<std::string, TokenType> keywords = {
    {"def", TokenType::KW_DEF},
    {"kernel", TokenType::KW_KERNEL},
    {"for", TokenType::KW_FOR},
    {"if", TokenType::KW_IF},
    {"else", TokenType::KW_ELSE},
    {"return", TokenType::KW_RETURN},
    {"import", TokenType::KW_IMPORT},
    {"tl", TokenType::KW_TL},
    {"load", TokenType::KW_LOAD},
    {"store", TokenType::KW_STORE},
    {"arange", TokenType::KW_ARANGE},
    {"program_id", TokenType::KW_PROGRAM_ID},
    {"constexpr", TokenType::KW_CONSTEXPR},
    {"float32", TokenType::KW_FLOAT32},
    {"float16", TokenType::KW_FLOAT16},
    {"int32", TokenType::KW_INT32},
    {"int64", TokenType::KW_INT64},
    {"bool", TokenType::KW_BOOL}
};

Lexer::Lexer(const std::string& src) 
    : source(src), current(0), line(1), column(1) {}

std::vector<Token> Lexer::tokenize() {
    while (!isAtEnd()) {
        skipWhitespace();
        if (!isAtEnd()) {
            tokens.push_back(scanToken());
        }
    }
    tokens.push_back(Token(TokenType::EOF_TOKEN, "", line, column));
    return tokens;
}

Token Lexer::scanToken() {
    size_t startLine = line;
    size_t startColumn = column;
    
    char c = advance();
    
    switch (c) {
        case '(': return Token(TokenType::LEFT_PAREN, "(", startLine, startColumn);
        case ')': return Token(TokenType::RIGHT_PAREN, ")", startLine, startColumn);
        case '[': return Token(TokenType::LEFT_BRACKET, "[", startLine, startColumn);
        case ']': return Token(TokenType::RIGHT_BRACKET, "]", startLine, startColumn);
        case '{': return Token(TokenType::LEFT_BRACE, "{", startLine, startColumn);
        case '}': return Token(TokenType::RIGHT_BRACE, "}", startLine, startColumn);
        case ',': return Token(TokenType::COMMA, ",", startLine, startColumn);
        case '.': return Token(TokenType::DOT, ".", startLine, startColumn);
        case ':': return Token(TokenType::COLON, ":", startLine, startColumn);
        case ';': return Token(TokenType::SEMICOLON, ";", startLine, startColumn);
        case '+': return Token(TokenType::OP_PLUS, "+", startLine, startColumn);
        case '-': 
            if (peek() == '>') {
                advance();
                return Token(TokenType::ARROW, "->", startLine, startColumn);
            }
            return Token(TokenType::OP_MINUS, "-", startLine, startColumn);
        case '*': return Token(TokenType::OP_MUL, "*", startLine, startColumn);
        case '/': 
            if (peek() == '/') {
                skipComment();
                return scanToken();
            }
            return Token(TokenType::OP_DIV, "/", startLine, startColumn);
        case '%': return Token(TokenType::OP_MOD, "%", startLine, startColumn);
        case '=':
            if (peek() == '=') {
                advance();
                return Token(TokenType::OP_EQ, "==", startLine, startColumn);
            }
            return Token(TokenType::OP_ASSIGN, "=", startLine, startColumn);
        case '!':
            if (peek() == '=') {
                advance();
                return Token(TokenType::OP_NE, "!=", startLine, startColumn);
            }
            break;
        case '<':
            if (peek() == '=') {
                advance();
                return Token(TokenType::OP_LE, "<=", startLine, startColumn);
            }
            return Token(TokenType::OP_LT, "<", startLine, startColumn);
        case '>':
            if (peek() == '=') {
                advance();
                return Token(TokenType::OP_GE, ">=", startLine, startColumn);
            }
            return Token(TokenType::OP_GT, ">", startLine, startColumn);
        case '@':
            return Token(TokenType::DECORATOR, "@", startLine, startColumn);
        case '"':
            return scanString();
        default:
            if (std::isdigit(c)) {
                current--;
                column--;
                return scanNumber();
            }
            if (std::isalpha(c) || c == '_') {
                current--;
                column--;
                return scanIdentifier();
            }
    }
    
    return Token(TokenType::ERROR, std::string(1, c), startLine, startColumn);
}

Token Lexer::scanIdentifier() {
    size_t start = current;
    size_t startLine = line;
    size_t startColumn = column;
    
    while (std::isalnum(peek()) || peek() == '_') {
        advance();
    }
    
    std::string value = source.substr(start, current - start);
    
    auto it = keywords.find(value);
    TokenType type = (it != keywords.end()) ? it->second : TokenType::IDENTIFIER;
    
    return Token(type, value, startLine, startColumn);
}

Token Lexer::scanNumber() {
    size_t start = current;
    size_t startLine = line;
    size_t startColumn = column;
    
    while (std::isdigit(peek())) {
        advance();
    }
    
    if (peek() == '.' && std::isdigit(source[current + 1])) {
        advance(); // consume '.'
        while (std::isdigit(peek())) {
            advance();
        }
        return Token(TokenType::FLOAT_LITERAL, 
                    source.substr(start, current - start), 
                    startLine, startColumn);
    }
    
    return Token(TokenType::INTEGER_LITERAL, 
                source.substr(start, current - start), 
                startLine, startColumn);
}

// ... 其他辅助函数实现 ...

} // namespace triton