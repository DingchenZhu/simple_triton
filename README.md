```cpp
// project structure
triton-frontend/
├── include/
│   ├── Lexer/
│   │   └── Lexer.h
│   ├── Parser/
│   │   ├── Parser.h
│   │   └── AST.h
│   ├── Sema/
│   │   └── SemanticAnalyzer.h
│   ├── CodeGen/
│   │   └── MLIRGenerator.h
│   └── Dialect/
│       └── TritonDialect.h
├── lib/
│   ├── Lexer/
│   ├── Parser/
│   ├── Sema/
│   ├── CodeGen/
│   └── Dialect/
└── tools/
    └── triton-compiler/
        └── main.cpp
```
