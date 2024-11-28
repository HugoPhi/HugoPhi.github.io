# Introduction

中文：[zh](./index_zh.html)

[TOC]

## Why Rust?

​	Rust is a systems programming language designed to provide memory safety, concurrency, and high performance while avoiding common issues found in traditional systems programming languages like C and C++, such as null pointer dereferencing, memory leaks, and data races. Rust's design philosophy emphasizes **safety**, **concurrency**, and **speed**, making it an ideal choice for building efficient and secure applications. 
​	Rust is developed by Mozilla and is open-source. Its syntax is similar to modern programming languages and offers high expressiveness. One of Rust's core features is its **Ownership** and **Borrowing** system, which allows for automatic memory management without the need for garbage collection (GC), avoiding problems like memory leaks and dangling pointers. 
​	Rust supports multiple programming paradigms, including object-oriented programming (through structs and traits), functional programming (via closures and higher-order functions), and concurrent programming (using lightweight threads and asynchronous programming), offering powerful expressiveness and flexibility.

### 🔥 Main Features

- **Memory Safety**: Rust's ownership system and borrowing checker ensure memory safety, preventing common memory issues such as dangling pointers and memory leaks.
- **Concurrency**: Rust's concurrency model eliminates data races by utilizing ownership, borrowing, and locks, providing thread-safe concurrency support.
- **Zero-Cost Abstractions**: Rust's abstractions (such as generics, traits, and closures) are highly efficient, with minimal runtime overhead, ensuring high execution performance.
- **Modern Syntax**: Rust has a modern, easy-to-understand syntax that is still suitable for low-level systems programming.
- **No Garbage Collection**: Rust manages memory through ownership and lifetimes, without relying on a garbage collector, leading to better performance.

## What You Will Learn in This Guide?

​	This guide introduces some advanced features of Rust programming, including: generics, polymorphism, metaprogramming, functional programming, metaprogramming, high concurrency, and parallel programming. It also covers additional topics such as web development, scientific computing, and machine learning. Here is the table of contents:

### Advanced Features

- [Generic & Trait](./post/1/index.html)
- [Inheritance & Polymorphism](./post/2/index.html)
- [Meta Programming](./post/3/index.html)
- [Memory Management & Ownership](./post/4/index.html)
- [Error Trace](./post/5/index.html)
- [Compile Optimization & System Programming](./post/6/index.html)

### Topics

- [Science Computing for Rust](./post/7/index.html)
- [Machine Learning by Rust](./post/8/index.html)
- [Web Programming](./post/9/index.html)

## Things You Should Know Before Starting

Before diving into this guide, it is recommended that you have a basic understanding of the following:

- Rust Compilation Process
- Cargo Package Management
- Basic Rust Syntax:
  - Variable Declarations
    - Immutable Variables (`let`)
    - Mutable Variables (`let mut`)
    - Constants (`const`)
    - Type Inference
    - Explicit Type Declaration
    - Destructuring Assignment
  - Data Types
    - Scalar Types
      - Integer Types (`i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`, `usize`, `isize`)
      - Floating-Point Types (`f32`, `f64`)
      - Boolean Type (`bool`)
      - Character Type (`char`)
    - Compound Types
      - Tuples (`tuple`)
      - Arrays (`array`)
      - Slices (`slice`)
  - Control Flow
    - Conditional Statements: `if` expressions, `else` statements, `else if` statements
    - Looping Statements: `loop`, `while`, `for`
    - Match Expressions
      - `match` statement
      - Pattern Matching
  - Functions
    - Function Declaration
      - Function Signature
      - Function Return Values
      - Multiple Return Values (Tuples)
    - Function Scope
      - Parameter Passing (by Value vs. by Reference)
      - Anonymous Functions (Closures)
    - Variable Parameters
  - Ownership & Borrowing
    - Ownership
      - Ownership Transfer
      - Borrowing Ownership (Immutable Borrowing, Mutable Borrowing)
    - Borrowing
      - Immutable Borrowing (`&T`)
      - Mutable Borrowing (`&mut T`)
    - Lifetimes
      - Lifetime Annotations
      - Lifetime Inference
  - Structs
    - Struct Definition
      - Similar to Classes in OOP but without Inheritance
      - Struct Fields Definition
      - Struct Initialization
    - Methods and Associated Functions
      - `impl` Block
      - `self` Parameter
    - Struct Pattern Matching
  - Enums
    - Enum Definition
      - Variants
      - Enum Types with Data (e.g., Tuples or Structs)
    - `match` Statement and Enums
  - Error Handling
    - `Result` Enum
      - `Ok` and `Err`
      - Error Propagation (`?` Operator)
    - `Option` Enum
      - `Some` and `None`
  - Modules & Packages
    - Module System (`mod`)
      - File Modules
      - Nested Modules
      - `use` Import
    - Packages and Dependency Management
      - `Cargo.toml` Configuration File
      - Third-Party Libraries (Crates)

​	It is recommended that you first familiarize yourself with these syntax concepts through dialogue with a large language model, and for more detailed knowledge, you can refer to Rust's official documentation or read its source code.