# Python Lambda Functions Guide

Lambda functions, also known as anonymous functions, are small, inline functions that are defined on the fly in Python. They can take any number of arguments but can only have one expression.

## 1. Basic Lambda Function
The simplest form of a lambda function takes one argument and applies a single expression to it.

**Syntax:**
```
lambda arguments: expression
```

**Example:**
```python
double = lambda x: x * 2
print(double(5))  # Outputs: 10
```

## 2. Lambda Function with Multiple Arguments
Lambda functions can take any number of arguments.

**Syntax:**
```
lambda arg1, arg2, ...: expression
```

**Example:**
```python
add = lambda x, y: x + y
print(add(5, 3))  # Outputs: 8
```

## 3. Lambda Function in Higher Order Functions
Lambda functions are commonly used with higher-order functions, which take one or more functions as arguments or return one or more functions.

**Example with `map`:**
```python
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))  # [2, 4, 6, 8, 10]
```

**Example with `filter`:**
```python
numbers = [1, 2, 3, 4, 5]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
```

**Example with `sort`:**
```python
words = ['apple', 'banana', 'cherry']
words.sort(key=lambda s: len(s))
print(words)  # Outputs: ['apple', 'cherry', 'banana']
```

Remember, while lambda functions can make code more concise, they can also make code harder to read if used inappropriately. If the function would be complex or long, it's better to define a regular function with `def`.
```
This guide covers the basics of creating and using lambda functions in Python, including using them with higher-order functions like `map`, `filter`, and `sort`. Lambda functions can be used in many other ways as well, and they're a powerful tool for making your code more concise.
