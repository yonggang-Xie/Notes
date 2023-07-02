# Python List Comprehensions Guide

## 1. Basic List Comprehension
The simplest form of list comprehension generates a list from another list without any conditions or transformations.

**Syntax:**
```markdown
[expression for item in iterable]
```

**Example:**
```python
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]  # [2, 4, 6, 8, 10]
```

## 2. List Comprehension with Condition
You can include an `if` condition in a list comprehension to select only certain items.

**Syntax:**
```markdown
[expression for item in iterable if condition]
```

**Example:**
```python
numbers = [1, 2, 3, 4, 5]
evens = [x for x in numbers if x % 2 == 0]  # [2, 4]
```

## 3. List Comprehension with Transformation
You can apply a transformation to each item using the expression at the start of the list comprehension.

**Syntax:**
```markdown
[expression(item) for item in iterable]
```

**Example:**
```python
numbers = [1, 2, 3, 4, 5]
strings = [str(x) for x in numbers]  # ['1', '2', '3', '4', '5']
```

## 4. List Comprehension with if-else Condition
You can use an `if-else` clause in the expression part of a list comprehension to create more complex transformations.

**Syntax:**
```markdown
[expression1 if condition else expression2 for item in iterable]
```

**Example:**
```python
numbers = [1, 2, 3, 4, 5]
new_numbers = [x if x > 2 else x * 10 for x in numbers]  # [10, 20, 3, 4, 5]
```

---

Remember, while list comprehensions are powerful, they can become difficult to read if you try to do too much in a single expression. If a list comprehension would be overly complex, it might be better to use a traditional loop instead.
