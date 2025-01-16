print("hello world!")
print("こんにちわ世界！")

# Type Hintsの練習


def add(a: int, b: int) -> int:
    return a + b


print(add(1, 2))


# Docstringの練習
def fib(n: int) -> int:
    """calc fibonacci number

    Args:
        n (int): index of fibonacci number

    Returns:
        int: fibonacci number
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


print(fib(10))
