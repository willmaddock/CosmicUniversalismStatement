import sys

sys.setrecursionlimit(10000)  # Increase recursion limit, but won't work for huge cases

def tetrate_iterative(base, height):
    """Computes base^^height iteratively."""
    result = base
    for _ in range(height - 1):  # Avoids deep recursion
        result = base ** result  # Exponentiation chain
        if result > 10**100:  # Prevents overflowing memory
            return "Number is too large to compute"
    return result

n = 65536  # Replace with 65536 for the full case, but it's too large for direct computation
result = tetrate_iterative(2, n)

print(f"2^^{n} = {result}")