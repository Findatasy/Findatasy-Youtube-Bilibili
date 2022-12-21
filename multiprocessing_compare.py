import time
from multiprocessing import Pool

def f(n):
    # a simple function to test the speed of multiprocessing
    # return n*n
    
    # a computationally intensive function (fibonacci) to test the speed of multiprocessing
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return f(n-1) + f(n-2)

# decorator to count the time taken by a function
def count_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

# run the function with multiprocessing
@count_time
def some_function():
    with Pool(10) as p:
        print(p.map(f, [35, 36, 37]))

# run the funtion without multiprocessing (with simple loop)
@count_time
def some_function2():
    for i in [35, 36, 37]:
        print(f(i))

if __name__ == '__main__':
    some_function()
    some_function2()
