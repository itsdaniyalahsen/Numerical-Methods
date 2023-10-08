from root_finding import bisection_solve, incremental_search, incremental_bisection, fixed_point_iteration, newton_raphson, secant_solve
from math import sqrt, sin, pi

def fn(x):
    return  sin(x) 

print(bisection_solve(fn, -pi/2, pi/2));
print(incremental_search(fn, -pi, pi, 1e-6))
print(incremental_bisection(fn, -pi, pi, 1e-2, 1e-6))
print(fixed_point_iteration(fn, 0))
print(newton_raphson(fn, 2, 1e-6))
print(secant_solve(fn, 2))
