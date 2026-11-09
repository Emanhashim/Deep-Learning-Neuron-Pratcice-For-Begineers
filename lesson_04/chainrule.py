def inner_function(x):
    return 4 * x + 6

def outer_derivative(u):
    return 2 * u

def inner_derivative():
    return 4

x = 2

u = inner_function(x)

dy_du = outer_derivative(u)
du_dx = inner_derivative()

dy_dx = dy_du * du_dx

print("Inner function:", u)
print("Outer derivative:", dy_du)
print("Inner derivative:", du_dx)
print("Final derivative:", dy_dx)