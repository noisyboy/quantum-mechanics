import numpy as np 
import numpy.linalg as alg

# defining physics params 
N=1000 # inner Elements
x_max=5 # boundary
x_min=-5
hbar=1
m=1
dx= (x_max-x_min)/(N+1)
t = hbar**2/(2*m *dx**2)

V=np.zeros(N)
main_dia = 2*t + V
off_dia = -t*np.ones(N-1)
H= np.diag(main_dia,0) + np.diag(off_dia,1) + np.diag(off_dia,-1)
print(f"The tridiagonal Hamiltonian is :\n{H}")
eigenval, eigenvec = alg.eigh(H)

print(f"The eigen values are: \n{eigenval}")