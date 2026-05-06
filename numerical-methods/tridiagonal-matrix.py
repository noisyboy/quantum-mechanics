import numpy as np 
import numpy.linalg as alg
N= 5
main_dia = [5]*N
upper_dia =[4]*(N-1)
lower_dia=[4]*(N-1)

Z = np.diag(main_dia,0)+\
      np.diag(upper_dia,+1) +\
          np.diag(lower_dia,-1)

eigen_val,eigen_vec= alg.eigh(Z) #hermitian eigen val
print(f"The tridiagonal matrix is \n {Z}")
print(f"The respective eigen value are\n {eigen_val}")
print(f"The respective eigen vectors are \n{eigen_vec}")