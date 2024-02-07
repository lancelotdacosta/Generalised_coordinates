'''symbolic algebra helpers'''

import sympy as sp
def sympy_det(M):
    dim=M.shape[0]
    if dim==1:
        return M#sp.Matrix([M])
    else:
        det_M=sp.Matrix([0])
        col = range(1,dim)
        for r in range(dim):
            # print(r)
            row=[i for i in range(dim) if i != r]
            det_M += (-1)**r * M[r,0] * sympy_det(M[row,col])
        return det_M