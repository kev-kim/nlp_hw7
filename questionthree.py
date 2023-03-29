# Question 1
import numpy as np
u = np.full((2, 3), 2)
q1 = np.einsum('ij->j', u)
q1_2 = np.einsum('ij->i', u)

# Question 2

def test(original, modified, test_name):
    if torch.equal(original, modified):
        print("PASS: " + test_name)
        return 1
    else:
        print("FAIL: " + test_name)
        return 0

cumul = 0

import torch
x = torch.rand(4, 4)
y0 = torch.rand(5)
y1 = torch.rand(4)
z0 = torch.rand(3, 2, 5)
z1 = torch.rand(3, 5, 4)
w = torch.rand(2, 3, 4, 5)
r0 = torch.rand(2, 5)
r1 = torch.rand(3, 5, 4)
r2 = torch.rand(2, 4)
s0 = torch.rand(2, 3, 5, 7)
s1 = torch.rand(11, 3, 17, 5)

# identity
a0 = torch.einsum('i', y0)
a1 = torch.einsum('ij', x)
a2 = torch.einsum('ijk', z0)

a0_m = y0
a1_m = x
a2_m = z0

cumul += test(a0, a0_m, "identity 1")
cumul += test(a1, a1_m, "identity 2")
cumul += test(a2, a2_m, "identity 3")

# permute
b0=torch.einsum('ij->ji',x)
b1=torch.einsum('ba',x)
b2=torch.einsum('jki',z0)
b3=torch.einsum('ijk->kij',z0)
b4=torch.einsum('kjil',w)
b5=torch.einsum('...ij->...ji',w)
b6=w

b0_m = x.permute(1,0)
b1_m = x.transpose(1,0)
b2_m = z0.permute(2,0,1)
b3_m = z0.permute(2,0,1)
b4_m = w.permute(2,1,0,3)
b5_m = w.permute(*range(w.dim())[:-2], -1, -2)
b6_m = w.permute(*range(w.dim())[::-1])

cumul += test(b0, b0_m, "permute 1")
cumul += test(b1, b1_m, "permute 2")
cumul += test(b2, b2_m, "permute 3")
cumul += test(b3, b3_m, "permute 4")
cumul += test(b4, b4_m, "permute 5")
cumul += test(b5, b5_m, "permute 6")
cumul += test(b6, b6_m, "permute 7")

# trace
c=torch.einsum('ii',x)
c_m = x.trace()

cumul += test(c, c_m, "trace 1 (may fail but is correct)")

# sum
d0=torch.einsum('ij->',x)
d1=torch.einsum('xyz->',z0)
d2=torch.einsum('ijkl->',w)

d0_m = x.sum()
d1_m = z0.sum()
d2_m = w.sum()

cumul += test(d0, d0_m, "sum 1")
cumul += test(d1, d1_m, "sum 2")
cumul += test(d2, d2_m, "sum 3")

# sum axis
e0=torch.einsum('ijk->i',z0)
e1=torch.einsum('ijk->j',z0)
e2=torch.einsum('ijk->ij',z0)

e0_m = z0.sum(dim=(1,2))
e1_m = z0.sum(dim=(0,2))
e2_m = z0.sum(dim=(2))

cumul += test(e0, e0_m, "sum axis 1")
cumul += test(e1, e1_m, "sum axis 2")
cumul += test(e2, e2_m, "sum axis 3")

# matrix-vector
f0=torch.einsum('ij,j->i',r0,y0)
f1=torch.einsum('i,jki->jk',y1,r1)

f0_m = torch.matmul(r0, y0)
f1_m = torch.matmul(r1, y1)

cumul += test(f0, f0_m, "matrix-vector 1 (may fail but is correct)")
cumul += test(f1, f1_m, "matrix-vector 2")

# vector-vector outer product
g0=torch.einsum('i,j->ij',y0,y1)
g1=torch.einsum('a,b,c,d->abcd',y0,y1,y0,y1)

g0_m = torch.outer(y0, y1)
g1_m = y0

cumul += test(g0, g0_m, "vector-vector 1")
cumul += test(g1, g1_m, "vector-vector 2")

# batch mm
h0=torch.einsum('bij,bjk->bik',z0,z1)
h1=torch.einsum('bjk,bij->bik',z1,z0)

h0_m = torch.matmul(z0, z1)
h1_m = z1

cumul += test(h0, h0_m, "batch mm 1")
cumul += test(h1, h1_m, "batch mm 2")

# bilinear
i=torch.einsum('bn,anm,bm->ba',r0,r1,r2)

i_m = r0

cumul += test(i, i_m, "bilinear 1")

# tensor contraction
j=torch.einsum('pqrs,tqvr->pstv',s0,s1)

j_m = s0

cumul += test(j, j_m, "tensor contraction 1")

if cumul == 25:
    print("ALL TESTS PASSED!")
else:
    print(f"{25 - cumul} tests failed.")