import itertools

import torch

k = 10
n = 5
coefficients = torch.rand(k)
gs = [torch.rand(k) for _ in range(n)]

# method 1
sum1 = torch.zeros(k)
for i in range(n):
    products = 1
    for j in range(n):
        if i != j:
            products *= (gs[j] * coefficients).sum()
    sum1 += products * gs[i]
print("Method 1:")
print(sum1.shape)
print(sum1)
print()

# method 2. Outer product method
outer_products_gs = torch.einsum('a,b,c,d,e->abcde', gs[0], gs[1], gs[2], gs[3], gs[4])
outer_products_cs = torch.einsum('a,b,c,d->abcd', coefficients, coefficients, coefficients, coefficients)
sum2 =      torch.einsum(('abcde,abcd->e'), outer_products_gs, outer_products_cs) + \
            torch.einsum(('abcde,abce->d'), outer_products_gs, outer_products_cs) + \
            torch.einsum(('abcde,abde->c'), outer_products_gs, outer_products_cs) + \
            torch.einsum(('abcde,acde->b'), outer_products_gs, outer_products_cs) + \
            torch.einsum(('abcde,bcde->a'), outer_products_gs, outer_products_cs)
print("Method 2:")
print(sum2.shape)
print(sum2)
print()

# method 3. Outer product but sum the cs
combined = (outer_products_gs +
            outer_products_gs.transpose(-1, -2) +
            outer_products_gs.transpose(-1, -3) +
            outer_products_gs.transpose(-1, -4) +
            outer_products_gs.transpose(-1, -5))
# combined = torch.zeros(k, k, k, k, k)
# combined[:,:,:,:,0] += outer_products_cs
# combined[:,:,:,0,:] += outer_products_cs
# combined[:,:,0,:,:] += outer_products_cs
# combined[:,0,:,:,:] += outer_products_cs
# combined[0,:,:,:,:] += outer_products_cs
# combined = (outer_products_cs.expand(10, 10, 10, 10, k) +
#             outer_products_cs.expand(10, 10, 10, k, 10) +
#             outer_products_cs.expand(10, 10, k, 10, 10) +
#             outer_products_cs.expand(10, k, 10, 10, 10) +
#             outer_products_cs.expand(k, 10, 10, 10, 10))
sum3 = torch.einsum(('abcde,abcd->e'), combined, outer_products_cs)
print("Method 3:")
print(sum3.shape)
print(sum3)
print()

# method 4. Flatten matrices
sum4 = combined.reshape(-1, k).T @ outer_products_cs.reshape(-1)
print("Method 4:")
print(sum4.shape)
print(sum4)
print()
