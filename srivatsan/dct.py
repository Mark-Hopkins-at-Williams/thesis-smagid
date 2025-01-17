import math
import torch

def dct_matrix(n):
    ret = torch.FloatTensor(n, n)
    
    for k in range(n):
        for i in range(n):
            ret[k, i] = math.pi / n * (i + .5) * k
            
    ret = torch.cos(ret)  
    ret[0] /= math.sqrt(2) # X_0 /= sqrt(2)
    return ret * math.sqrt(2. / n)


def idct_matrix(n):
    ret = torch.FloatTensor(n, n)
    
    for k in range(n):
        for i in range(n):
            ret[k, i] = math.pi / n * i * (k + .5)
            
    ret = torch.cos(ret)
    ret[:, 0] /= math.sqrt(2) # x_0 /= sqrt(2)
    return ret * math.sqrt(2. / n)

def apply_matrix(mat, x):
    mat = mat.unsqueeze(0)
    return torch.matmul(torch.transpose(torch.matmul(torch.transpose(x, 1, 2), mat), 1, 2), mat)

def main():
    N = 8

    dct_mat = dct_matrix(N)
    idct_mat = idct_matrix(N)

    t = torch.cos(torch.arange(N, dtype=torch.float32) + torch.arange(N, dtype=torch.float32).unsqueeze(0).t())
    print("T:" + t)

    s = apply_matrix(dct_mat, t)
    print("S:" + s)

    t = apply_matrix(idct_mat, s)
    print("T:" + t)

if __name__ == "__main__":
    main()
