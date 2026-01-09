import math

import torch
import torch.nn.functional as F

"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""

def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)
        # print("for")
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        X_copy = X.clone()
        A_copy = A.clone()
        Aa = A
        Xa = X
        index = torch.arange(0,L)

        # we have only 2 or 1 nodes left
        if Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        elif Xa.size(2) == 1:
            return

        for k in range(num_steps):
            T = Xa.size(2)
            
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            index = index.view(T//2, 2)

            # print(index[:,0], index[:,1])
            # print("hi", Aa)
            Xa_copy = Aa[:, :, :, 1].mul(Xa[:, :, :, 0])
            Xa[:, :, :, 1].add_(Xa_copy)
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
            index = index[:,1]

           
    

        
        # print("de", X, A)

        # down sweep (first 2 steps unfolded)
        # print("down")
        X[:, :, -1] = 0.
        A[:, :, -1] = 1.
        # print(X[:, :, -1].shape)


        for k in range(num_steps, 0, -1):
            
            index = torch.arange(0,L)
            index = index[2**(k-1)-1:L:2**(k-1)]
            Aa = A[:, :, 2**(k-1)-1:L:2**(k-1)]
            Xa = X[:, :, 2**(k-1)-1:L:2**(k-1)]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            index = index.view(T//2, 2)

            # print(index[:,0], index[:,1])
            # print("be", [index.squeeze()], X.squeeze()[index.squeeze()], A.squeeze()[index.squeeze()])

            Xa_copy = Xa[:, :, :, 0].clone()
            Aa_copy = Aa[:, :, :, 0].clone()
            Xa[:, :, :, 0] = Xa[:, :, :, 1].add_(0) # assign value
            Aa[:, :, :, 0] = Aa[:, :, :, 1].add_(0) # assign value
            Xa[:, :, :, 1].mul_(Aa_copy) # add
            Xa[:, :, :, 1].add_(Xa_copy) 
            Aa[:, :, :, 1].mul_(Aa_copy)

            # print("af", [index.squeeze()], X.squeeze()[index.squeeze()], A.squeeze()[index.squeeze()])


        
        X.mul_(A_copy)
        X.add_(X_copy) # Final add
        # print("de", X)


    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        # print("for")

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        X_copy = X.clone()
        A_copy = A.clone()
        Aa = A
        Xa = X
        index = torch.arange(0,L)

        # we have only 2 or 1 nodes left
        if Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        elif Xa.size(2) == 1:
            return

        for k in range(num_steps):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            index = index.view(T//2, 2)

            # print(index[:,0], index[:,1])
            Xa_copy = Xa[:, :, :, 1].mul(Aa[:, :, :, 0])
            Xa[:, :, :, 0].add_(Xa_copy)
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]
            index = index[:,0]


       
        # down sweep (first 2 steps unfolded)
        # print("down")
        X[:, :, 0] = 0.
        A[:, :, 0] = 1.
        # print(X[:, :, -1].shape)


        for k in range(num_steps, 0, -1):
            
            index = torch.arange(0,L)
            index = index[0:L:2**(k-1)]
            Aa = A[:, :, 0:L:2**(k-1)]
            Xa = X[:, :, 0:L:2**(k-1)]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            index = index.view(T//2, 2)

            Xa_copy = Xa[:, :, :, 1].clone() 
            Aa_copy = Aa[:, :, :, 1].clone()
            Xa[:, :, :, 1] = Xa[:, :, :, 0].add_(0) # assign value
            Aa[:, :, :, 1] = Aa[:, :, :, 0].add_(0) 
            Xa[:, :, :, 0].mul_(Aa_copy) # add
            Xa[:, :, :, 0].add_(Xa_copy) # add
            Aa[:, :, :, 0].mul_(Aa_copy)


        
        
        X.mul_(A_copy)
        X.add_(X_copy) # Final add
        # print("de", X)


    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """
        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in) # (B, npo2(L), D, N)
            X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
        # prepare tensors
        A = A.transpose(2, 1) # (B, D, npo2(L), N)
        X = X.transpose(2, 1) # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """
        # print("start back")
        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)
        # A = A_in.transpose(2, 1)
        # A.requires_grad_(True)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        # print("de", X, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
pscan = PScan.apply
# import numpy as np
# np.random.seed(1)
# torch.manual_seed(42)  # You can use any integer
# x = torch.arange(0,32, requires_grad=True, dtype=torch.double).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
# a = torch.rand(32, requires_grad=True, dtype=torch.double).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
# y = torch.ones_like(x, requires_grad=True, dtype=torch.double) * 100
# output = torch.mean((y-pscan(a,x)))
# # print(x,a, pscan(a,x))
# x.retain_grad()
# a.retain_grad() 
# y.retain_grad()
# output.backward()
# print(x.grad, a.grad)
