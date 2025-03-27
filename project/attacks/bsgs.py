from math import ceil, sqrt

class BSGS:
    def __init__(self,g,h,p):
        self.g=g
        self.h=h
        self.p=p
        
        self.bsgs(self.g,self.h,self.p)

    def bsgs(self,g, h, p):
        '''
        Solve for x in h = g^x mod p given a prime p.
        If p is not prime, you shouldn't use BSGS anyway.
        '''
        N = ceil(sqrt(p - 1))  # phi(p) is p-1 if p is prime

        # Store hashmap of g^{1...m} (mod p). Baby step.
        tbl = {pow(g, i, p): i for i in range(N)}

        # Precompute via Fermat's Little Theorem
        c = pow(g, N * (p - 2), p)

        # Search for an equivalence in the table. Giant step.
        for j in range(N):
            y = (h * pow(c, j, p)) % p
            if y in tbl:
                return j * N + tbl[y]

        # Solution not found
        return None