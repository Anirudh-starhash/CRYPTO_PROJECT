from sympy import mod_inverse

class POLLARD_RHO:
    def __init__(self,G, H, P):
        self.G=G
        self.H=H
        self.P=P
        
        self.pollard(self.G,self.H,self.P)
        
        

    def ext_euclid(self,a, b):
        """
        Extended Euclidean Algorithm
        :param a:
        :param b:
        :return:
        """
        if b == 0:
            return a, 1, 0
        else:
            d, xx, yy = self.ext_euclid(b, a % b)
            x = yy
            y = xx - (a / b) * yy
            return d, x, y

    def inverse(self,a, n):
        """
        Inverse of a in mod n
        :param a:
        :param n:
        :return:
        """
        return self.ext_euclid(a, n)[1]

    def xab(self,x, a, b, params):
        """
        Pollard Step
        :param x: Current x value
        :param a: Current a value
        :param b: Current b value
        :param params: Tuple (G, H, P, Q) containing the group parameters
        :return: Updated (x, a, b)
        """
        G, H, P, Q = params  # Unpack the parameters
        sub = x % 3  # Determine the subset

        if sub == 0:
            x = x * G % P
            a = (a + 1) % Q
        elif sub == 1:
            x = x * H % P
            b = (b + 1) % Q
        else:  # sub == 2
            x = x * x % P
            a = a * 2 % Q
            b = b * 2 % Q

        return x, a, b

    def verify(self,g, h, p, x):
        """
        Verifies a given set of g, h, p and x
        :param g: Generator
        :param h:
        :param p: Prime
        :param x: Computed X
        :return:
        """
        return pow(g, x, p) == h

    def pollard(self,G, H, P):
        """
        Pollard's Rho Algorithm for Discrete Logarithm
        :param G: Generator
        :param H: Target value
        :param P: Prime modulus
        :return: The discrete logarithm of H base G
        """
        Q = (P - 1) // 2  # Ensure integer division

        x = G * H % P
        a, b = 1, 1

        X, A, B = x, a, b  # Hare (fast pointer)

        # Using range() for loop
        for _ in range(1, P):
            # Hedgehog (slow pointer)
            x, a, b = self.xab(x, a, b, (G, H, P, Q))

            # Hare (fast pointer, moves twice as fast)
            X, A, B = self.xab(X, A, B, (G, H, P, Q))
            X, A, B = self.xab(X, A, B, (G, H, P, Q))

            if x == X:
                break  # Collision found

        nom = (a - A) % Q  # Numerator
        denom = (B - b) % Q  # Denominator

        print(nom, denom)  # Debug output

        # Compute modular inverse safely
        if denom == 0:
            raise ValueError("Denominator is zero; failure in collision step.")

        res = (mod_inverse(denom, Q) * nom) % Q

        # Verify and return the correct result
        if self.verify(G, H, P, res):
            return res

        return res + Q

