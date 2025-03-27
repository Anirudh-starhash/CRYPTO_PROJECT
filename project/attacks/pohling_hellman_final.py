from sympy import factorint, mod_inverse

class POHLING_HELLMAN:
    def __init__(self,g=None, h=None, p=None):
        self.g=g
        self.h=h
        self.p=p
        
        self.pohlig_hellman(self.g, self.h, self.p)

    def discrete_log_prime_power(self,g, h, p, q, e):
        """
        Solves the discrete logarithm problem g^x ≡ h (mod p) when the order is q^e.
        Uses the method of successive approximations.
        """
        x = 0
        q_e = q**e
        g_inv = mod_inverse(g, p)  # Compute modular inverse of g

        for k in range(e):
            gk = pow(g, x, p)
            hk = (h * mod_inverse(gk, p)) % p
            d_k = pow(hk, (p - 1) // q**(k+1), p)  # Reduce to smaller exponent
            log_dk = self.discrete_log_bruteforce(pow(g, (p - 1) // q**(k+1), p), d_k, p, q)
            x += log_dk * q**k

        return x % q_e

    def discrete_log_bruteforce(self,g, h, p, q):
        """
        Brute-force method to find x such that g^x ≡ h (mod p) when order is q.
        """
        for x in range(q):
            if pow(g, x, p) == h:
                return x
        return None  # Should never reach this if the problem is well-formed

    def chinese_remainder_theorem(self,congruences, moduli):
        """
        Solves a system of congruences using the Chinese Remainder Theorem.
        """
        N = 1
        for n in moduli:
            N *= n

        result = 0
        for ai, ni in zip(congruences, moduli):
            Ni = N // ni
            Mi = mod_inverse(Ni, ni)
            result += ai * Ni * Mi

        return result % N

    def pohlig_hellman(self,g, h, p):
        """
        Solves g^x ≡ h (mod p) using the Pohlig-Hellman algorithm.
        Works when p-1 is smooth (has small prime factors).
        """
        order = p - 1
        factorization = factorint(order)  # Factorize p-1 into prime powers

        congruences = []
        moduli = []

        for q, e in factorization.items():
            qe = q**e
            gq = pow(g, order // qe, p)
            hq = pow(h, order // qe, p)

            xq = self.discrete_log_prime_power(gq, hq, p, q, e)

            congruences.append(xq)
            moduli.append(qe)

        return self.chinese_remainder_theorem(congruences, moduli)