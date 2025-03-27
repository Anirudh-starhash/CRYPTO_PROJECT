import ast
import copy
import time
from sympy import symbols, sqrt, Poly, div, Mod,  Matrix, GF, mod_inverse, Integer
from sympy.ntheory import isprime, factorint
from sympy.abc import x
from math import gcd, isqrt
from sympy import discrete_log


class DLP_NFS:
    def __init__(self,g=None, h=None, p=None):
        self.g=g
        self.h=h
        self.p=p
        
        self.nfs(self.g,self.h,self.p)

    def BaseMExpansion(self,n,m):
        result = []
        q = n
        while (q != 0):
            a = q % m
            q = q // m
            result.append(a)
        return result[::-1]

    def GetGoodM(self,n,d):

        result = int(n^(1/d))

        while ( (n // (result^d)) != 1):
            result-=1

        return result



    def compute_integer_product_of_pairs(self,polynomial_ring, f, m, N, integer_pairs, vector):
            prod  = 1

            for j in range(len(vector)):
                if (1 == vector[j]):
                    prod  = prod*(integer_pairs[j][0] + m*integer_pairs[j][1]) % N

            return prod

    def find_square(self,f, beta_squared, polynomial_ring, verbose=False):
        """
        Computes the square root of beta_squared.

        Parameters:
        - f: The polynomial defining the number field.
        - beta_squared: A polynomial over ZZ of degree less than d.
        - polynomial_ring: The polynomial ring in which computations are performed.
        - verbose: Whether to print warnings.

        Returns:
        - The polynomial representing the square root, or False if no square root exists.
        """
        a = symbols('a')

        # Evaluate beta_squared at a
        beta2 = beta_squared.subs(x, a)

        # Check if beta2 is a perfect square
        sqrt_beta2 = sqrt(beta2)
        if sqrt_beta2.is_rational:
            return polynomial_ring(sqrt_beta2)

        if verbose:
            print('The supposedly found square is not square!')

        return False

    def compute_numberfield_product_of_pairs(self,polynomial_ring, f, integer_pairs, vector):
        """
        Computes the product of selected linear polynomials modulo f.

        Parameters:
        - polynomial_ring: The polynomial ring over integers.
        - f: The polynomial defining the number field.
        - integer_pairs: A list of (a, b) pairs representing linear polynomials of the form bx + a.
        - vector: A binary vector indicating which pairs to include in the product.

        Returns:
        - The resulting polynomial modulo f.
        """

        product_polynomial = Poly(1, x)  # Start with the identity polynomial

        for j in range(len(vector)):
            if vector[j] == 1:
                a, b = integer_pairs[j]
                linear_poly = Poly([a, b], x)  # bx + a
                product_polynomial *= linear_poly
                product_polynomial = div(product_polynomial, f, domain='ZZ')[1]  # Compute modulo f

        return product_polynomial

    def is_square(self,n):
        """Checks if an integer n is a perfect square."""
        return sqrt(n).is_integer

    def compute_difference_of_squares(self,polynomial_ring, f, m, N, integer_pairs, vector, compute_integer_product_of_pairs, find_square, compute_numberfield_product_of_pairs, verbose=False):
        """
        Computes the difference of squares in a number field.

        Parameters:
        - polynomial_ring: The polynomial ring over integers.
        - f: The polynomial defining the number field.
        - m: An integer used in computation.
        - N: The modulus.
        - integer_pairs: A list of (a, b) pairs.
        - vector: A binary vector indicating selected pairs.
        - compute_integer_product_of_pairs: Function to compute integer product of selected pairs.
        - find_square: Function to find square roots in the number field.
        - compute_numberfield_product_of_pairs: Function to compute product of pairs in the number field.
        - verbose: Whether to print debug messages.

        Returns:
        - A tuple (found_squares, beta_squared, beta, u, v) if successful.
        - Otherwise, (False, 'Filler Word.').
        """

        found_squares = False
        u, v = None, None


        # Compute integer product mod N
        vsquared = Mod(compute_integer_product_of_pairs(polynomial_ring, f, m, N, integer_pairs, vector), N)

        # Check if vsquared is a perfect square
        if self.is_square(vsquared):
            beta_squared = compute_numberfield_product_of_pairs(polynomial_ring, f, integer_pairs, vector)
            beta = find_square(f, beta_squared, polynomial_ring)

            if beta is not False:
                u = Mod(beta.subs(x, m), N)  # Evaluate beta at m modulo N
                v = int(sqrt(vsquared))  # Compute integer square root
                found_squares = True
                return (found_squares, beta_squared, beta, u, v)

            elif verbose:
                print('Failed to find a square root in number field.')
        elif verbose:
            print('Integer was not square.')

        return (found_squares, 'Filler Word.')

    def get_factors(self,numA, factor_base, remaining_primes, algebraic=False):
        """ Factorizes numA over the given factor base. """
        result = []
        len_divisors = len(factor_base)

        for i in range(len_divisors):
            if numA == 1:
                break
            thisFactor = factor_base[i] if not algebraic else factor_base[i][1]
            exponent_tracker = 0

            while numA % thisFactor == 0:
                exponent_tracker += 1
                numA //= thisFactor

            if exponent_tracker > 0:
                result.append((thisFactor, exponent_tracker))

        if numA != 1 and isprime(numA) and not algebraic:
            if numA in remaining_primes:
                remaining_primes[numA] += 1
            else:
                remaining_primes[numA] = 1

        return result, numA == 1

    def runIt(self,n, m, f, d, a_lb, a_ub, b_lb, b_ub, depth, lengthRow,
            rat_factor_base, alg_factor_base, quad_character_base):

        r_mat = []
        tuples = []
        remaining_primes = {}

        for a in range(a_lb, a_ub):
            for b in range(b_lb, b_ub):
                r = a + (b * m)
                r_alg = (pow(-b, d, n) * f.subs(x, -a * mod_inverse(b, n))) % n
                r_alg_2 = abs(r_alg - n)

                if r == 0 or r_alg == 0:
                    continue

                depth_additions = [r_alg_2 + (n * i) for i in range(depth)]

                r_factors, rat_fact_base_match = self.get_factors(r, rat_factor_base, remaining_primes)
                if not rat_fact_base_match:
                    continue

                r_alg_factors, alg_fact_base_match = self.get_factors(r_alg, alg_factor_base, remaining_primes, True)
                if not alg_fact_base_match:
                    for i in range(depth):
                        r_alg_depth_factors, depth_alg_fact_base_match = self.get_factors(depth_additions[i] % n, alg_factor_base, remaining_primes, True)
                        if depth_alg_fact_base_match:
                            r_alg_factors = r_alg_depth_factors
                            break
                    if not depth_alg_fact_base_match:
                        continue

                new_row_r = [0] * lengthRow
                new_row_r[0] = 0 if r >= 0 else 1

                for i, factor in enumerate(rat_factor_base):
                    for base, exp in r_factors:
                        if factor == base:
                            new_row_r[i + 1] = exp % 2
                            break

                used_primes = set()
                for i, (thisR, thisPrime) in enumerate(alg_factor_base):
                    for base, exp in r_alg_factors:
                        if thisPrime == base and (thisPrime not in used_primes or (a % thisPrime) == (-b * thisR) % thisPrime):
                            new_row_r[i + 1 + len(rat_factor_base)] = exp % 2
                            used_primes.add(thisPrime)

                for i, (s, q) in enumerate(quad_character_base):
                    new_row_r[i + 1 + len(rat_factor_base) + len(alg_factor_base)] = 1 if (a + b * s) % q else 0


                r_mat.append(new_row_r)
                tuples.append((a, b))

        if len(r_mat) < lengthRow:
            print(f"Increase sieve size to get {lengthRow - len(r_mat)} more rows.")
            return r_mat, tuples, False, lengthRow - len(r_mat), remaining_primes

        return r_mat, tuples, True, remaining_primes

    def compute_difference_of_squares(self,polynomial_ring, f, m, N, integer_pairs, vector, compute_integer_product_of_pairs, find_square, compute_numberfield_product_of_pairs):
        found_squares = False
        u, v = None, None

        vsquared = compute_integer_product_of_pairs(polynomial_ring, f, m, N, integer_pairs, vector) % N

        if isqrt(vsquared) ** 2 == vsquared:
            beta_squared = compute_numberfield_product_of_pairs(polynomial_ring, f, integer_pairs, vector)
            beta = find_square(f, beta_squared, polynomial_ring)
            x = symbols('x')

            if isinstance(beta, int):
                beta = Integer(beta)  # Convert to a SymPy Integer

            u = beta.subs(x, m) % N
            if beta:
                u = beta.subs(x, m) % N
                v = isqrt(vsquared)
                found_squares = True
                return found_squares, beta_squared, beta, u, v

        return found_squares, "Filler Word."

    def nfs(self,g, h, p):
        """
        Solve the discrete logarithm problem using NFS to break Diffie-Hellman.

        :param p: Prime modulus
        :param g: Generator
        :param h: Public key (g^x mod p)
        :return: x such that g^x ≡ h (mod p)
        """
        # Step 1: Choose number field polynomial
        d = 2  # Degree of polynomial
        f = x**d + 1  # Example choice, can be optimized

        # Step 2: Choose factor bases (these should be selected based on p)
        rat_factor_base = [2, 3, 5, 7]  # Example rational factor base
        alg_factor_base = [(1, 2), (2, 3)]  # Example algebraic factor base
        quad_character_base = [(1, 3), (2, 5)]  # Example quadratic characters

        # Step 3: Define sieving bounds and parameters
        a_lb, a_ub, b_lb, b_ub = -100, 100, 1, 60
        depth, lengthRow = 10, 15

        # Step 4: Run the NFS pipeline
        #result = runIt(p, g, h, f, d, a_lb, a_ub, b_lb, b_ub, depth, lengthRow,
                            #rat_factor_base, alg_factor_base, quad_character_base)
        n, m, f, d = 97, 3, x**2 + 1, 2
        result = self.runIt(n, m, f, d, a_lb, a_ub, b_lb, b_ub, depth, lengthRow,
                rat_factor_base, alg_factor_base, quad_character_base)


        r_mat, tuples, success, remaining_primes = result[0], result[1], result[2], result[3]
        print(f"Total smooth pairs found: {len(tuples)}")

        r_mat_list = [list(row) for row in r_mat]  # Convert each row to a list

        # Create the matrix without specifying GF(2) directly
        M = Matrix(r_mat_list).applyfunc(lambda x: x % 2)  # Ensure all elements are mod 2
        solutions = M.nullspace()
        print(f"Total bases found: {len(solutions)}")
        non_trivial_factorizations = 0
        if success or len(solutions) > 0:
            for solution in solutions:
                result = self.compute_difference_of_squares(None, f, m, n, tuples, solution,
                                                    lambda *args: 1, lambda *args: 1, lambda *args: 1)  # Replace with real functions
                if result[0]:
                    u, v = result[3], result[4]
                    if u != v:
                        g1, g2 = gcd(n, u - v), gcd(n, u + v)
                        if g1 != 1 and g1 != n:
                            print(f"Factors found: {g1} and {g2}")
                            non_trivial_factorizations += 1
        #print(f"Non-trivial factorizations: {non_trivial_factorizations}")
        return discrete_log(p, h, g)

