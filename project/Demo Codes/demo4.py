import time
import random
import sympy
from math import gcd

class AttackBuffer:
    """Buffer to store attack results"""
    def __init__(self):
        self.attacks = []

    def add_attack(self, attack_name, success, details):
        """Add attack result"""
        self.attacks.append({"Attack": attack_name, "Success": success, "Details": details})

    def show_results(self):
        """Print attack results"""
        print("\nAttack Results:")
        for attack in self.attacks:
            print(f"🔹 {attack['Attack']} → {'✅ Success' if attack['Success'] else '❌ Failed'}")
            print(f"   Details: {attack['Details']}\n")


class NFSAttack:
    """Number Field Sieve Attack to break Diffie-Hellman"""
    def __init__(self, p, alpha, Y_A,Y_B):
        self.p = p
        self.alpha = alpha
        self.Y_A = Y_A
        self.Y_B=Y_B

    def run(self):
        """Simulate NFS attack using sympy's discrete_log"""
        try:
            a = sympy.discrete_log(self.p, self.Y_A, self.alpha)
            b = sympy.discrete_log(self.p, self.Y_B, self.alpha)
            shared_secret = pow(self.Y_B, a, self.p)
            shared_secret1 = pow(self.Y_A, b, self.p)
            return True,f"Recovered DH secret: {shared_secret}, {shared_secret1}"
        except Exception as e:
            return False, str(e)


class PollardsRhoAttack:
    """Pollard's Rho Algorithm to find discrete logarithm"""
    def __init__(self, p, alpha, Y_A):
        self.p = p
        self.alpha = alpha
        self.Y_A = Y_A

    def run(self):
        """Simulate Pollard's Rho Attack"""
        x, a, b = 1, 0, 0
        seen = {}
        while (x, a, b) not in seen:
            seen[(x, a, b)] = len(seen)
            if x % 3 == 0:
                x = (x * self.alpha) % self.p
                a = (a + 1) % (self.p - 1)
            elif x % 3 == 1:
                x = (x * x) % self.p
                a = (2 * a) % (self.p - 1)
                b = (2 * b) % (self.p - 1)
            else:
                x = (x * self.Y_A) % self.p
                b = (b + 1) % (self.p - 1)

        return True, f"Found exponent using Pollard's Rho: {a}"


class MITMAttack:
    """Man-in-the-Middle Attack on Diffie-Hellman"""
    def __init__(self, p, alpha):
        self.p = p
        self.alpha = alpha

    def run(self):
        """Simulate MITM by generating fake keys"""
        eve_a = random.randint(2, self.p - 2)
        eve_b = random.randint(2, self.p - 2)
        fake_Y_A = pow(self.alpha, eve_a, self.p)
        fake_Y_B = pow(self.alpha, eve_b, self.p)
        shared_secret_A = pow(fake_Y_B, eve_a, self.p)
        shared_secret_B = pow(fake_Y_A, eve_b, self.p)

        return True, f"Eve intercepted and established fake keys: {shared_secret_A}, {shared_secret_B}"


class TimingAttack:
    """Timing Attack on modular exponentiation in DH"""
    def __init__(self, p, alpha, Y_A):
        self.p = p
        self.alpha = alpha
        self.Y_A = Y_A

    def run(self):
        """Simulate timing attack by measuring computation time"""
        start = time.time()
        for _ in range(100000):  # Simulate repeated operations
            pow(self.alpha, random.randint(2, self.p - 2), self.p)
        end = time.time()
        elapsed_time = end - start

        return True, f"Observed timing leak: {elapsed_time:.6f} seconds"


class MonitorT:
    """Monitor T applies attacks on captured Diffie-Hellman exchange"""
    def __init__(self, p, alpha, Y_A,Y_B):
        self.p = p
        self.alpha = alpha
        self.Y_A = Y_A
        self.Y_B=Y_B
        self.attack_buffer = AttackBuffer()

    def run_attacks(self):
        """Run all attacks and store results in buffer"""
        attacks = [
            NFSAttack(self.p, self.alpha, self.Y_A,self.Y_B),
            PollardsRhoAttack(self.p, self.alpha, self.Y_A),
            MITMAttack(self.p, self.alpha),
            TimingAttack(self.p, self.alpha, self.Y_A),
        ]

        for attack in attacks:
            success, details = attack.run()
            self.attack_buffer.add_attack(type(attack).__name__, success, details)

        self.attack_buffer.show_results()


# Example Run
if __name__ == "__main__":
    # Simulated Diffie-Hellman exchange
    p = 5# A prime number
    alpha = 3  # Generator
    

    Y_A = 4
    Y_B=350

    print(f"Diffie-Hellman Exchange:")
    print(f"   Public P: {p}, Alpha: {alpha}")
    print(f"   Y_A (Alice): {Y_A} Y_B Bob :{Y_B}")

    monitor = MonitorT(p, alpha, Y_A,Y_B)
    monitor.run_attacks()
 
''' 408 annswer'''