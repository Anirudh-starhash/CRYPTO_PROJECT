import numpy as np
import sympy

class DH:
    
    def __init__(self,p=None,alpha=None):
        
        """
            Constructor to initialize P and alpha for Difiie Hellmann Key Exchnage
        """
        
        self.p=p
        self.alpha=alpha
        self.private_key=0
        self.public_key=0
        self.opublic_key=0
        
        print("Diffie Helmann Protocol Instance Created.")
        
        
    def __del__(self):
        
        """  Destructor to clean up if necessary. """
        print("Diffie Helmann Protocol Instance Destroyed.")
        
        
    
    def generate_p(self):
        
        """
            Generate a 10-bit prime number for p.
            Generates a random prime number between 2^9 and 2^10
        """ 
       
        self.p=sympy.randprime(2**9,2**10) 
        
        
    def generate_alpha(self):
        
        """ Generate a primitive root modulo p. """
        
        self.alpha=sympy.primitive_root(self.p)
        
    def generate_private_key(self):
        
        """
            Generate a random private key (1 < X < p)
        """
        self.private_key=np.random.randint(2, self.p - 1)
    
    def generate_public_key(self):
        
        """
            Compute the public key Y = alpha^X mod p
        """
        self.public_key=pow(self.alpha, self.private_key, self.p)
        
    def genrate_shared_key(self):
        
        """
            Compute the shared key K = [Yb^Xa mod p] or [Ya^Xb mod p]
        """
        self.shared_key=pow(self.opublic_key, self.private_key, self.p)
        
        
        
        
        
        
    
    