import hashlib
import sympy
import numpy as np
import random

class RSA:
    
    def __init__(self,p=None,q=None):
        
        """
            Constructor to initialize P and Q for RSA
        """
        
        self.p=p
        self.q=q
        self.n=0
        self.phiN=0
        self.d=0
        self.e=0
        
        self.public_key={}
        self.opublic_key={}
        self.private_key={}
        
        print("Diffie Helmann Protocol Instance Created.")
        
        
    def __del__(self):
        
        """  Destructor to clean up if necessary. """
        print("RSA Protocol Instance Destroyed.")
        
    def generate_p(self):
        
        """
            Generate a 10-bit prime number for p.
            Generates a random prime number between 2^9 and 2^10
        """ 
       
        self.p=sympy.randprime(2**255,2**256) 
        
    def generate_q(self):
        
        """
            Generate a 10-bit prime number for q.
            Generates a random prime number between 2^9 and 2^10
        """ 
       
        self.q=sympy.randprime(2**255,2**256) 
        if(self.p==self.q):
            self.generate_q()
            
        
    def generate_n(self):
        ''' n= p*q '''
        self.n=self.p * self.q
        
    def generate_phiN(self):
        
        ''' phiN= (p-1)*(q-1)'''
        
        self.phiN=(self.p-1)*(self.q-1)
        
    
    def generate_e(self):
        
        """
            Generate a public exponent e such that 1 < e < phiN and gcd(e, phiN) = 1
        """
        
        while True:
            self.e=random.randrange(2,self.phiN) # e is in the range (1, phiN)
            if sympy.gcd(self.e,self.phiN)==1:
                break
            
            
    def generate_d(self):
        
        """
            Compute the private exponent d such that d ≡ e^(-1) mod phiN
        """
        
        self.d=pow(self.e,-1,self.phiN)
        
    def compute_hash(self,m):
        '''
            Function for computing hash
            requied for integrity
        '''
        
        hash_obj=hashlib.sha256(m.encode())
        hash_digest=int(hash_obj.hexdigest(),16)
        return hash_digest %self.n
        
        
   
        
    def encrypt(self,m):
        '''
            Encryption function like  
            1) X=M+E(H(M),Pa)
            2) Y= E(M,Pb)
            
        ''' 
        
        print(f"The Message Before Encryption is...{m}")
        
        print(f"Pra is {self.private_key['d']},{self.private_key['n']}\n")
        print(f"Pub is {self.opublic_key['e']},{self.opublic_key['n']}\n")
        print(f"Pua is {self.public_key['e']},{self.public_key['n']}\n")
        
        
        ''' Compute Hash '''
        h=self.compute_hash(m)
        print(f"Computed Hash is ...{h}")
        
        ''' Signature using your private key'''
        s=pow(h,self.private_key['d'],self.private_key['n'])
        print(f"Signature is  {s}")
        
        
        ''' Encrypt message and hash separately '''
        message_int = pad_message(m, self.opublic_key['n']) 
        encrypted_msg=pow(message_int,self.opublic_key['e'],self.opublic_key['n'])
        encrypted_hash=pow(s,self.opublic_key['e'],self.opublic_key['n'])  
       
        print(f"Encrypted Message and Hash are...  {encrypted_msg},{encrypted_hash}")
        return encrypted_msg,encrypted_hash
    
    def decrypt(self,c_m,c_h):
        
        '''
            Decryption function like  
            1) X_msg=D(C_m,Prb)
            2) X_hash=D(C_h,Prb)
            3) 
                h1=compute_hash(X_msg)
                h=D(h,pua)
                then compare h1 and h 
            
            
        ''' 
        
        print(f"The Encrypted Message and hash given {c_m},{c_h}")
        
        print(f"Prb is {self.private_key['d']},{self.private_key['n']}\n")
        print(f"Pua is {self.opublic_key['e']},{self.opublic_key['n']}\n")
        print(f"Pub is {self.public_key['e']},{self.public_key['n']}\n")
        
        X_msg=pow(c_m,self.private_key['d'],self.private_key['n'])
        X_hash=pow(c_h,self.private_key['d'],self.private_key['n'])
        
        print(f"Signature we got  {X_hash}\n")
        print(f"The Message int we got {X_msg}\n")
        
        decrypted_message = unpad_message(X_msg)
        print(f"Decrypted Message: {decrypted_message}\n")
        
        
        '''Verify Integrity'''
        s_decrypted=pow(X_hash,self.opublic_key['e'],self.opublic_key['n'])
        h_compute=self.compute_hash(decrypted_message)
        
        if(h_compute==s_decrypted):
            print("Verification Successfull and Integrity Preserved\n")
            return decrypted_message,True
        
        else:
            return "",False
            
            
    
def pad_message(message, n):
    """
        Convert message to integer and ensure it's smaller than n using padding.
    """
    message_bytes = message.encode('utf-8')
    hash_padding = hashlib.sha256(message_bytes).digest()[:4]  # Short hash padding
    padded_message = message_bytes + hash_padding  # Add padding
    message_int = int.from_bytes(padded_message, 'big')

    if message_int >= n:
        raise ValueError("Message integer is still too large for n! Increase n.")
    
    return message_int   
      
        
        
        
def unpad_message(message_int):
    """
        Convert integer back to original message, removing padding.
    """
    message_bytes = message_int.to_bytes((message_int.bit_length() + 7) // 8, 'big')
    original_message = message_bytes[:-4].decode('utf-8')  # Remove padding
    return original_message
    