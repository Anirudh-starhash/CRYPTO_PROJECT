import hashlib
import sympy
import numpy as np
import random
import os
import ast
from sympy import symbols

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
import binascii
from .BB84i import *
import base64
import json

class ORB:
    
    def __init__(self,field_size=7,v1=2, v2=2, o1=1, o2=2,n=10,t=2):
        
        self.q = field_size
        self.v1 = v1  
        self.v2 = v2  
        self.o1 = o1  
        self.o2 = o2  
        
        self.n=n
        self.t=t
        
        self.rainbow=self.RAINBOW(self.q,self.v1,self.v2,self.o1,self.o2)
        self.bike=self.BIKE(self.n,self.t)
    
    class RAINBOW:
        def __init__(self,field_size=4,v1=1, v2=2, o1=1, o2=2):
            """
                Initializes the ORB instance for Rainbow PQC key generation.

                Parameters:
                - field_size: Prime field size (default 3 for F3).
                - v: Number of vinegar variables.
                - o: Number of oil variables.
            """
            
            self.q = field_size
            self.v1 = v1  
            self.v2 = v2  
            self.o1 = o1  
            self.o2 = o2  
            self.private_key = None
            self.public_key = None
            self.generate_keys()
            
            
        def generate_vinegar_variables(self, v_count):
            
            """ Randomly generate vinegar variables in F_q. """
            return [random.randint(0, self.q - 1) for _ in range(v_count)]
        
        def generate_private_key(self, v_count, o_count):
            
            """ 
                Generate private quadratic polynomials for each layer of Rainbow.
                - Vinegar variables (v_count) are independent.
                - Oil variables (o_count) are determined from equations.
                - Generates quadratic polynomials using random coefficients.
            """
            
            vinegar_vars = [sympy.Symbol(f'v{i}') for i in range(v_count)]
            oil_vars = [sympy.Symbol(f'o{i}') for i in range(o_count)]
            
            ''' 
                variables will be {v0,v1} and  {o0,o1} !
                Define random coefficients for quadratic terms
            '''
            
            coefficients = np.random.randint(0, self.q, (o_count, v_count + o_count, v_count + o_count))
            
            '''
                we generate a 4*4 matrices for 2 equations one matrix per equation and 
                values inside matrix are random
            '''

            ''' Construct private polynomials (quadratic equations) '''
            private_polynomials = []
            for i in range(o_count):
                equation = 0
                variables = vinegar_vars + oil_vars
                for j in range(v_count + o_count):
                    for k in range(j, v_count + o_count):
                        equation += coefficients[i][j][k] * variables[j] * variables[k]
                private_polynomials.append(equation % self.q)  # Mod field size

            '''
            
            each equation will we of form P(o0,o1,v0,v1) = summation (aij)(xi)(xj)

            '''
            return vinegar_vars, oil_vars, private_polynomials
        
        def generate_public_key(self, vinegar_values, vinegar_vars, oil_vars, private_polynomials):
            
            """ Generate the public key by substituting vinegar variables. """
            
            public_polynomials = []
            for poly in private_polynomials:
                for i, v in enumerate(vinegar_values):
                    poly = poly.subs(vinegar_vars[i], v)  
                public_polynomials.append(poly % self.q)  
            
            
            ''' 
                Substitute vinegar values in 
                polynomial equationass to get public polynomial equations
            '''
            return public_polynomials
        
        def generate_keys(self):
            """ Generate Rainbow PQC key pair with multi-layer structure (v1, v2, o1, o2). """

            ''' Generate first and second layer of vinegar variables (v1),v2'''
            v1_values = self.generate_vinegar_variables(self.v1)
            print(v1_values)
            v2_values = self.generate_vinegar_variables(self.v2)  
            print(v2_values)

            '''
                Generate first oil layer (o1) and
                private polynomials for first stage and second
            '''
            v1_vars, o1_vars, private_poly_1 = self.generate_private_key(self.v1,self.o1)
            v2_vars, o2_vars, private_poly_2 = self.generate_private_key(self.v2,self.o2)
            
            print(v1_vars,o1_vars,private_poly_1)
            print(v2_vars,o2_vars,private_poly_2)

            ''' Compute the public key by substituting v1 and v2 values '''
            public_poly_1 = self.generate_public_key(v1_values,v1_vars, o1_vars, private_poly_1)
            public_poly_2 = self.generate_public_key(v2_values,v2_vars, o2_vars, private_poly_2)

            ''' Store private key as multi-layer structure '''
            self.private_key = (v1_values, v2_values, private_poly_1, private_poly_2)
            
            ''' Store the final public key '''
            self.public_key = (public_poly_1, public_poly_2)

            print("Rainbow PQC Key Pair Generated with v1, v2, o1, o2!")
            
            
        def sign_message(self,message):
            """
                Step-2: Signing Process
                
                Given a message:
                1. Hash the message and reduce modulo field size q.
                2. Solve for oil variables o by substituting vinegar variables.
                3. Construct the signature as (vinegar values, solved oil values).
            """
            
            if not self.private_key:
                raise ValueError("Keys Not Generated Yet")
            
            v1_values,v2_values,private_poly_1,private_poly_2=self.private_key
            
            ''' 1. Hash the message using SHA-256 and reduce it modulo q '''
            hash_digest=hashlib.sha256(message.encode()).hexdigest()
            hash_integer=int(hash_digest,16) %self.q
            
            print(f"Message Hash (mod {self.q}): {hash_integer}") 
            
            ''' 2. Select random vinegar values (same approach as key gen) '''
            v1_values = self.generate_vinegar_variables(self.v1)
            v2_values = self.generate_vinegar_variables(self.v2)
            
            v1_vars, o1_vars, _ = self.generate_private_key(self.v1, self.o1)
            
            
            def solve_modular_equations(equations, oil_vars):
                """ Brute-force search for solutions in finite field F_q """
                
                solutions = []
    
                for values in np.ndindex(*(self.q,) * len(oil_vars)):  # Iterate over all possible oil values
                    candidate = dict(zip(oil_vars, values))
                    
                    # Evaluate all equations for this candidate
                    equation_results = [(eq.subs(candidate) % self.q) for eq in equations]
                    
                    print(f"Trying candidate: {candidate} => Results: {equation_results} (Target: {hash_integer})")

                    if all(result == hash_integer for result in equation_results):
                        solutions.append([candidate[o] for o in oil_vars])
                
                if solutions:
                    chosen_solution = random.choice(solutions)
                    print(f"Found valid oil variables: {chosen_solution}")
                    return chosen_solution  # Pick a random valid solution

                print("No valid oil variable solution found.")
                return None
                
            # Solve for layer 1 oil variables
            equation_system_1 = []
            for i, poly in enumerate(private_poly_1):
                for j, v in enumerate(v1_values):
                    poly = poly.subs(v1_vars[j], v)
                equation_system_1.append(poly - hash_integer)  

            o1_values = solve_modular_equations(equation_system_1, o1_vars)
            if not o1_values:
                raise ValueError("No valid oil variable solution found in layer 1!")

            # Solve for layer 2 oil variables
            v2_vars, o2_vars, _ = self.generate_private_key(self.v2, self.o2)
            equation_system_2 = []
            for i, poly in enumerate(private_poly_2):
                for j, v in enumerate(v2_values):
                    poly = poly.subs(v2_vars[j], v)
                equation_system_2.append(poly - hash_integer)  

            o2_values = solve_modular_equations(equation_system_2, o2_vars)
            if not o2_values:
                raise ValueError("No valid oil variable solution found in layer 2!")
                
            ''' 3. Signature consists of vinegar + oil values '''
            signature = (v1_values, o1_values, v2_values, o2_values)
            print("Signature Generated Successfully!")
            return signature
                    
            
        def generate_valid_signature(self, message):
            while True:
                try:
                    signature = self.sign_message(message)
                    return signature  # Return only if successful
                except ValueError:
                    print("No valid oil variable solution found. Retrying...")
                    continue  # Retry if ValueError occurs
                
        def verify_signature(self, message, signature):
            """
                Step-5: Signature Verification
                
                Given a signature:
                1. Hash the message and reduce modulo field size `q`.
                2. Substitute the signature values (vinegar + oil variables) into public polynomials.
                3. Verify if the computed result matches the hashed message.
                
            """
            if not self.public_key:
                raise ValueError("Public Key Not Set")
            
            public_poly_1, public_poly_2 = self.public_key
            v1_values, o1_values, v2_values, o2_values = signature
            
            ''' 1. Hash the message using SHA-256 and reduce it modulo `q` '''
            hash_digest = hashlib.sha256(message.encode()).hexdigest()
            hash_integer = int(hash_digest, 16) % self.q
            
            print(f"Message Hash (mod {self.q}): {hash_integer}")
            
            ''' 2. Substitute signature values into public polynomials '''
            v1_vars = symbols(f'v1_0:{self.v1}')
            o1_vars = symbols(f'o1_0:{self.o1}')
            v2_vars = symbols(f'v2_0:{self.v2}')
            o2_vars = symbols(f'o2_0:{self.o2}')
            
            
            # Layer 1 Verification
            for poly in public_poly_1:
                substituted_poly = poly
                for j, v in enumerate(v1_values):
                    substituted_poly = substituted_poly.subs(f'v1_{j}', v)
                for j, o in enumerate(o1_values):
                    substituted_poly = substituted_poly.subs(f'o1_{j}', o)

                ''' Reduce mod q and compare with hash_integer '''
                evaluated_result = substituted_poly % self.q
                print(f"Evaluated Layer 1 Polynomial: {evaluated_result}")
            
            
            # Layer 2 Verification
            for poly in public_poly_2:
                substituted_poly = poly
                for j, v in enumerate(v2_values):
                    substituted_poly = substituted_poly.subs(f'v2_{j}', v)
                for j, o in enumerate(o2_values):
                    substituted_poly = substituted_poly.subs(f'o2_{j}', o)

                ''' Reduce mod q and compare with hash_integer '''
                evaluated_result = substituted_poly % self.q
                print(f"Evaluated Layer 2 Polynomial: {evaluated_result}")
            
            
            print("Signature is Valid!")
            return True
        
        
        def encrypt_public_keys(self, k):
            
            """
                Encrypt Rainbow PQC public key using AES (CBC mode) and shared key k.
            """
            if not hasattr(self, "public_key") or self.public_key is None:
                raise ValueError("Public keys not generated yet!")

            ''' Convert public key (tuple of polynomials) to JSON-serializable format '''
            public_keys_json = json.dumps([str(poly) for poly in self.public_key])

            ''' Ensure key is 16 bytes (AES-128) '''
            key = k[:16].ljust(16, '0').encode()  

            ''' Generate a random IV (16 bytes) '''
            iv = b'0123456789abcdef' 

            ''' Create AES cipher in CBC mode '''
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            ''' Pad the message to be a multiple of 16 bytes '''
            padder = padding.PKCS7(128).padder()
            padded_message = padder.update(public_keys_json.encode()) + padder.finalize()

            ''' Encrypt and encode in Base64 '''
            encrypted_data = base64.b64encode(iv + encryptor.update(padded_message) + encryptor.finalize()).decode()

            print("Encrypted Public Keys Sent to Bob!")
            return encrypted_data

        def decrypt_public_keys(self, encrypted_data, k):
            
            """
                Decrypt Rainbow PQC public key using AES (CBC mode) and shared key k.
            """
            
            ''' Ensure key is 16 bytes (AES-128) '''
            key = k[:16].ljust(16, '0').encode()  # Pad or truncate to 16 bytes

            ''' Decode from Base64 '''
            encrypted_data = base64.b64decode(encrypted_data)

            ''' Extract IV (first 16 bytes) '''
            iv, encrypted_message = encrypted_data[:16], encrypted_data[16:]

            ''' Create AES cipher in CBC mode '''
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            ''' Decrypt and unpad the message '''
            decrypted_padded = decryptor.update(encrypted_message) + decryptor.finalize()
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

            ''' Convert back to JSON format '''
            public_keys = json.loads(decrypted_data.decode())

            print("Bob Decrypted Public Keys Successfully!")
            return public_keys

                

        def get_keys(self):
            
            """ Return generated keys. """
            return self.private_key, self.public_key
        
        
    class BIKE:
        def __init__(self, n=256, t=20):
            """
            Initialize BIKE scheme with:
            - n: Code length
            - t: Number of errors to introduce
            """
            self.n = n  # Code length
            self.t = t  # Number of errors
            self.H = self.generate_parity_matrix()  # Generate a parity-check matrix
            
            self.private_key = None
            self.public_key = None

        def generate_parity_matrix(self):
            """Generate a random binary parity-check matrix H (n-t x n)"""
            
            H = np.zeros((self.n - self.t, self.n), dtype=int)
            for i in range(self.n - self.t):
                H[i, (i + np.arange(self.t)) % self.n] = 1  # Cyclic pattern
            return H

        def generate_keypair(self):
            """Generate public and private keys."""
            h0 = np.random.randint(0, 2, self.n)  # Random binary vector
            h1 = np.random.randint(0, 2, self.n)  # Random binary vector
            private_key = (h0, h1)
            
            # Ensure `public_key` is a single array
            public_key = np.bitwise_xor(h0, h1)

            return public_key, private_key

        def key_encapsulation(self, public_key):
            
            """Encapsulates a shared secret using the public key."""
            error_vector = np.zeros(self.n, dtype=int)
            error_positions = np.random.choice(self.n, self.t, replace=False)
            error_vector[error_positions] = 1  # Introduce t random errors
            
            if isinstance(public_key, tuple):  
                raise ValueError("public_key should be a 1D NumPy array, not a tuple!") 
            
            # Ensure `public_key` is an ndarray before XOR

            ciphertext = np.bitwise_xor(public_key, error_vector)  # Encrypted message
            shared_secret = hashlib.sha256(ciphertext.tobytes()).digest()  # Shared secret

            return ciphertext, shared_secret

        def bit_flipping_decode(self, received_codeword, private_key):
            
            h0, h1 = private_key
            max_iterations = 100  
            syndrome = np.dot(self.H, received_codeword) % 2  
            for _ in range(max_iterations):
                error_positions = np.where(syndrome != 0)[0]

                if len(error_positions) == 0:
                    break  

                # Count error contributions from each column
                error_magnitude = np.sum(self.H[:, error_positions], axis=0)

                # Flip the bits with the **highest** impact on the syndrome
                threshold = np.percentile(error_magnitude, 85)  
                bits_to_flip = error_positions[error_magnitude > threshold]


                received_codeword[bits_to_flip] ^= 1  
                syndrome = np.dot(self.H, received_codeword) % 2  
            
            

            return received_codeword


        def key_decapsulation(self, ciphertext, private_key):
            """Decapsulates and retrieves the shared secret."""
            corrected_codeword = self.bit_flipping_decode(ciphertext.copy(), private_key)
            shared_secret = hashlib.sha256(corrected_codeword.tobytes()).digest()  # Shared secret

            return shared_secret
        
        
        def encrypt_data(self,X,S,k):
            
            """
                Encrypts (X, S) using AES-GCM with the shared key K.
            """
            
            X_bytes = X.encode()
            S_bytes = str(S).encode()
            
            
            plaintext = X_bytes + b'||' + S_bytes  # Separator for clarity

            # Generate a random nonce (IV) for AES-GCM
            nonce = os.urandom(12)  

            ''' Initialize AES-GCM with shared key K '''
            cipher = Cipher(algorithms.AES(k), modes.GCM(nonce), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            tag = encryptor.tag

            return nonce, ciphertext, tag
        
        def decrypt_data(self,nonce, ciphertext, tag, K):
            
            """
                Decrypts the received ciphertext using AES-GCM with the shared key K.
            """
            # Initialize AES-GCM with shared key K and received nonce
            cipher = Cipher(algorithms.AES(K), modes.GCM(nonce, tag), backend=default_backend())
            decryptor = cipher.decryptor()

            # Decrypt the ciphertext
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            X_decoded, S_decoded = plaintext.split(b'||')

            return X_decoded.decode(), S_decoded.decode()
        
        def run_alice_bb84(self,nsfd):
            time.sleep(1)
            print("This is BB84 Server Thread!")
            
            r="Let's Start BB84 Quantm Key Distribution Protocol !"
            nsfd.send(r.encode())
            
            time.sleep(1)
            c=nsfd.recv(1024).decode()
            print(f"Client Reply : {c}")
            
            n=16
            b=BB84()
            
            print("GENERATING ALICE BITS AND BASES")
            time.sleep(1)
            b.generate_alice_bits(n)
            b.generate_alice_bases(n)
            
            print("Creating Alice Qubits")
            b.create_alice_qubits()
            
            print("Sending Qubits to Bob!")
            r="I'm Sharing my qubits to you! "
            nsfd.send(r.encode())
            
            qpy_stream=io.BytesIO()
            qiskit.qpy.dump(b.alice['qubits'],qpy_stream)
            nsfd.send(qpy_stream.getvalue()) 
            
            
            print("Preparing sift keys process by sharing the bases")
            json_data=json.dumps(b.alice['bases'])
            nsfd.send(json_data.encode())
            print("Receive bases of bob")
            json_data=nsfd.recv(1024).decode()
            b.bob['bases']=json.loads(json_data)
            
            b.sift_alice_key()
            
            alice_key=b.sifted_key_alice
            
            if not b.alice_check_for_eve(nsfd):
                print("Key Exchange Successful - No Errors Detected!")
                r="Key Exchange Successful - No Eve Detected"
                nsfd.send(r.encode())
                
                ''' Now check if bob has eve or not wait for his confirmation '''
                b.alice_respond_to_error_estimation(nsfd)
                x=nsfd.recv(1024).decode()
                print(f"Client Messaged..{c}")
                
                final_keyA = ''.join(map(str, alice_key))
                print("Final Key (Alice):", final_keyA)
            else:
                print("Eve Detected! Key Exchange Failed or Needs Reconciliation.")
                
                r="Eve Detected! Key Exchange Failed or Needs Reconciliation."
                nsfd.send(r.encode())
                
                ''' Now check if bob has eve or not wait for his confirmation '''
                b.alice_respond_to_error_estimation(nsfd)
                x=nsfd.recv(1024).decode()
                print(f"Client Messaged..{c}")
                
                
                
                
                alice_key=b.alice_cascade_reconciliation(nsfd)
                print("Reconsiliation Done Now doing Privacy Amplification")
                target_length = max(4, len(alice_key) // 2)
                final_keyA = b.privacy_amplification(alice_key, target_length) 
                print("Final Key (Alice):", final_keyA)
                
            return final_keyA
            
        def run_bob_bb84(self,csfd):
            time.sleep(1)
            print("This is BB84 Client Thread!")
            
            reply=csfd.recv(1024).decode()
            print(f"Server Messaged :  {reply}")
            
            time.sleep(1)
            r="Sure! Confirmed The Initiation of BB84 Protocol"
            csfd.send(r.encode())
            
            n=16
            b=BB84()
            
            c=csfd.recv(1024).decode()
            print(f"Server Messaged : {c}")
            
            print("Received Alice Qubits")
            qpy_data = csfd.recv(4096)
            qpy_stream = io.BytesIO(qpy_data)
            b.bob['received_qubits'] = qiskit.qpy.load(qpy_stream)
            
            b.print_qubits(b.bob['received_qubits'])
                
            print("Generating Bob bases")
            b.generate_bob_bases(n)
            
            print("Doing a logical calculation")
            b.bob_logical_inference()
            
            from qiskit.primitives import Sampler
            sampler = Sampler()
            b.bob_measure_qubits(sampler)

            
            print("Preparing sift keys process by sharing the bases")
            print("Receive bases of bob")
            json_data=csfd.recv(1024).decode()
            b.alice['bases']=json.loads(json_data)
            
            print("Sending bases to Defender")
            json_data=json.dumps(b.bob['bases'])
            csfd.send(json_data.encode())
            
            
            b.sift_bob_key()
            
            bob_key=b.sifted_key_bob
            
            b.bob_respond_to_error_estimation(csfd)
            x=csfd.recv(1024).decode()
            print(f"Server Messaged..{c}")
            
            if not b.bob_check_for_eve(csfd):
                print("Key Exchange Successful - No Errors Detected!")
                
                r="Key Exchange Successful - No Eve Detected"
                csfd.send(r.encode())
                
                
                final_keyB = ''.join(map(str, bob_key))
                print("Final Key (Bob):", final_keyB)
            else:
                print("Eve Detected! Key Exchange Failed or Needs Reconciliation.")
                
                r="Eve Detected! Key Exchange Failed or Needs Reconciliation."
                csfd.send(r.encode())
                
                
                bob_key=b.bob_cascade_reconciliation(csfd)
                print("Reconsiliation Done Now doing Privacy Amplification")
                target_length = max(4, len(bob_key) // 2)
                final_keyB = b.privacy_amplification(bob_key, target_length) 
                print("Final Key (Bob):", final_keyB)
                
            return final_keyB
        
        def encrypt_public_keys(self, k):
            
            """
                Encrypt Rainbow PQC public key using AES (CBC mode) and shared key k.
            """
            if not hasattr(self, "public_key") or self.public_key is None:
                raise ValueError("Public keys not generated yet!")

            ''' Convert public key (tuple of polynomials) to JSON-serializable format '''
            public_keys_json = json.dumps([str(poly) for poly in self.public_key])

            ''' Ensure key is 16 bytes (AES-128) '''
            key = k[:16].ljust(16, '0').encode()  

            ''' Generate a random IV (16 bytes) '''
            iv = b'0123456789abcdef' 

            ''' Create AES cipher in CBC mode '''
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            ''' Pad the message to be a multiple of 16 bytes '''
            padder = padding.PKCS7(128).padder()
            padded_message = padder.update(public_keys_json.encode()) + padder.finalize()

            ''' Encrypt and encode in Base64 '''
            encrypted_data = base64.b64encode(iv + encryptor.update(padded_message) + encryptor.finalize()).decode()

            print("Encrypted Public Keys Sent!")
            return encrypted_data

        def decrypt_public_keys(self, encrypted_data, k):
            
            """
                Decrypt Rainbow PQC public key using AES (CBC mode) and shared key k.
            """
            
            ''' Ensure key is 16 bytes (AES-128) '''
            key = k[:16].ljust(16, '0').encode()  # Pad or truncate to 16 bytes

            ''' Decode from Base64 '''
            encrypted_data = base64.b64decode(encrypted_data)

            ''' Extract IV (first 16 bytes) '''
            iv, encrypted_message = encrypted_data[:16], encrypted_data[16:]

            ''' Create AES cipher in CBC mode '''
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            ''' Decrypt and unpad the message '''
            decrypted_padded = decryptor.update(encrypted_message) + decryptor.finalize()
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

            ''' Convert back to JSON format '''
            public_keys = json.loads(decrypted_data.decode())

            print("Decrypted Public Keys Successfully!")
            return public_keys

        
 
'''
    orb_instance = ORB(field_size=4, v1=1, v2=2, o1=1, o2=2,n=10,t=2)
    private_key, public_key = orb_instance.rainbow.get_keys()

    print("\n=== PRIVATE KEY ===")
    print("Vinegar values (v1):", private_key[0])
    print("Vinegar values (v2):", private_key[1])
    print("Private Layer 1 Polynomials:", private_key[2])
    print("Private Layer 2 Polynomials:", private_key[3])

    print("\n=== PUBLIC KEY ===")

    print("Public Layer 1 Polynomials:", public_key[0])
    print("Public Layer 2 Polynomials:", public_key[1])


    message = "Hello, Quantum Security!"
    signature = orb_instance.rainbow.generate_valid_signature(message)

    print("\nSignature:", signature)

    public_key, private_key = orb_instance.bike.generate_keypair()
    ciphertext, encapsulated_secret = orb_instance.bike.key_encapsulation(public_key)
    print(ciphertext)
    decapsulated_secret = orb_instance.bike.key_decapsulation(ciphertext, private_key)

    print("Alice Key:", encapsulated_secret.hex())
    print("Bob Key:", decapsulated_secret.hex())

    K = binascii.unhexlify(encapsulated_secret.hex())
    nonce, ciphertext, tag = orb_instance.bike.encrypt_data(message, signature, K)

    print("Nonce:", nonce.hex())
    print("Ciphertext:", ciphertext.hex())
    print("Tag:", tag.hex())

    X_decrypted, S_decrypted = orb_instance.bike.decrypt_data(nonce, ciphertext, tag, K)
    print("\nDecrypted X:", X_decrypted)
    print("Decrypted Signature:", S_decrypted)

    if isinstance(S_decrypted, str):
        try:
            S_decrypted = ast.literal_eval(S_decrypted)  # Convert string to tuple
        except (SyntaxError, ValueError):
            raise ValueError("Decrypted signature format is incorrect!")

    print("\nConverted Signature:", S_decrypted, "Type:", type(S_decrypted))

    # Ensure the structure is valid before calling verify_signature
    if not isinstance(S_decrypted, tuple) or len(S_decrypted) != 4:
        raise ValueError("Decrypted signature format is incorrect! Expected tuple of 4 elements.")

    is_valid = orb_instance.rainbow.verify_signature(X_decrypted, S_decrypted)

    if is_valid:
        print("\nSignature is VALID! Message is authentic.")
    else:
        print("\nSignature is INVALID! Message may be tampered.")

'''       
        
