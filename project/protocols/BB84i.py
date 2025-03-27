import socket
import threading
import time
import random
from typing import List, Tuple
import numpy as np 
import hashlib
import json
import qiskit.qpy
import io


# Core Qiskit (1.x) imports
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler  # Modern Qiskit uses Primitives for execution

# Backend handling (no need for AerSimulator directly in Qiskit 1.x+ since Sampler handles it)
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as IBMRuntimeSampler

# Optionally if you want local simulation via primitives (Qiskit 1.x way)
from qiskit_ibm_runtime import Options

# To set up visualization for Bloch sphere if needed
from qiskit.visualization.bloch import Bloch
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

class BB84:
    
    def __init__(self):
        
        """
            Constructor to initialize Alice and Bob's data structures.
            Alice has bits, bases, and qubits.
            Bob has received_qubits, bases, and measured_bits.
        """
        
        self.alice = {
            'received_qubits': [],
            'bits': [],        # Alice's random bits (0 or 1)
            'bases': [],       # Alice's random bases ('+' for rectilinear, 'x' for diagonal)
            'qubits': []       # Alice's qubits prepared based on bits and bases
        }

        self.bob = {
            'received_qubits': [],  # Bob's received qubits (could be modified if Eve exists)
            'bases': [],            # Bob's random bases ('+' for rectilinear, 'x' for diagonal)
            'measured_bits': [],# Bob's bits after measurement
            'logical_bits' :[]
        }
        
        self.sifted_key_alice=[]
        self.sifted_key_bob=[]
        
        
        print("BB84 Protocol Instance Created.")

    def __del__(self):
        
        """  Destructor to clean up if necessary. """
        print("BB84 Protocol Instance Destroyed.")
        
    def generate_alice_bits(self, n):\
        
        """
           Generate n random bits (0 or 1) for Alice.
        """
        self.alice['bits'] = np.random.randint(0, 2, size=n).tolist()
        print(f"Alice's bits: {self.alice['bits']}")

    def generate_alice_bases(self, n):
        
        """
           Generate n random bases for Alice.
           '+' for rectilinear basis (Z-basis), 'x' for diagonal basis (X-basis)
        """
        self.alice['bases'] = np.random.choice(['Z', 'X'], size=n).tolist()
        print(f"Alice's bases: {self.alice['bases']}")
        
    def create_alice_qubits(self):
        
        """
          Create qubits based on Alice's bits and bases.
          Z-basis (|0>, |1|) and X-basis (|+>, |->).
        """
        n = len(self.alice['bits'])
        self.alice['qubits'] = []

        for i in range(n):
            qc = QuantumCircuit(1, 1)  # 1 qubit, 1 classical bit

            if self.alice['bases'][i] == 'Z':
                # Z-basis: Standard |0⟩ or |1⟩
                if self.alice['bits'][i] == 1:
                    qc.x(0)  # Prepare |1⟩ if bit is 1 (|0⟩ is default)

            elif self.alice['bases'][i] == 'X':
                qc.h(0)  # Start with |+⟩
                if self.alice['bits'][i] == 1:
                    qc.z(0)  # Apply Z to convert |+⟩ to |−⟩

            self.alice['qubits'].append(qc)

        print(f"Alice prepared {n} qubits (using Z and X bases).")
        
    def print_qubits(self,qubits):
        
        """
            Print the quantum circuits for all Alice's prepared qubits (in pretty format).
        """
        print("\nAlice's Prepared Qubits (Quantum Circuits):\n")
        for i, qc in enumerate(qubits):
            print(f"Qubit {i+1} Circuit:")
            print(qc.draw(output='text'))  # Pretty print using Qiskit's built-in drawer
            print("-" * 40)
            
    
            
            
    def generate_bob_bases(self, n):
        """Generate random bases (Z or X) for Bob."""
        self.bob['bases'] = [random.choice(['Z', 'X']) for _ in range(n)]
        print(f"Bob's bases: {self.bob['bases']}")

   
    def introduce_eve(self):
        """Simulate Eve's presence and potential eavesdropping."""

        r = random.random()  # Between 0 and 1
        eve_present = r < 0.5  # Eve has 50% chance to intercept
        tamper_prob=0.1
        sampler = Sampler()  # Qiskit primitive sampler

        if eve_present:
            print("\nEve is Present! Introducing Eavesdropping...\n")

            for i in range(len(self.alice['qubits'])):
                if random.random() < tamper_prob :
                    eve_basis = random.choice(['Z', 'X'])

                    # Step 1: Eve copies and measures Alice's qubit
                    eve_qc = self.alice['qubits'][i].copy()

                    if eve_basis == 'X':
                        eve_qc.h(0)  # Measure in X-basis

                    eve_qc.measure(0, 0)

                    # Step 2: Get Eve's measurement result using the sampler
                    job = sampler.run([eve_qc])
                    result = job.result()
                    counts = result.quasi_dists[0]
                    eve_bit = max(counts, key=counts.get)

                    # Step 3: Eve resends the qubit with potential disturbance
                    resend_qc = QuantumCircuit(1, 1)

                    if eve_basis == 'X':
                        if eve_bit == 0:
                            resend_qc.h(0)  # Prepare |+⟩
                        else:
                            resend_qc.x(0)
                            resend_qc.h(0)  # Prepare |−⟩
                    else:  # Z basis
                        if eve_bit == 1:
                            resend_qc.x(0)  # Prepare |1⟩
                        # |0⟩ is default

                    # Replace original Alice qubit with disturbed one
                    self.alice['qubits'][i] = resend_qc
                    return True

        else:
            print("\nNo Eve this time. Transmission is clean.\n")
            return False
            
    
    def bob_logical_inference(self):
        
        logical_bits = []
        n = len(self.bob['received_qubits'])
        alice_qubits = self.bob['received_qubits']
        bob_bases = self.bob['bases']

        for i in range(n):
            qc = alice_qubits[i].copy()

            if bob_bases[i] == 'Z':
                # Measuring in Z basis (standard basis)

                if qc.data == []:
                    # No gates applied - state is |0⟩
                    logical_bits.append(0)

                elif qc.data[0].operation.name == 'x':
                    # First gate is X, so state is |1⟩
                    logical_bits.append(1)

                elif qc.data[0].operation.name == 'h':
                    # If first gate is H, Alice prepared in X basis
                    # Check if H was followed by Z (for |−⟩) or nothing (for |+⟩)
                    if len(qc.data) > 1 and qc.data[1].operation.name == 'z':
                        # |−⟩ collapses randomly in Z basis
                        logical_bits.append(random.choice([0, 1]))
                    else:
                        # |+⟩ collapses randomly in Z basis
                        logical_bits.append(random.choice([0, 1]))

            elif bob_bases[i] == 'X':
                # Measuring in X basis (superposition basis)

                if qc.data == []:
                    # No gates applied, |0⟩ measured in X basis (random between |+⟩ or |−⟩)
                    logical_bits.append(random.choice([0, 1]))

                elif qc.data[0].operation.name == 'x':
                    # First gate is X, |1⟩ measured in X basis (collapses randomly)
                    logical_bits.append(random.choice([0, 1]))

                elif qc.data[0].operation.name == 'h':
                    # H gate means Alice prepared in X basis
                    if len(qc.data) > 1 and qc.data[1].operation.name == 'z':
                        # H followed by Z => |−⟩ state
                        logical_bits.append(1)
                    else:
                        # H alone => |+⟩ state
                        logical_bits.append(0)

        print(logical_bits)
        self.bob['logical_bits'] = logical_bits  # Save to Bob's data for later comparison
        print("Bob's Logical Inferred Bits (Direct Analysis):", logical_bits)


    def bob_measure_qubits(self, sampler):
        
        """
            Bob actually measures Alice's qubits using Qiskit's sampler.
            This simulates how a real quantum computer would measure the qubits.
        """
        bob_measured_bits = []
        n = len(self.bob['received_qubits'])
        alice_qubits = self.bob['received_qubits']
        bob_bases = self.bob['bases']

        for i in range(n):
            qc = alice_qubits[i].copy()

            if bob_bases[i] == "X":
                qc.h(0)

            qc.measure(0, 0)  # Measurement collapses the qubit into either 0 or 1.

            job = sampler.run([qc])
            result = job.result()

            counts = result.quasi_dists[0]  # {0: prob, 1: prob}
            measured_bit = max(counts, key=counts.get)  # Most likely outcome
            bob_measured_bits.append(measured_bit)

        self.bob['measured_bits'] = bob_measured_bits
        print("Bob's Measured Bits (via Quantum Simulation):", bob_measured_bits)
        
    # Sift Key Function
    def sift_alice_key(self):
        
        """
            Compare Alice's and Bob's bases to create sifted keys.
        """
        self.sifted_key_alice = []

        for i in range(len(self.alice['bases'])):
            if self.alice['bases'][i] == self.bob['bases'][i]:
                self.sifted_key_alice.append(self.alice['bits'][i])

        print("Alice's Sifted Key: ", self.sifted_key_alice)
        
    def sift_bob_key(self):
        """
            Compare Alice's and Bob's bases to create sifted keys.
        """
        self.sifted_key_bob = []
        for i in range(len(self.bob['bases'])):
            if self.alice['bases'][i] == self.bob['bases'][i]:
                self.sifted_key_bob.append(self.bob['measured_bits'][i])

        print("Bob's Sifted Key: ", self.sifted_key_bob)
        
    def bob_respond_to_error_estimation(self, csfd):
        """
            Bob receives a request from Alice with a list of indices.
            He returns the corresponding bits from his sifted key.
        """
        # Receive selected indices from Alice
        sample_indices = list(map(int, csfd.recv(1024).decode().split(",")))

        # Extract the corresponding bits from Bob's sifted key
        bob_sample_bits = [self.sifted_key_bob[i] for i in sample_indices]

        # Send them back to Alice
        csfd.sendall(",".join(map(str, bob_sample_bits)).encode())

        print("Bob: Sent requested bits for error estimation.")
        
    def alice_respond_to_error_estimation(self, nsfd):
        """
            Bob receives a request from Alice with a list of indices.
            He returns the corresponding bits from his sifted key.
        """
        # Receive selected indices from Alice
        sample_indices = list(map(int, nsfd.recv(1024).decode().split(",")))

        # Extract the corresponding bits from alice's sifted key
        alice_sample_bits = [self.sifted_key_alice[i] for i in sample_indices]

        # Send them back to Bob
        nsfd.sendall(",".join(map(str, alice_sample_bits)).encode())

        print("Alice: Sent requested bits for error estimation.")

    # Estimate Error Function
    def alice_estimate_error(self,nsfd):
        """
            Estimate the error rate by comparing a subset of sifted keys.
            Alice picks random indices and asks Bob for his corresponding bits.
        """
        
        if len(self.sifted_key_alice) == 0:
            return 0

        # Select half of the sifted key for comparison
        sample_size = max(1, len(self.sifted_key_alice) // 2)
        sample_indices = random.sample(range(len(self.sifted_key_alice)), sample_size)

        # Send selected indices to Bob
        nsfd.sendall(",".join(map(str, sample_indices)).encode())

        # Receive Bob's corresponding bits
        bob_sample_bits = list(map(int, nsfd.recv(1024).decode().split(",")))

        # Count mismatches
        mismatches = sum(
            1 for idx, bob_bit in zip(sample_indices, bob_sample_bits) 
            if self.sifted_key_alice[idx] != bob_bit
        )

        return mismatches / sample_size

    def bob_estimate_error(self,csfd):
        """
            Estimate the error rate by comparing a subset of sifted keys.
            Alice picks random indices and asks Bob for his corresponding bits.
        """
        
        if len(self.sifted_key_bob) == 0:
            return 0

        # Select half of the sifted key for comparison
        sample_size = max(1, len(self.sifted_key_bob) // 2)
        sample_indices = random.sample(range(len(self.sifted_key_bob)), sample_size)

        # Send selected indices to Bob
        csfd.sendall(",".join(map(str, sample_indices)).encode())

        # Receive Bob's corresponding bits
        bob_sample_bits = list(map(int, csfd.recv(1024).decode().split(",")))

        # Count mismatches
        mismatches = sum(
            1 for idx, bob_bit in zip(sample_indices, bob_sample_bits) 
            if self.sifted_key_bob[idx] != bob_bit
        )

        return mismatches / sample_size
    
    # Check for Eve Function
    def alice_check_for_eve(self,nsfd):
        """
            Check for Eve by estimating the error rate.
            If the error is too high, Alice suspects an eavesdropper.
        """
        error_rate = self.alice_estimate_error(nsfd)
        print(f"Estimated Error Rate: {error_rate:.2%}")

        if error_rate > 0.11:
            print("High error rate detected — Eve might be present. Triggering Key Reconciliation...")
            return True
        else:
            print("No significant Eve-like interference detected.")
            return False
        
        
    def bob_check_for_eve(self,csfd):
        """
            Check for Eve by estimating the error rate.
            If the error is too high, Alice suspects an eavesdropper.
        """
        error_rate = self.bob_estimate_error(csfd)
        print(f"Estimated Error Rate: {error_rate:.2%}")

        if error_rate > 0.11:
            print("High error rate detected — Eve might be present. Triggering Key Reconciliation...")
            return True
        else:
            print("No significant Eve-like interference detected.")
            return False
        
    '''
        def cascade_reconciliation(self, alice_key, bob_key):
        
            """
                Perform a simplified Cascade-like reconciliation process between Alice's and 
                Bob's sifted keys.This method tries to correct errors introduced by 
                Eve using a two-pass parity check method.
            """
            # Start with Bob's initial key (this might have errors due to Eve's interference)
            corrected_key = bob_key[:]

            # --- Pass 1: Full key parity check ---
            # Check if the total number of 1s in Alice's and Bob's keys match (parity check over the whole key)
            if sum(alice_key) % 2 != sum(bob_key) % 2:
                # If parity mismatch is found, there is at least one error somewhere in the key
                print("Parity mismatch - error detected in the whole block")
                
                # Introduce a random flip in one bit (this is a naive correction step since we don't know the error location)
                flip_index = random.randint(0, len(bob_key) - 1)
                corrected_key[flip_index] ^= 1  # Flip the bit at flip_index

            # --- Pass 2: Parity check on halves of the key ---
            # Now divide the key into two halves and check parity for each half separately
            half = len(alice_key) // 2

            # Check two halves: [0:half] and [half:end]
            for part in [slice(0, half), slice(half, len(alice_key))]:
                alice_part = alice_key[part]
                bob_part = corrected_key[part]

                # Check if the parity (number of 1s) in Alice's and Bob's halves match
                if sum(alice_part) % 2 != sum(bob_part) % 2:
                    # If mismatch, at least one bit is wrong in this half
                    print(f"Parity mismatch detected in sub-block {part}")

                    # Randomly flip one bit within this sub-block to try correcting the error
                    flip_index = random.randint(part.start, part.stop - 1)
                    corrected_key[flip_index] ^= 1  # Flip the bit at flip_index

            # After these two passes, corrected_key is hopefully closer to Alice's key
            return corrected_key
    '''
    
   
    
    def compute_parity(self, key):
        """ Compute the parity of the key (sum of bits modulo 2) """
        return sum(key) % 2
    
    def alice_cascade_reconciliation(self,nsfd):
        """ Defender (Alice) performs the reconciliation process """
        print(f"My Sifted Key (before reconciliation): {self.sifted_key_alice}")

        # Compute full parity and send it to Bob
        alice_parity = self.compute_parity(self.sifted_key_alice)
        nsfd.sendall(str(alice_parity).encode())

        # Receive Bob's full parity
        bob_parity = int(nsfd.recv(1024).decode())
        if alice_parity != bob_parity:
            print("Alice: Parity mismatch in full key detected!")

        # Compute parity for two halves of the key
        half = len(self.sifted_key_alice) // 2
        alice_parity_left = self.compute_parity(self.sifted_key_alice[:half])
        alice_parity_right = self.compute_parity(self.sifted_key_alice[half:])

        # Send Alice's half-key parities
        nsfd.sendall(f"{alice_parity_left},{alice_parity_right}".encode())

        # Receive correction indices from Bob
        correction_data = nsfd.recv(1024).decode()  # Bob sends flipped bit indices
        correction_indices = list(map(int, correction_data.split(",")))

        # Apply the corrections (flip the bits at received indices)
        for index in correction_indices:
            self.sifted_key_alice[index] ^= 1  # Flip the bit

        # Compute final parity after correction
        alice_final_parity = self.compute_parity(self.sifted_key_alice)
        nsfd.sendall(str(alice_final_parity).encode())  # Send final parity to Bob

        # Final verification
        bob_final_parity = int(nsfd.recv(1024).decode())
        if alice_final_parity == bob_final_parity:
            print("Alice: Final reconciliation successful - keys are now synchronized!")
        else:
            print("Alice: Final parity mismatch - reconciliation may need improvement.")

        # Return the corrected key for encryption
        return self.sifted_key_alice
    

    def bob_cascade_reconciliation(self,csfd):
        """ Defender (Bob) performs the reconciliation process """
        print(f"My Sifted Key (before reconciliation): {self.sifted_key_bob}")

        # Compute full parity and send it to Bob
        bob_parity = self.compute_parity(self.sifted_key_bob)
        csfd.sendall(str(bob_parity).encode())

        # Receive Alice's full parity
        alice_parity = int(csfd.recv(1024).decode())
        if alice_parity != bob_parity:
            print("Bob: Parity mismatch in full key detected!")

        # Compute parity for two halves of the key
        half = len(self.sifted_key_bob) // 2
        bob_parity_left = self.compute_parity(self.sifted_key_bob[:half])
        bob_parity_right = self.compute_parity(self.sifted_key_bob[half:])

        # Send Bob's half-key parities
        csfd.sendall(f"{bob_parity_left},{bob_parity_right}".encode())

        # Receive correction indices from Bob
        correction_data = csfd.recv(1024).decode()  # Bob sends flipped bit indices
        correction_indices = list(map(int, correction_data.split(",")))

        # Apply the corrections (flip the bits at received indices)
        for index in correction_indices:
            self.sifted_key_bob[index] ^= 1  # Flip the bit

        # Compute final parity after correction
        bob_final_parity = self.compute_parity(self.sifted_key_bob)
        csfd.sendall(str(bob_final_parity).encode())  # Send final parity to Alice

        # Final verification
        alice_final_parity = int(csfd.recv(1024).decode())
        if alice_final_parity == bob_final_parity:
            print("Alice: Final reconciliation successful - keys are now synchronized!")
        else:
            print("Alice: Final parity mismatch - reconciliation may need improvement.")

        # Return the corrected key for encryption
        return self.sifted_key_bob

            
            
    def privacy_amplification(self, sifted_key, target_length):
        
        """
            Perform Privacy Amplification to reduce Eve's potential knowledge.

            Steps:
            1. Convert sifted key (list of bits) into a binary string.
            2. Hash the binary string using SHA-256.
            3. Convert the hash (hex string) into a binary string.
            4. Truncate the binary string to the desired target length.

            This process shrinks the key while removing any partial information
            Eve may have gained.

            Parameters:
            - sifted_key: The reconciled key (list of 0/1).
            - target_length: Desired final key length after privacy amplification.

            Returns:
            - amplified_key: A string representing the amplified key (binary).
        """
        # Step 1: Convert key from list of bits to binary string
        key_str = ''.join(map(str, sifted_key))
        
        # Step 2: Hash it using SHA-256
        hash_digest = hashlib.sha256(key_str.encode()).hexdigest()
        
        # Step 3: Convert hex digest to binary string
        binary_hash = bin(int(hash_digest, 16))[2:].zfill(256)  # SHA-256 produces 256 bits
        
        # Step 4: Truncate the hash to target length
        amplified_key = binary_hash[:target_length]
        
        return amplified_key

    '''
    
    def key_confirmation(self, alice_final_key, bob_final_key):
        
        """
            Final confirmation step: Both parties hash their final keys and compare.
            This confirms successful key exchange after privacy amplification.
        """
        alice_hash = hashlib.sha256(alice_final_key.encode()).hexdigest()
        bob_hash = hashlib.sha256(bob_final_key.encode()).hexdigest()

        print(f"\nKey Confirmation Step")
        print(f"Alice's Key Hash: {alice_hash}")
        print(f"Bob's Key Hash:   {bob_hash}")

        if alice_hash == bob_hash:
            print("Final keys match after privacy amplification and confirmation!")
            return True
        else:
            print("Final keys do NOT match after privacy amplification — discard and restart!")
            return False
            
    '''
    
    