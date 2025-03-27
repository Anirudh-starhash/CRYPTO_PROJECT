import random
from typing import List, Tuple
import numpy as np 
import hashlib

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
import threading


class BB84:
    
    def __init__(self):
        
        """
            Constructor to initialize Alice and Bob's data structures.
            Alice has bits, bases, and qubits.
            Bob has received_qubits, bases, and measured_bits.
        """
        
        self.alice = {
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
        
    def print_alice_qubits(self):
        
        """
            Print the quantum circuits for all Alice's prepared qubits (in pretty format).
        """
        print("\nAlice's Prepared Qubits (Quantum Circuits):\n")
        for i, qc in enumerate(self.alice['qubits']):
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
        n = len(self.alice['qubits'])
        alice_qubits = self.alice['qubits']
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

        self.bob['logical_bits'] = logical_bits  # Save to Bob's data for later comparison
        print("Bob's Logical Inferred Bits (Direct Analysis):", logical_bits)


    def bob_measure_qubits(self, sampler):
        
        """
            Bob actually measures Alice's qubits using Qiskit's sampler.
            This simulates how a real quantum computer would measure the qubits.
        """
        bob_measured_bits = []
        n = len(self.alice['qubits'])
        alice_qubits = self.alice['qubits']
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
    def sift_key(self):
        
        """
            Compare Alice's and Bob's bases to create sifted keys.
        """
        self.sifted_key_alice = []
        self.sifted_key_bob = []

        for i in range(len(self.alice['bases'])):
            if self.alice['bases'][i] == self.bob['bases'][i]:
                self.sifted_key_alice.append(self.alice['bits'][i])
                self.sifted_key_bob.append(self.bob['measured_bits'][i])

        print("Alice's Sifted Key: ", self.sifted_key_alice)
        print("Bob's Sifted Key: ", self.sifted_key_bob)

    # Estimate Error Function
    def estimate_error(self):
        
        """
            Estimate error rate between Alice's and Bob's sifted keys.
        """
        if len(self.sifted_key_alice) == 0:
            return 0
        
        sample_size = max(1, len(self.sifted_key_alice) // 2)
        sample_indices = random.sample(range(len(self.sifted_key_alice)), sample_size)
        
        mismatches = sum(
            1 for i in sample_indices if self.sifted_key_alice[i] != self.sifted_key_bob[i]
        )
        
        return mismatches / sample_size

    # Check for Eve Function
    def check_for_eve(self):
        
        """
            Check for Eve by estimating error rate between sifted keys.
        """
        error_rate = self.estimate_error()
        print(f"Estimated Error Rate: {error_rate:.2%}")
        
        if error_rate > 0.11:
            print("High error rate detected — Eve might be present. Triggering Key Reconciliation...")
            return True
        else:
            print("No significant Eve-like interference detected.")
            return False
    
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

    
        
 

    def visualize_qubits(self, qubits, title="Qubit Visualization"):
        """
            Visualize the Bloch sphere representation for each qubit.
            Useful to show qubit states before and after Eve's intervention.
        """
        fig, axes = plt.subplots(1, len(qubits), figsize=(len(qubits) * 3, 3), subplot_kw={'projection': '3d'})
        if len(qubits) == 1:
            axes = [axes]  # Handle case where only 1 qubit (single subplot)

        for i, qc in enumerate(qubits):
            try:
                # Extract the statevector for the qubit circuit
                state = Statevector(qc)

                # Convert to Bloch sphere coordinates (x, y, z)
                bloch_vector = self.state_to_bloch_vector(state)

                # Plot Bloch sphere for each qubit
                self.plot_single_bloch_sphere(axes[i], bloch_vector, f"Qubit {i}")

            except Exception as e:
                print(f"Error visualizing qubit {i}: {e}")

        plt.suptitle(title)
        plt.pause(10)
        plt.close()
        
    def state_to_bloch_vector(self,state):
        """
            Converts a statevector into Bloch vector coordinates (x, y, z).
            Assumes single qubit states only.
        """
        if len(state) != 2:
            raise ValueError("Statevector is not for a single qubit")

        a, b = state  # state = [a, b] where |ψ⟩ = a|0⟩ + b|1⟩
        x = 2 * (a.real * b.real + a.imag * b.imag)
        y = 2 * (a.imag * b.real - a.real * b.imag)
        z = (a.real**2 + a.imag**2) - (b.real**2 + b.imag**2)
        return np.array([x, y, z])


    def plot_single_bloch_sphere(self,ax, bloch_vector, label):
        """
            Draws a simple Bloch sphere with a vector for a single qubit.
        """
        # Plot sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.2)

        # Plot vector
        ax.quiver(0, 0, 0, bloch_vector[0], bloch_vector[1], bloch_vector[2], color='red', linewidth=2)

        # Set labels and limits
        ax.set_title(label)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add sphere axis lines
        ax.plot([-1, 1], [0, 0], [0, 0], color="k", linewidth=0.5)
        ax.plot([0, 0], [-1, 1], [0, 0], color="k", linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-1, 1], color="k", linewidth=0.5)
        
    

    
        
    def run_protocol(self) :
        # Step 2: Number of qubits to generate (you are doing 10 for now)
        n = 10

        # Step 3: Generate Alice's random bits and bases
        self.generate_alice_bits(n)
        self.generate_alice_bases(n)

        # Optional — print bits and bases to check
        print("Alice's Bits:  ", self.alice['bits'])
        print("Alice's Bases: ", self.alice['bases'])

        # Step 4: Prepare Alice's qubits based on the bits and bases
        self.create_alice_qubits()

        # Step 5: Print all prepared qubits' circuits
        self.print_alice_qubits()
        
        # Step 6: Generate Bob's bases (now included)
        self.generate_bob_bases(n)
        self.visualize_qubits(self.alice['qubits'], "Alice's Qubits Before Eve")
        # Step 7: Introduce Eve (now included)
        x=self.introduce_eve()
        if x :
            self.visualize_qubits(self.alice['qubits'], "Alice's Qubits After Eve")

            # Optional: Print final Alice qubits again to see disturbance (if any)
            print("\nAlice's Qubits After Eve's Possible Interference:")
            self.print_alice_qubits()

        # Step 8: Logical inference (Bob analyzes qubits logically based on gates & bases)
        self.bob_logical_inference()

        # Step 9: Real quantum measurement (Bob measures qubits using Qiskit Sampler)
        from qiskit.primitives import Sampler
        sampler = Sampler()
        self.bob_measure_qubits(sampler)
        
        # After measurement - Step 10: Sift Key
        self.sift_key()
        sifted_key_alice = self.sifted_key_alice
        sifted_key_bob = self.sifted_key_bob
        
        # Step 11: Check for Eve
        if not self.check_for_eve():
            print("Key Exchange Successful - No Errors Detected!")
            final_keyA = ''.join(map(str, sifted_key_alice))
            final_keyB = ''.join(map(str, sifted_key_bob))
            print("Final Key (Alice):", final_keyA)
            print("Final Key (Bob):  ", final_keyB)
        else:
            print("Eve Detected! Key Exchange Failed or Needs Reconciliation.")
            # Attempt Cascade-like Reconciliation since Eve may have introduced errors
            sifted_key_bob = self.cascade_reconciliation(sifted_key_alice, sifted_key_bob)

            print("Privacy Amplification Required")
            final_keyA = ''.join(map(str, sifted_key_alice))
            final_keyB = ''.join(map(str, sifted_key_bob))

            if sifted_key_alice == sifted_key_bob:
                print("Reconciliation successful - keys match.")
                print("Final Key (Alice):", final_keyA)
                print("Final Key (Bob):  ", final_keyB)
                
                # Privacy Amplification - Reduce final key length to enhance security
                target_length = max(4, len(sifted_key_alice) // 2)  # At least 4 bits for safety
                target_length2 = max(4, len(sifted_key_bob) // 2)  # At least 4 bits for safety

                final_keyA = self.privacy_amplification(sifted_key_alice, target_length)
                final_keyB = self.privacy_amplification(sifted_key_bob, target_length2)

                print("Privacy Amplified Final Key (Alice):", final_keyA)
                print("Privacy Amplified Final Key (Bob):  ", final_keyB)
            else:
                print("Warning: Reconciliation failed - mismatch remains.")
                print("Final Key (Alice):", final_keyA)
                print("Final Key (Bob):  ", final_keyB)
                
        if self.key_confirmation(final_keyA, final_keyB):
            print("Secure communication can proceed.")
            print(f"Final Secure Key (after Privacy Amplification if needed) Alice: {final_keyA}")
            print(f"Final Secure Key (after Privacy Amplification if needed) Bob: {final_keyB}")
            return True
        else:
            print("Key mismatch detected, session terminated.")
            return self.run_protocol()


                
if  __name__== "__main__" :
    
    # Step 1: Create BB84 object
    bb84 = BB84()
    x=bb84.run_protocol()
    if x:
        print("Successfully Implemented BB84 PROTOCOL!")
    del bb84


        
    
