import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import hashlib

def bb84() :
    '''  STEP=1  -> Alice preparing bits and bases  ''' 

    #ip_bits and ip_basis
    n = 10
    alice_bits = np.random.randint(2, size=n)
    alice_bases = np.random.choice(['Z', 'X'], size=n)

    print("Alice's Bits: ", alice_bits)
    print("Alice's Bases: ", alice_bases)

    #ip_qubits

    ''' STEP-2 -> Alice creating qubits from bits and bases '''

    alice_qubits = []

    for i in range(n):
        qc = QuantumCircuit(1, 1)  # 1 qubit, 1 classical bit

        if alice_bases[i] == 'Z':
            # Z-basis: Standard |0⟩ or |1⟩
            if alice_bits[i] == 1:
                qc.x(0)  # Prepare |1⟩ if bit is 1 (|0⟩ is the default)

        elif alice_bases[i] == 'X':
            # X-basis: Superposition
            if alice_bits[i] == 0:
                qc.h(0)  # 0 in X-basis = |+⟩ = H|0⟩
            elif alice_bits[i] == 1:
                qc.x(0)  # Flip to |1⟩ first
                qc.h(0)  # Apply Hadamard to get |−⟩ = H|1⟩

        alice_qubits.append(qc)

    # Optional - print circuits to verify
    for circuit in alice_qubits:
        print(circuit)

    ''' STEP-3 ->   BOB generating random bases and based on alice qubits generating logical bits '''

    # 1. Generate random Bob bases (Z or X) for each qubit
    bob_bases = [random.choice(['Z', 'X']) for _ in range(n)]
    print("Bob's Bases: ", bob_bases)

    '''----EVE SIMULATION---'''

    r=random.random() 
    # Uses Mersenne Twister (MT19937) algorithm for pseudorandom number generation.
    eve_present=r < 0.5
    sampler=Sampler()  #initializes a qiskit sampler

    if eve_present:
        print("\nEve is Present! Introducing Eavesdropping...\n")

        for i in range(n):
            eve_basis = random.choice(['Z', 'X'])

            # Step 1: Measure Alice's qubit in Eve's basis
            eve_qc = alice_qubits[i].copy()

            if eve_basis == 'X':
                eve_qc.h(0)

            eve_qc.measure(0, 0)

            # Step 2: Sample the result (0 or 1) using the sampler
            job = sampler.run([eve_qc])
            result = job.result()
            counts = result.quasi_dists[0]
            eve_bit = max(counts, key=counts.get)

            # Step 3: Eve prepares a new qubit in her observed bit & basis
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

            # Replace Alice's original qubit with Eve's disturbed one
            alice_qubits[i] = resend_qc

            
    logical_bits = []  # To store the logically inferred bits for comparison

    for i in range(n):
        qc = alice_qubits[i].copy()

        ### LOGICAL INFERENCE PART (based on gates and Bob's basis)

        if bob_bases[i] == 'Z':
            # Measuring in Z basis (standard basis)

            if qc.data == []:
                # No gates applied - state is |0⟩
                logical_bits.append(0)

            elif qc.data[0].operation.name == 'x':
                # First gate is X, so state is |1⟩
                logical_bits.append(1)

            elif qc.data[0].operation.name == 'h':
                # If first gate is H, it was meant to be in X basis
                # If there's an X after H, it's |−⟩ (which maps randomly in Z basis)
                if len(qc.data) > 1 and qc.data[1].operation.name == 'x':
                    logical_bits.append(random.choice([0, 1]))  # |−⟩ maps randomly to 0 or 1
                else:
                    logical_bits.append(random.choice([0, 1]))  # |+⟩ maps randomly too

        elif bob_bases[i] == 'X':
            # Measuring in X basis (superposition basis)

            if qc.data == []:
                # No gates applied, |0⟩ measured in X basis (random between |+⟩ or |−⟩)
                logical_bits.append(random.choice([0, 1]))

            elif qc.data[0].operation.name == 'x':
                # First gate is X, |1⟩ measured in X basis
                logical_bits.append(random.choice([0, 1]))  # Random collapse in X basis

            elif qc.data[0].operation.name == 'h':
                # First gate is Z  (prepared in X basis)

                if len(qc.data) > 1 and qc.data[1].operation.name == 'x':
                    # X followed by H => |−⟩ state
                    logical_bits.append(1)
                else:
                    # H alone => |+⟩ state
                    logical_bits.append(0)


    ### PRINT RESULTS
    print("Bob's Logical Inferred Bits (Direct Analysis): ", logical_bits)

    '''STEP-4 -> MEASURED BITS USING A SAMPLER FROM QISKIT PRIMITIVES AND GETTING MEASURED BITS  '''


    bob_measured_bits=[]

    for i in range(n):
        qc=alice_qubits[i].copy()

        if bob_bases[i]=="X":
            qc.h(0)

        qc.measure(0,0)  #Measurement collapses the qubit into either 0 or 1.

        job=sampler.run([qc])
        result=job.result()

        counts=result.quasi_dists[0] #quasi_dists is a dictionary-like object returned by Qiskit’s Sampler.

        '''
        content in quasi_dists
        { 0: 0.88, 1: 0.12 } means there's an 88% chance Bob sees a 0, and 12% chance he sees a 1.
        '''
        meaured_bit=max(counts,key=counts.get)

        bob_measured_bits.append(meaured_bit)


    print("Bob's measured bits are...: ",bob_measured_bits)

    '''STEP-5 KEY SHIFTING AMD LAST SHARED KEY '''
    sifted_key_alice = []
    sifted_key_bob = []

    for i in range(n):
        if alice_bases[i] == bob_bases[i]:
            sifted_key_alice.append(alice_bits[i])
            sifted_key_bob.append(bob_measured_bits[i])

    print("Alice's Sifted Key: ", sifted_key_alice)
    print("Bob's Sifted Key: ", sifted_key_bob)

    # Step 7 - Error Estimation (part of check_for_eve)
    def estimate_error(alice_key, bob_key):
        if len(alice_key) == 0:
            return 0
        sample_size = max(1, len(alice_key) // 2)
        sample_indices = random.sample(range(len(alice_key)), sample_size)
        mismatches = sum(1 for i in sample_indices if alice_key[i] != bob_key[i])
        return mismatches / sample_size



    def check_for_eve(Sa,Sb):
        error_rate = estimate_error(Sa, Sb)
        print(f"Estimated Error Rate: {error_rate:.2%}")
        if error_rate > 0.11:
            print("High error rate detected — Eve might be present. Triggering Key Reconciliation...")
            return True
        else:
            print("No significant Eve-like interference detected.")
            return False
        
    # Step 9 - Cascade-like Key Reconciliation
    def cascade_reconciliation(alice_key, bob_key):
        corrected_key = bob_key[:]
        
        # Pass 1 - Whole key parity check
        if sum(alice_key) % 2 != sum(bob_key) % 2:
            print("Parity mismatch - error detected in whole block")
            flip_index = random.randint(0, len(bob_key) - 1)
            corrected_key[flip_index] ^= 1

        # Pass 2 - Divide into two halves, check parity for each
        half = len(alice_key) // 2
        for part in [slice(0, half), slice(half, len(alice_key))]:
            alice_part = alice_key[part]
            bob_part = corrected_key[part]

            if sum(alice_part) % 2 != sum(bob_part) % 2:
                print(f"Parity mismatch in sub-block {part}")
                flip_index = random.randint(part.start, part.stop - 1)
                corrected_key[flip_index] ^= 1

        return corrected_key



    def privacy_amplification(sifted_key,target_length):
        """
        
            Apply a simple hash-based privacy amplification.
            Convert sifted key to a string, hash it, and truncate.
            STEPS  :
            1) CONVERT KEY FROM BINARY TO STRING
            2) HASH IT USING SHA256
            3) CONVERT HEX_DIGEST BACK TO BINARY
            4) TRUNCATE TO TARGET LENGTH
            
        """
        key_str = ''.join(map(str, sifted_key))
        hash_digest = hashlib.sha256(key_str.encode()).hexdigest()
        binary_hash = bin(int(hash_digest, 16))[2:].zfill(256)  # SHA256 gives 256 bits
        amplified_key = binary_hash[:target_length]
        
        return amplified_key


    def key_confirmation(alice_final_key, bob_final_key):
        alice_hash = hashlib.sha256(alice_final_key.encode()).hexdigest()
        bob_hash = hashlib.sha256(bob_final_key.encode()).hexdigest()
        if alice_hash == bob_hash:
            print("Final keys match after privacy amplification and confirmation!")
            return True
        else:
            print("Final keys do NOT match after privacy amplification — discard and restart!")
            return False



    # Step 6: Error checking (eve may be present)
    if not check_for_eve(sifted_key_alice, sifted_key_bob):
        print("Key Exchange Successful - No Errors Detected!")
        final_keyA = ''.join(map(str, sifted_key_alice)) 
        final_keyB = ''.join(map(str, sifted_key_bob)) 
        print(final_keyA,final_keyB)
    else:
        sifted_key_bob = cascade_reconciliation(sifted_key_alice, sifted_key_bob)
        
        print("Key Exchange Completed - Privacy Amplification Required")
        if sifted_key_alice == sifted_key_bob:
            print("Reconciliation successful - keys match.")
        else:
            print("Warning: Reconciliation failed - mismatch remains.")
        # Example: Suppose we want to shrink key to half length
        target_length = max(4, len(sifted_key_alice) // 2)  # Ensure at least 4 bits remain
        final_keyA = privacy_amplification(sifted_key_alice, target_length)
        final_keyB = privacy_amplification(sifted_key_bob, target_length)

    if key_confirmation(final_keyA, final_keyB):
        print("Secure communication can proceed.")
        print(f"Final Secure Key (after Privacy Amplification if needed) Alice: {final_keyA}")
        print(f"Final Secure Key (after Privacy Amplification if needed) Bob: {final_keyB}")
        return True
    else:
        print("Key mismatch detected, session terminated.")
        return bb84()

if  __name__=="__main__" :
    b=bb84()
    if b==True:
        print("Successfully Implemented BB84 protocol !")
