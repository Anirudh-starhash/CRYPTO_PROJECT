# -*- coding: utf-8 -*-
"""grovers.py
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit,ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit_ibm_provider import IBMProvider
from qiskit import transpile
from qiskit.visualization import plot_histogram
import math
from qiskit_aer import Aer
from qiskit.primitives import BackendEstimator as execute
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit.circuit.library import HGate, XGate, MCXGate

class GROVER:
    def __init__(self,n=None, marked_states=None, iterations=None):
        self.n=n
        self.marked_states=marked_states
        self.iterations=iterations
        
        self.grovers(self.n,self.marked_states,self.iterations)
        

    def initialize_s(self,qc, qubits):
        """Apply a H-gate to 'qubits' in qc"""
        for q in qubits:
            qc.h(q)
        return qc

    def diffuser(self,nqubits):
        qc = QuantumCircuit(nqubits)
        # Apply transformation |s> -> |00..0> (H-gates)
        for qubit in range(nqubits):
            qc.h(qubit)
        # Apply transformation |00..0> -> |11..1> (X-gates)
        for qubit in range(nqubits):
            qc.x(qubit)
        # Do multi-controlled-Z gate
        qc.h(nqubits-1)
        qc.append(MCXGate(nqubits - 1), list(range(nqubits)))  # multi-controlled-toffoli
        #qc.append(MCXGate(nqubits - 1, mode="noancilla"), list(range(nqubits)))
        #qc.mct(list(range(nqubits - 1)), nqubits - 1)
        qc.h(nqubits-1)
        # Apply transformation |11..1> -> |00..0>
        for qubit in range(nqubits):
            qc.x(qubit)
        # Apply transformation |00..0> -> |s>
        for qubit in range(nqubits):
            qc.h(qubit)
        U_s = qc.to_gate()
        U_s.name = "diffuser"
        return U_s

    def create_grover_oracle(self,n, marked_elements):
        """
        Creates an oracle for Grover's algorithm that flips the phase of multiple marked elements using an ancilla qubit.

        Parameters:
        n (int): Number of qubits
        marked_elements (list of str): A list of binary strings representing the marked elements

        Returns:
        QuantumCircuit: The oracle circuit
        """
        qr = QuantumRegister(n, "q")
        ancilla = AncillaRegister(1, "ancilla")  # Single ancilla qubit
        oracle = QuantumCircuit(qr, ancilla)

        for marked_element in marked_elements:
            # Apply X gates to match the marked element
            for i, bit in enumerate(marked_element):
                if bit == '0':
                    oracle.x(qr[i])

            # Use an ancilla to mark this state
            oracle.mcx(list(range(n)), ancilla[0])  # Control on all qubits, flips ancilla

            # Apply phase flip using Z on ancilla
            oracle.z(ancilla[0])

            # Uncompute: Revert the X gates
            oracle.mcx(list(range(n)), ancilla[0])  # Undo marking
            for i, bit in enumerate(marked_element):
                if bit == '0':
                    oracle.x(qr[i])

        return oracle

    def generate_sequence(self,n):
        return list(range(n))

    def simulate(self,grover_circuit):
        qasm_sim = Aer.get_backend('qasm_simulator')
        transpiled_grover_circuit = transpile(grover_circuit, qasm_sim)
        results = qasm_sim.run(transpiled_grover_circuit).result()
        counts = results.get_counts()
        #plot_histogram(counts)
        return counts

    def grovers(self,n, marked_states, iterations=None):
        # Add one ancilla qubit to the circuit
        qr = QuantumRegister(n, "q")
        ancilla = AncillaRegister(1, "ancilla")
        grover_circuit = QuantumCircuit(qr, ancilla)

        values = self.generate_sequence(n)
        grover_circuit = self.initialize_s(grover_circuit, values)

        # Define the oracle and diffuser once
        oracle = self.create_grover_oracle(n, marked_states)  # Oracle includes ancilla
        diffuser_op = self.diffuser(n)

        # Determine number of iterations based on marked states
        M = len(marked_states)  # Number of marked states
        N = 2**n  # Total states

        if iterations is None:
            iterations = int(np.floor((np.pi / 4) * np.sqrt(N / M))) if M > 0 else 1

        # Apply Grover iterations
        for _ in range(iterations):
            grover_circuit.compose(oracle, inplace=True)
            grover_circuit.append(diffuser_op, range(n))  # Apply diffuser

        grover_circuit.measure_all()
        data= self.simulate(grover_circuit)
        updated_data = {key[1:] if key.startswith('0') else key: value for key, value in data.items()}
        reversed_data = {key[::-1]: value for key, value in updated_data.items()}
        return reversed_data

#grovers(3,["001"])

