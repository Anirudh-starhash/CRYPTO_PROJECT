# defender

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
import binascii

# Core Qiskit (1.x) imports
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler  # Modern Qiskit uses Primitives for execution

# Backend handling (no need for AerSimulator directly in Qiskit 1.x+ since Sampler handles it)
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as IBMRuntimeSampler

# Optionally if you want local simulation via primitives (Qiskit 1.x way)
from qiskit_ibm_runtime import Options

# To set up visualization for Bloch sphere if needed
from qiskit.visualization.bloch import Bloch
from qiskit.visualization import plot_bloch_multivector # type: ignore
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from protocols.BB84i import *
from protocols.DHi import *
from protocols.RSAi import *
from protocols.ORBi import *


def rsa_protocol(nsfd):
    time.sleep(1)
    print("This is RSA Server Thread!")
    
    r="Let's Start RSA Encryption-Decryption Technique!"
    nsfd.send(r.encode())
    
    time.sleep(1)
    c=nsfd.recv(1024).decode()
    print(f"Client Reply : {c}")
    
    r=RSA()
    print("Generating Prime p and q")
    r.generate_p()
    r.generate_q()
    
    print(f"Sending p {r.p} and q  {r.q} to bob to make agreement")
    nsfd.send(f"{r.p},{r.q}".encode())
    
    print("Wait for Receival Confirmation From Bob")
    c1=nsfd.recv(1024).decode()
    print(f"Bob's Reply is  {c}")
    
    r.generate_n()
    r.generate_phiN()
    r.generate_e()
    r.generate_d()
    
    r.private_key={'d':r.d,'n':r.n}
    r.public_key={'e':r.e,'n':r.n}

    print("Public key is shared to Bob")
    nsfd.send(f"{r.e},{r.n}".encode())
    print("Wait for public key of Bob")
    c21,c22=nsfd.recv(1024).decode().split(',')
    
    r.opublic_key={'e':int(c21),'n':int(c22)}
    
    print(f"My public key is ({r.public_key['e']},{r.public_key['n']})\n")
    print(f"Bob's public key is ({r.opublic_key['e']},{r.opublic_key['n']})\n")
    
    print("I'm sending a message after performing encrypting do decrypt it")
    msg="I'm sending a message after performing encrypting do decrypt it\n"
    
    nsfd.send(msg.encode())
    c=nsfd.recv(1024).decode()
    print(f"Client :{c}")
    
    msg2="RSA1234"
    
    msg_e,hash_e=r.encrypt(msg2)
    print(f"The Encrypted Message and Encrypted Hash is  {msg_e},{hash_e}\n")
    print("Sent Encrypted!")
    
    nsfd.send(f"{msg_e},{hash_e}".encode())
    re=nsfd.recv(1024).decode()
    print(f"Client : {re}")
    
    
    
    
    
    
    
    return 


def orb_protocol(nsfd):
    time.sleep(1)
    print("This is Quantum ORB  Server Thread!")
    
    r="Let's Start Quantum ORB Encryption-Decryption Technique"
    nsfd.send(r.encode())
    
    time.sleep(1)
    c=nsfd.recv(1024).decode()
    print(f"Client Reply : {c}")
    
    o = ORB(field_size=7, v1=2, v2=2, o1=1, o2=2,n=10,t=2)
    private_key, public_key = o.rainbow.get_keys()


    ''' STEP-1 CREATING PUBLIC AND PRIVATE KETYS FOR RAINBOW '''
    
    print("\n=== PRIVATE KEY ===")
    print("Vinegar values (v1):", private_key[0])
    print("Vinegar values (v2):", private_key[1])
    print("Private Layer 1 Polynomials:", private_key[2])
    print("Private Layer 2 Polynomials:", private_key[3])

    print("\n=== PUBLIC KEY ===")

    print("Public Layer 1 Polynomials:", public_key[0])
    print("Public Layer 2 Polynomials:", public_key[1])
  
    k=o.bike.run_alice_bb84(nsfd)
    print(k)
    o.rainbow.public_key=public_key
    o.rainbow.private_key=private_key
    encrypted_public_key=o.rainbow.encrypt_public_keys(k)
    
    print("Encrypted Public Layer 1 Polynomials:", encrypted_public_key[0])
    print("Encrypted Public Layer 2 Polynomials:", encrypted_public_key[1])
    
    try:
        ''' Convert the encrypted public key to bytes.'''
        data = encrypted_public_key.encode('utf-8')
        data_length = len(data)
        
        ''' Send the length of the message first.'''
        nsfd.sendall(data_length.to_bytes(4, 'big'))
        nsfd.sendall(data)
        print("Encrypted Public Key sent to Bob!")
    except Exception as e:
        print("Error sending encrypted public key:", e)
        
    
    ''' STEP-2  GENERATING A VALID SIGNATURE FOR A MESSAGE USING RAINBOW '''
    message="Server Performing Quantum ORB Protocol" 
    signature=o.rainbow.generate_valid_signature(message)
    
    print("\nSignature:", signature)
    
    print("Wait for bob to Send his bike public key")
    try:
        ''' Read the first 4 bytes to determine the length of the message. '''
        length_bytes = nsfd.recv(4)
        if len(length_bytes) < 4:
            raise ValueError("Incomplete length received.")
        data_length = int.from_bytes(length_bytes, 'big')

        # Receive the encrypted public key data.
        data = b""
        while len(data) < data_length:
            packet = nsfd.recv(data_length - len(data))
            if not packet:
                break
            data += packet

        encrypted_bob_public_key = data.decode('utf-8')
        print("Encrypted Public Key received from Bob!")
    except Exception as e:
        print("Error receiving encrypted public key:", e)
    
   
   
    ''' STEP-3  RECEIVE PUBLIC KEY FROM BOB AND DECRYPT IT TO CREATE SHARED KEY K '''
    decrypted_bob_public_key_bike=o.bike.decrypt_public_keys(encrypted_bob_public_key,k)
    decrypted_bob_public_key_bike=[int(decrypted_bob_public_key_bike[i])
                                   for i in range(len(decrypted_bob_public_key_bike))]
    print(f"Decrypted bob public bike key is {decrypted_bob_public_key_bike}")
    ciphertext, encapsulated_secret = o.bike.key_encapsulation(decrypted_bob_public_key_bike)
    print(f"THE cipher text is {ciphertext}")
    print(f"Sending the cipher text to Defender so that he can generate bike shared key")
    
    
    ciphertext_str = json.dumps(ciphertext.tolist() if hasattr(ciphertext, "tolist") else ciphertext)
    
    ''' Convert the string to bytes. '''
    data = ciphertext_str.encode('utf-8')
    data_length = len(data)
    
    try:
        # Send the length first (4 bytes, big-endian)
        nsfd.sendall(data_length.to_bytes(4, 'big'))
        nsfd.sendall(data)
        print("Ciphertext sent successfully!")
    except Exception as e:
        print("Error sending ciphertext:", e)
    
    alice_shared_secret_key=encapsulated_secret.hex()
    
    ''' THE BIKE SHARED KEY IS HERE!'''
    print(f"The shared secret bike key of alice is {alice_shared_secret_key}")
    
    
    ''' STEP-4 NOW WE HAVE TO ENSURE CONFIDENTIALITY'''
    
    K = binascii.unhexlify(alice_shared_secret_key)
    nonce, ciphertext, tag = o.bike.encrypt_data(message, signature, K)
    
    
    print("Nonce:", nonce.hex())
    print("Ciphertext:", ciphertext.hex())
    print("Tag:", tag.hex())
    
    
    ''' Package the values in a dictionary.'''
    data_dict = {
        "nonce": nonce.hex(),
        "ciphertext": ciphertext.hex(),
        "tag": tag.hex()
    }
    
    ''' Convert the dictionary to a JSON string. '''
    data_json = json.dumps(data_dict)
    data_bytes = data_json.encode('utf-8')
    data_length = len(data_bytes)
    nsfd.sendall(data_length.to_bytes(4, 'big'))
    
    # Then send the actual data.
    nsfd.sendall(data_bytes)
    print("Encrypted data sent to Bob!")
    
    
    
    
  
    return 


def diffie_hellmann(nsfd) :
    time.sleep(1)
    print("This is Diffie Hellmann Server Thread!")
    
    r="Let's Start Diffie Hellmann Key Exchange!"
    nsfd.send(r.encode())
    
    time.sleep(1)
    c=nsfd.recv(1024).decode()
    print(f"Client Reply : {c}")
    
    d=DH()
    print("Generating Prime And primitive root")
    d.generate_p()
    d.generate_alpha()
    
    print(f"Sending p {d.p} and alpha  {d.alpha} to bob to make agreement")
    nsfd.send(f"{d.p},{d.alpha}".encode())
    
    print("Wait for Receival Confirmation From Client")
    c=nsfd.recv(1024).decode()
    print(f"Client's Reply is  {c}")
    
    d.generate_private_key()
    d.generate_public_key()
    
    ''' Ya generated now we need to share it '''
    
    print(f"Ya value is  {d.public_key}")
    
    print("Sending public key to bob")
    nsfd.send(str(d.public_key).encode())
    print("Wait to Receive Bob's public_key")
    x=nsfd.recv(1024).decode()
    
    d.opublic_key=int(x)
    
    print(f"My public Key Ya  is {d.public_key}")
    print(f"Bob's public Key Yb  is {d.opublic_key}")
    
    d.genrate_shared_key()
    print(f"The shared key on my side Yb^Xa mod p is  {d.shared_key}")
    
    return 
    

def bb84_protocol(nsfd):
    time.sleep(1)
    print("This is BB84 Server Thread!")
    
    r="Let's Start BB84 Quantm Key Distribution Protocol !"
    nsfd.send(r.encode())
    
    time.sleep(1)
    c=nsfd.recv(1024).decode()
    print(f"Client Reply : {c}")
    
    n=10
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
         
    return 
    

def begin():
    
    ''' This is the First Thread in the program '''
    print("Hi I am the Defender in the Quantum Cipher Wars Game!")

    sfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    HOST = '127.0.0.1'  
    PORT = 12345        

    sfd.bind((HOST, PORT))
    sfd.listen(5)

    print(f"Server listening on {HOST}:{PORT}...")

    while True:
        
        nsfd, cA = sfd.accept()
        print(f"Connection established with {cA}")
        
        data = nsfd.recv(1024).decode()
        print(f"Received from client: {data}")
        response = "Hello from server!"
        nsfd.send(response.encode())
        
        ''' 
            Here we create 
            two threads one for diffie hellman and one for BB84
        '''
        ''' Start both cryptographic threads '''
        time.sleep(1)
        print("Creating Two threads One for Diffie Hellmann and One for BB84")
        
        ''' Start Diffie-Hellman first and wait for it to finish '''
        
         
        time.sleep(1)
        print("Start Diffie Thread")
        diffie_thread = threading.Thread(target=diffie_hellmann,args=(nsfd,))
        diffie_thread.start()
        diffie_thread.join()  

        ''' Now start BB84 after Diffie-Hellman is finished '''
        
        time.sleep(1)
        print("Start BB84 Thread")
        bb84_thread = threading.Thread(target=bb84_protocol,args=(nsfd,))
        bb84_thread.start()
        bb84_thread.join()
        
        
        time.sleep(1)
        print("Start RSA Thread")
        rsa_thread = threading.Thread(target=rsa_protocol,args=(nsfd,))
        rsa_thread.start()
        rsa_thread.join()
        
        time.sleep(1)
        print("Start Quantum ORB Thread")
        orb_thread = threading.Thread(target=orb_protocol,args=(nsfd,))
        orb_thread.start()
        orb_thread.join()
        
        nsfd.close()
    
    


if __name__=='__main__':
    t=threading.Thread(target=begin)
    t.start()
    t.join()
    

