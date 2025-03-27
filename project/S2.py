# bob 

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
import json
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
# from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from protocols.BB84i import *
from protocols.DHi import *
from protocols.RSAi import *
from protocols.ORBi import *
    

def rsa_protocol(csfd):
    time.sleep(1)
    print("This is RSA Client Thread!")
    
    reply=csfd.recv(1024).decode()
    print(f"Server Messaged :  {reply}")
    
    time.sleep(1)
    r="Sure! Confirmed The Initiation of RSA Encryption-Decryption Technique"
    csfd.send(r.encode())
    
    r=RSA()
    print("Wait for Prime p and q From Server")
    p1,p2=csfd.recv(1024).decode().split(',')
    r.p=int(p1)
    r.q=int(p2)
    
    r1="Got Prime and alpha!"
    csfd.send(r1.encode())
    
    r.generate_n()
    r.generate_phiN()
    r.generate_e()
    r.generate_d()
    r.private_key={'d':r.d,'n':r.n}
    
    r.public_key={'e':r.e,'n':r.n}
    
    print("Wait for public key of Defender")
    c21,c22=csfd.recv(1024).decode().split(',')
    
    r.opublic_key={'e':int(c21),'n':int(c22)}
    
    print("Public key is shared to Defender")
    csfd.send(f"{r.e},{r.n}".encode())
    
    print(f"My public key is ({r.public_key})")
    print(f"Defender's public key is ({r.opublic_key})")
    
    reply=csfd.recv(1024).decode()
    print(f"Server : {reply}")
    r1="Ok Please Send"
    csfd.send(r1.encode())
    
    msg_e,hash_e=csfd.recv(1024).decode().split(',')
    
    msg_e=int(msg_e)
    hash_e=int(hash_e)
 
    
    xre="Received Encrypted Message!"
    csfd.send(xre.encode())
    
    msg_decrypted,Result=r.decrypt(msg_e,hash_e)
    
    if Result:
        print(f"The Encrypted Message and Hash given to us is {msg_e},{hash_e}")
        print(f"The Decrypted Message is {msg_decrypted}")
    else:
        print(f"Integrity was Compromised!")
        
    
    
    return 

def orb_protocol(csfd):
    time.sleep(1)
    print("This is Quantum ORB Client Thread!")
    
    reply=csfd.recv(1024).decode()
    print(f"Server Messaged :  {reply}")
    
    time.sleep(1)
    r="Sure! Confirmed The Initiation of Quantum ORB Encryption-Decryption Technique"
    csfd.send(r.encode())
    
    o = ORB(field_size=7, v1=2, v2=2, o1=1, o2=2,n=10,t=2)
    private_key, public_key = o.rainbow.get_keys()

    print("\n=== PRIVATE KEY ===")
    print("Vinegar values (v1):", private_key[0])
    print("Vinegar values (v2):", private_key[1])
    print("Private Layer 1 Polynomials:", private_key[2])
    print("Private Layer 2 Polynomials:", private_key[3])

    print("\n=== PUBLIC KEY ===")

    print("Public Layer 1 Polynomials:", public_key[0])
    print("Public Layer 2 Polynomials:", public_key[1])
    
    k=o.bike.run_bob_bb84(csfd)
    print(k)
    try:
        ''' Read the first 4 bytes to determine the length of the message. '''
        length_bytes = csfd.recv(4)
        if len(length_bytes) < 4:
            raise ValueError("Incomplete length received.")
        data_length = int.from_bytes(length_bytes, 'big')

        # Receive the encrypted public key data.
        data = b""
        while len(data) < data_length:
            packet = csfd.recv(data_length - len(data))
            if not packet:
                break
            data += packet

        encrypted_public_key = data.decode('utf-8')
        print("Encrypted Public Key received from Alice!")
    except Exception as e:
        print("Error receiving encrypted public key:", e)
    
    
    print("Encrypted Public Layer 1 Polynomials:", encrypted_public_key[0])
    print("Encrypted Public Layer 2 Polynomials:", encrypted_public_key[1])
    decrypted_public_key=o.rainbow.decrypt_public_keys(encrypted_public_key,k)
    print("Decrypted Public Layer 1 Polynomials:", decrypted_public_key[0])
    print("Decrypted Public Layer 2 Polynomials:", decrypted_public_key[1])
    
    
   
    bob_public_key_bike, bob_private_key_bike = o.bike.generate_keypair()
    
    print(f"Bob public key bike is  {bob_public_key_bike}")
    o.bike.private_key=bob_private_key_bike
    o.bike.public_key=bob_public_key_bike
    encrypted_bob_public_key=o.bike.encrypt_public_keys(k)
    print("I'm sending my public BIKE key")
    
    try:
        ''' Convert the encrypted public key to bytes.'''
        data = encrypted_bob_public_key.encode('utf-8')
        data_length = len(data)
        
        ''' Send the length of the message first.'''
        csfd.sendall(data_length.to_bytes(4, 'big'))
        csfd.sendall(data)
        print("Bike Public Key sent to Defender!")
    except Exception as e:
        print("Error sending encrypted public key:", e)
        
    
    try:
       
        length_bytes = csfd.recv(4)
        if len(length_bytes) < 4:
            raise ValueError("Incomplete length received.")
        data_length = int.from_bytes(length_bytes, 'big')
        
        # Receive the data.
        data = b""
        while len(data) < data_length:
            packet = csfd.recv(data_length - len(data))
            if not packet:
                break
            data += packet
        
        ciphertext_str = data.decode('utf-8')
        ciphertext = json.loads(ciphertext_str)
        print("Ciphertext received successfully!")
    except Exception as e:
        print("Error receiving ciphertext:", e)
        
        
    ciphertext=[
                int(ciphertext[i]) 
                for i in range(len(ciphertext))
        ]
    
    ciphertext = np.array(ciphertext)
    
    print(ciphertext)
    decapsulated_secret = o.bike.key_decapsulation(ciphertext, o.bike.private_key)
    bob_shared_secret_key=decapsulated_secret.hex()
    print(f"The bob shared bike secret key is {bob_shared_secret_key}") 
    
    
    ''' STEP-4 CONDIFENTIALITY NOW IN BIKE '''

    """
        Receive the encrypted data sent from Alice.
        First, read the 4-byte length header, then read the actual data,
        and finally decode the JSON to get the nonce, ciphertext, and tag.
    """
    try:
        # Receive the 4-byte length header.
        length_bytes = csfd.recv(4)
        if len(length_bytes) < 4:
            raise ValueError("Incomplete data length received.")
        data_length = int.from_bytes(length_bytes, 'big')
        
        # Now receive the JSON data.
        data_bytes = b""
        while len(data_bytes) < data_length:
            packet = csfd.recv(data_length - len(data_bytes))
            if not packet:
                break
            data_bytes += packet
        
        # Decode the received bytes to a JSON string.
        data_json = data_bytes.decode('utf-8')
        data_dict = json.loads(data_json)
        
        print("Encrypted data received successfully!")
    except Exception as e:
        print("Error receiving encrypted data:", e)
        
        

    nonce = bytes.fromhex(data_dict["nonce"])
    ciphertext = bytes.fromhex(data_dict["ciphertext"])
    tag = bytes.fromhex(data_dict["tag"])
    
    print("Nonce:", nonce.hex())
    print("Ciphertext:", ciphertext.hex())
    print("Tag:", tag.hex())
    
    
    K = binascii.unhexlify(bob_shared_secret_key)
    X_decrypted, S_decrypted = o.bike.decrypt_data(nonce, ciphertext, tag, K)
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

    is_valid = o.rainbow.verify_signature(X_decrypted, S_decrypted)

    if is_valid:
        print("\nSignature is VALID! Message is authentic.")
    else:
        print("\nSignature is INVALID! Message may be tampered.")
        
    print("Successful Encryption-Decryption Done")
    print("\nDecrypted X:", X_decrypted)
    

    
    return 
    
    
def diffie_hellmann(csfd) :
    time.sleep(1)
    print("This is Diffie Hellmann Client Thread!")
    
    reply=csfd.recv(1024).decode()
    print(f"Server Messaged :  {reply}")
    
    time.sleep(1)
    r="Sure! Confirmed The Initiation of Diffie Hellmann Key Exchange"
    csfd.send(r.encode())
    
  
    
    
    d=DH()
    print("Wait for Prime and alpha From Server")
    p1,p2=csfd.recv(1024).decode().split(',')
    d.p=int(p1)
    d.alpha=int(p2)
    
    r="Got Prime and alpha!"
    csfd.send(r.encode())
    
    print(f"Prime Number is  {d.p} and alpha is  {d.alpha}")
    d.generate_private_key()
    d.generate_public_key()
    
    ''' Yb generated now we need to share it '''
    
    print("Wait to Receive Defender's public_key")
    x=csfd.recv(1024).decode()
    d.opublic_key=int(x)
    print("Sending public key to Defender")
    csfd.send(str(d.public_key).encode())
    
    
    print(f"My public Key Yb  is {d.public_key}")
    print(f"Defender's public Key Ya  is {d.opublic_key}")
    
    d.genrate_shared_key()
    print(f"The shared key on my side Ya^Xb mod p is  {d.shared_key}")
    
    return 

def bb84_protocol(csfd):
    time.sleep(1)
    print("This is BB84 Client Thread!")
    
    reply=csfd.recv(1024).decode()
    print(f"Server Messaged :  {reply}")
    
    time.sleep(1)
    r="Sure! Confirmed The Initiation of BB84 Protocol"
    csfd.send(r.encode())
    
    n=10
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
        
    return 

def begin():
    
    csfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    HOST = '127.0.0.1'  
    PORT = 12345
            
    csfd.connect((HOST, PORT))
    message = "Hello from client!"
    csfd.send(message.encode())
    response = csfd.recv(1024).decode()
    print(f"Received from server: {response}")
    
    ''' 
         Here we create 
        two threads one for diffie hellman and one for BB84
    '''
    
    time.sleep(1)
    print("Creating Two threads One for Diffie Hellmann and One for BB84")
    
    ''' Start both cryptographic threads '''
    
    ''' Start Diffie-Hellman first and wait for it to finish '''
    
    time.sleep(1)
    print("Start Diffie Thread")
    diffie_thread = threading.Thread(target=diffie_hellmann,args=(csfd,))
    diffie_thread.start()
    diffie_thread.join()  

    ''' Now start BB84 after Diffie-Hellman is finished '''
    
    time.sleep(1)
    print("Start BB84 Thread")
    bb84_thread = threading.Thread(target=bb84_protocol,args=(csfd,))
    bb84_thread.start()
    bb84_thread.join()
    
    time.sleep(1)
    print("Start RSA Thread")
    rsa_thread = threading.Thread(target=rsa_protocol,args=(csfd,))
    rsa_thread.start()
    rsa_thread.join()
    
    time.sleep(1)
    print("Start Quantum ORB Thread")
    orb_thread = threading.Thread(target=orb_protocol,args=(csfd,))
    orb_thread.start()
    orb_thread.join()
    
    csfd.close()
     

if __name__=='__main__':
    t=threading.Thread(target=begin)
    t.start()
    t.join()
