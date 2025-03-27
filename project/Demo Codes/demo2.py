import random

# Prime number (p) and primitive root (g) - Publicly agreed values
p = 23  
g = 5   

alice_private_key = random.randint(1, p-1) 

''' STEP-1 A = g^a mod p '''
alice_public_key = pow(g, alice_private_key, p)  


''' STEP-2 B=g^b mod p '''
bob_private_key = random.randint(1, p-1) 
bob_public_key = pow(g, bob_private_key, p)  # B = g^b mod p

alice_shared_secret = pow(bob_public_key, alice_private_key, p)   # Sa=B^a modp
bob_shared_secret = pow(alice_public_key, bob_private_key, p)    # Sb=A^b mod p

# Both should have the same shared secret key
print(f"Prime (p): {p}")
print(f"Primitive Root (g): {g}")
print("\n--- Alice ---")
print(f"Alice Private Key: {alice_private_key}")
print(f"Alice Public Key: {alice_public_key}")

print("\n--- Bob ---")
print(f"Bob Private Key: {bob_private_key}")
print(f"Bob Public Key: {bob_public_key}")

print("\n--- Shared Secret Key ---")
print(f"Alice's Computed Shared Key: {alice_shared_secret}")
print(f"Bob's Computed Shared Key: {bob_shared_secret}")

# Verification
if alice_shared_secret == bob_shared_secret:
    print("\nKey Exchange Successful! Both keys match.")
else:
    print("\nKey Exchange Failed! Keys do not match.")
