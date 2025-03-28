# attacker
from attacks import dlp_nfs,bsgs,dlp_shor,pohling_hellman_final,pollard_rho,shor_algo,grovers
import time
class Monitor:
    
    def __init__(self):
        
        ''' ATTACK BUFFER SIMULATOR '''
        
        self.AttackBuffer={
            "DLP_NFS" : dlp_nfs.DLP_NFS(),
            "BSGS" : bsgs.BSGS(),
            "DLP_SHOR" :  dlp_shor.DLP_SHOR(),
            "POHLIG_HELLMAN": pohling_hellman_final.POHLING_HELLMAN(),
            "POLLARD_RHO": pollard_rho.POLLARD_RHO(),
            "SHOR": shor_algo.SHOR(),
            "GROVER":  grovers.GROVER()
        }
        
    def __del__(self):
        print("MONITOR INSTANCE DESRTROYED")
        
        
    def caller(self,diffie_hellmann_pairs):
        for x in self.AttackBuffer:
            if x in ['DLP_NFS','BSGS',"POHLIG_HELLMAN","POLLARD_RHO"]:
                time.sleep(1)
                print(f"For the Five Pairs Let's simulate {x}Attack")
                
                i=0
                for y in diffie_hellmann_pairs:
                    g=diffie_hellmann_pairs[y]['alpha']
                    h=diffie_hellmann_pairs[y]['Ya']
                    p=diffie_hellmann_pairs[y]['P']
                    Yb=diffie_hellmann_pairs[y]['Yb']
                    
                    instance=self.AttackBuffer[x]
                    Xa=0
                    R=True
                    if x=="DLP_NFS":
                        R,Xa=instance.nfs(g,h,p)
                       
                        
                    elif x=="POLLARD_RHO":
                        Xa,R=instance.pollard(g,h,p)
                        
                    elif x=="POHLIG_HELLMAN":
                        Xa,R=instance.pohlig_hellman(g,h,p)
                        
                    else:
                        Xa,R=instance.bsgs(g,h,p)
                    
                    ''' Now use the Xa with Yb to get shared key'''
                    
                    if R==False:
                        print("There is no log so unabe to break for this case")
                    else:
                        K=pow(Yb,Xa,p)
                        
                        if K==diffie_hellmann_pairs[y]['K']:
                            print(f"Iteration - {i} Diffie Hellmann Compromised using {x}!")
                            print(f"The shared key we got with attack is {diffie_hellmann_pairs[y]['K']}")
                    i+=1
            else:
                pass
        
        
        
diffie_hellmann_pairs={
    
    '0': {
        'Ya':270,
        'Yb':203,
        'alpha':3,
        'P':253,
        'k':141
    },
    '1': {
        'Ya':106,
        'Yb':200,
        'alpha':3,
        'P':631,
        'K':25
    },
    '2': {
        'Ya':329,
        'Yb':763,
        'alpha':2,
        'P':853,
        'K':419
    },
    '3': {
        'Ya':498,
        'Yb':131,
        'alpha':2,
        'P':797,
        'K':716
    },
    '4': {
        'Ya':288,
        'Yb':449,
        'alpha':2,
        'P':523,
        'K':37
    },
    '5':{
        'alpha':2,
        'Ya':11,
        'P':59,
        'Yb':56,
        'K':8
    }
}

m=Monitor()
m.caller(diffie_hellmann_pairs)


