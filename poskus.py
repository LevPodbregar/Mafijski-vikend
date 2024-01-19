import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3)



# stevilo vmesnih korakov
st_vmesni_koraki = 100
#stevilo korakov... v celoti naredimo st_vmesnih*st_korakov
st_korakov = 100
# stevilo spinov na levi oz. desni strani
N = 100
# stevilo vzbujenih na levi strani
I1 = 3
# stevilo vzbujenih na desni strani
I2 = 70

# generacija seznama spinov na levi in desni strani
spini1 = np.zeros(N)
spini2 = np.zeros(N)
indeksi = np.arange(0,N,1)

#naključno izbiranje vzbujenih spinov na levi strani
izbrani1 = np.random.choice(indeksi, size=I1, replace=False)
spini1[izbrani1] +=1

#naključno izbiranje vzbujenih spinov na desni strani
izbrani2 = np.random.choice(indeksi, size=I2, replace=False)
spini2[izbrani2] +=1

# združimo levi in desni verigi
spini = np.concatenate((spini1,spini2))

# definicija koraka
def korak(spini):
    M = len(spini)
    zamenjava = np.random.binomial(1, 0.5, M)
    for i in range(M-2):
        spin1 = spini[i]
        spin2 = spini[i+1]
        spin3 = spini[i+2]
        if (spin1 == spin3 == 0 and spin2 ==1) or (spin1 == spin3 == 1 and spin2 ==0):
            if zamenjava[i] == 1:
                spini[i+1], spini[i+2] = spin3, spin2
            else:
                spini[i], spini[i+1] = spin2, spin1
        elif ((spin1 == spin2 == 0 and spin3 ==1) or (spin1 == spin2 == 1 and spin3 ==0)) and zamenjava[i] == 1:
            spini[i+1], spini[i+2] = spin3, spin2
            
        elif ((spin3 == spin2 == 0 and spin1 ==1) or (spin3 == spin2 == 1 and spin1 ==0)) and  zamenjava[i] == 1:
            spini[i], spini[i+1] = spin2, spin1
    return spini


# korakanje
l = []
d = []
matrikaspinov = []
for _ in range(st_korakov):
    matrikaspinov.append(list(spini))
    print(np.sum(spini[:N]), np.sum(spini[N:]))
    l.append(np.sum(spini[:N]))
    d.append(np.sum(spini[N:]))
    for _ in range(st_vmesni_koraki):
        spini = korak(spini)

matrikaspinov= np.array(matrikaspinov)


# risanje grafov števila vzbujenih spinov
t = np.arange(0, len(l), 1)    
plt.plot(t,l, label="leva")
plt.plot(t,d, label = "desna")

plt.ylabel("N")
plt.xlabel('t['+str(st_vmesni_koraki)+']')
plt.legend()
plt.show()

#risanje evolucije spinov

plt.imshow(matrikaspinov, extent=[0,2*N,t[-1], 0],  interpolation='nearest', aspect='auto')
plt.colorbar()
plt.ylabel('t['+str(st_vmesni_koraki)+']')
plt.xlabel("vrednosti spinov")
plt.title('Evolucija spinov skozi čas')
plt.show()

