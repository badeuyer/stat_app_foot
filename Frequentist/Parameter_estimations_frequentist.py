import pandas as pd
import numpy as np
import copy
import math
import time
import matplotlib.pyplot as plt


# Importer les données
data = pd.read_csv("F1.csv")
data_2=data.copy()

start_time = time.time()

#Initialisation
liste_equ = ['Lille', 'Bastia', 'Marseille', 'Montpellier', 'Nantes', 'Nice', 'Troyes', 'Bordeaux', 'Lyon', 'Toulouse',
             'Monaco', 'Angers', 'Caen', 'Guingamp', 'Rennes', 'St Etienne', 'Lorient', 'Paris SG', 'Reims',
             'Ajaccio GFCO']
liste_bm = [0] * len(liste_equ)
liste_be = [0] * len(liste_equ)

#DataFrame but marqués à domicile et à l'extérieur
data_bm=pd.DataFrame(index=liste_equ, columns=["bm"])
data_be=pd.DataFrame(index=liste_equ,columns=["be"])
#DataFrame but encaissés à domicile et à l'extérieur
data_benc_dm=pd.DataFrame(index=liste_equ, columns=["benc_dm"])
data_benc_ext=pd.DataFrame(index=liste_equ,columns=["benc_ext"])

# le nombre de but marque et encaisse pour chaque équipe à domicile :
for i in range(len(liste_equ)):
    liste_bm[i] = data[data.HomeTeam == liste_equ[i]]['FTHG'].sum()

#On remplit les Dataframes de buts marqués et encaissés pour chaque équipe à domicile
for i in liste_equ:
    data_bm.loc[i,"bm"]=data[data.HomeTeam == i]['FTHG'].sum()
    data_benc_dm.loc[i,"benc_dm"]=data[data.HomeTeam == i]['FTAG'].sum()

# le nombre de buts marqués pour chaque équipe à l'extérieur :
for i in range(len(liste_equ)):
    liste_be[i] = data[data.AwayTeam == liste_equ[i]]['FTAG'].sum()

#On remplit les Dataframes de buts marques et encaissés pour chaque équipe à l'extérieur

for i in liste_equ:
    data_be.loc[i, "be"]=data[data.AwayTeam == i]['FTAG'].sum()
    data_benc_ext.loc[i,"benc_ext"]=data[data.AwayTeam == i]['FTHG'].sum()

# On créée une dataframe pour nos parametres Alpha Beta et A
alpha=pd.DataFrame(index=liste_equ,columns=["alpha_i_1","alpha_i","alpha_i+1"])
beta = pd.DataFrame(index=liste_equ,columns=["beta_i_1","beta_i","beta_i+1"])
A = pd.DataFrame(index=[0],columns=["A_i_1","A_i","A_i+1"])

# Parametre epsilon
epsilon = 0.001

# Initalisation de alpha, beta et A
alpha_0=3
beta_0=0.2
A_0 = 20


#Algortihme de minimisation alternée
# Contrainte : moyenne des alpha vaut n : le nombre d'équipes

#Initialisation des alpha, beta et A
for i in liste_equ:
    alpha.loc[i,"alpha_i_1"]=alpha_0
    beta.loc[i,"beta_i_1"]=beta_0
    A.loc[0,"A_i_1"]=A_0

mean1 = alpha["alpha_i_1"].mean()
alpha["alpha_i_1"]=alpha["alpha_i_1"]*1.0/mean1

#Calcul du premier alpha

for i in liste_equ:
    alpha.loc[i,"alpha_i"]=(data_bm.loc[i,"bm"]+data_be.loc[i,"be"])/(A.loc[0,"A_i_1"]*beta[beta.index!=i]["beta_i_1"].sum()+beta[beta.index!=i]["beta_i_1"].sum())

# Contrainte sur alpha

mean1 = alpha["alpha_i"].mean()
alpha["alpha_i"]=alpha["alpha_i"]*1.0/mean1

#Calcul du premier beta

for i in liste_equ:
    beta.loc[i,"beta_i"]=(data_benc_dm.loc[i,"benc_dm"]+data_benc_ext.loc[i,"benc_ext"])/(A.loc[0,"A_i_1"]*alpha[alpha.index!=i]["alpha_i_1"].sum()+alpha[alpha.index!=i]["alpha_i_1"].sum())

#Calcul du premier A

data=data.dropna()
temp=0
for i in data.index:
    temp=temp+alpha.loc[data.loc[i,"HomeTeam"],"alpha_i_1"]*beta.loc[data.loc[i,"AwayTeam"],"beta_i_1"]
A.loc[0,"A_i"]=(data_bm["bm"].sum())/temp
iter =0

#Calcul du premier élément de la log vraisemblance

temp_2=0
log_vrais=[]

for i in liste_equ:
    for j in liste_equ:
        if i != j:
            x = float(data_2[(data_2.HomeTeam == i) & (data_2.AwayTeam == j)]['FTHG'])
            y = float(data_2[(data_2.HomeTeam == i) & (data_2.AwayTeam == j)]['FTAG'])
            temp_2 += x * math.log(A.loc[0, "A_i"] * alpha.loc[i, "alpha_i"] * beta.loc[j, "beta_i"]) - A.loc[
                                                                                                                  0, "A_i"] * \
                                                                                                              alpha.loc[
                                                                                                                  i, "alpha_i"] * \
                                                                                                              beta.loc[
                                                                                                                  j, "beta_i"] - \
                      alpha.loc[j, "alpha_i"] * beta.loc[i, "beta_i"] + y * alpha.loc[j, "alpha_i"] * beta.loc[
                i, "beta_i"] - math.log(math.factorial(x)) - math.log(math.factorial(y))
log_vrais.append(temp_2)


list_iter=[iter]
list_mean_beta=[]
list_mean_beta.append(beta["beta_i"].mean())
list_A=[]
list_A.append(A["A_i"])

# Algorithme de minimisation alternée :

while abs(beta["beta_i"]-beta["beta_i_1"]).mean()>epsilon:
    temp=0
    temp_2=0
    iter +=1
    list_iter.append(iter)
    list_mean_beta.append(beta["beta_i"].mean())
    list_A.append(A["A_i"])
    for i in liste_equ:
        alpha.loc[i, "alpha_i+1"] = (data_bm.loc[i, "bm"] + data_be.loc[i, "be"]) / (A.loc[0, "A_i"] * beta[beta.index != i]["beta_i"].sum() + beta[beta.index != i]["beta_i"].sum())
    mean1 = alpha["alpha_i+1"].mean()
    alpha["alpha_i+1"]=alpha["alpha_i+1"]*1.0/mean1
    for i in liste_equ:
        beta.loc[i, "beta_i+1"] = (data_benc_dm.loc[i,"benc_dm"]+data_benc_ext.loc[i,"benc_ext"]) / (A.loc[0, "A_i"] * alpha[alpha.index != i]["alpha_i"].sum() + alpha[alpha.index != i]["alpha_i"].sum())
    #mean = beta["beta_i+1"].mean()
    #beta["beta_i+1"]=beta["beta_i+1"]*1.0/mean
    for i in data.index:
        temp = temp + alpha.loc[data.loc[i, "HomeTeam"], "alpha_i"] * beta.loc[data.loc[i, "AwayTeam"], "beta_i"]
    A.loc[0, "A_i+1"] = (data_bm["bm"].sum()) / temp
    
    alpha["alpha_i_1"] = alpha["alpha_i"]
    alpha["alpha_i"] = alpha["alpha_i+1"]
    beta["beta_i_1"] = beta["beta_i"]
    beta["beta_i"] = beta["beta_i+1"]
    A["A_i_1"]=A["A_i"]
    A["A_i"]=A["A_i+1"]
    for i in liste_equ:
        for j in liste_equ:
            if i != j:
                x = float(data_2[(data_2.HomeTeam == i) & (data_2.AwayTeam == j)]['FTHG'])
                y = float(data_2[(data_2.HomeTeam == i) & (data_2.AwayTeam == j)]['FTAG'])
                temp_2 += x*math.log(A.loc[0, "A_i+1"]*alpha.loc[i, "alpha_i+1"]*beta.loc[j, "beta_i+1"])-A.loc[0, "A_i+1"]*alpha.loc[i, "alpha_i+1"]*beta.loc[j, "beta_i+1"]-alpha.loc[j, "alpha_i+1"]*beta.loc[i, "beta_i+1"]+y*alpha.loc[j, "alpha_i+1"]*beta.loc[i, "beta_i+1"]-math.log(math.factorial(x))-math.log(math.factorial(y))
    log_vrais.append(temp_2)


alpha.sort_values(["alpha_i"], axis=0, ascending=False,inplace=True)
beta.sort_values(["beta_i"], axis=0, ascending=True,inplace=True)

elapsed_time = time.time() - start_time

# Tracé de la log vraisemblance

plt.plot(list_iter,list_mean_beta)
plt.title("Mean Beta in function of the iterations")
plt.xlabel("Algortihm iterations")
plt.ylabel("Mean of Beta with alpha init = " + str(alpha_0) + " Beta = " + str(beta_0)+ " and A =" + str(A_0))
plt.show()
plt.plot(list_iter,list_A)
plt.title("A")
plt.plot(list_iter,log_vrais)
plt.title("tracé de la log-vraisemblance en fonction du nombre d'itérations")
plt.xlabel("Itérations")
plt.ylabel("l avec alpha init = " + str(alpha_0) + " Beta = " + str(beta_0)+ " et A =" + str(A_0))

list_params=[]

# Calcul de l'écart type

for i in liste_equ:
    list_params.append("alpha "+i)
for i in liste_equ:
    list_params.append("beta "+i)

list_params.append("A")
I= pd.DataFrame(index= list_params,columns=list_params)
for i in range(len(list_params)):
    if i<=19:
        for j in range(len(list_params)):
            if i==j:
                I.iloc[i,j]=float(data_bm.loc[liste_equ[i%20],"bm"]/(alpha.loc[liste_equ[i%20],"alpha_i"])**2+data_benc_dm.loc[liste_equ[i%20],"benc_dm"]/(alpha.loc[liste_equ[i%20],"alpha_i"])**2)
            elif i!=j and j<=19 :
                I.iloc[i,j]=float(0.0)
            elif j>19 and j<(len(list_params)-1):
                I.iloc[i,j]=float(A.loc[0,"A_i"]+1)
            elif j==(len(list_params)-1):
                I.iloc[i, j]=float(beta[beta.index != liste_equ[i%20]]["beta_i"].sum())
    elif 19<i and i<=39:
        for j in range(len(list_params)):
            if i == j:
                I.iloc[i, j] = float(data_bm.loc[liste_equ[i % 20], "bm"] / (beta.loc[liste_equ[i % 20], "beta_i"])**2 + data_benc_dm.loc[liste_equ[i % 20], "benc_dm"] / (beta.loc[liste_equ[i % 20], "beta_i"])**2)
            elif j==(len(list_params)-1):
                I.iloc[i, j] =float(alpha[alpha.index != liste_equ[i%20]]["alpha_i"].sum())
            elif j<=19 :
                I.iloc[i, j] = float(A.loc[0,"A_i"]+1)
            elif j!=i and j>19 and j<(len(list_params)-1):
                I.iloc[i,j] =float(0.0)
    else:
        for j in range(len(list_params)):
            if i == j:
                I.iloc[i, j] = float((data_bm["bm"].sum()) / A.loc[0,"A_i"]**2)
            elif j!=i and j<= 19:
                I.iloc[i, j] = float(beta[beta.index != liste_equ[j%20]]["beta_i"].sum())
            elif j!=i and j>19 and j<= 39:
                I.iloc[i, j] = float(alpha[alpha.index != liste_equ[j % 20]]["alpha_i"].sum())
n=len(data.index)
I=I/n
mat=pd.DataFrame.as_matrix(I)
mat=mat.astype(float)
df_inv = pd.DataFrame(np.linalg.pinv(mat), I.columns, I.index)

# Restitution des résultats

results = pd.DataFrame(index=liste_equ,columns=["alpha","std_alpha","beta","std_beta","A","std_A"])
for i in liste_equ:
    results.loc[i,"alpha"]=alpha.loc[i,"alpha_i"]
    results.loc[i, "beta"] = beta.loc[i, "beta_i"]
    results.loc[i,"std_alpha"]=df_inv.loc["alpha "+i,"alpha "+i]**(-1/2)
    results.loc[i,"std_beta"]=df_inv.loc["beta "+i,"beta "+i]**(-1/2)
    results.loc[i, "A"] = A.loc[0, "A_i"]
    results.loc[i,"std_A"]=df_inv.loc["A","A"]**(-1/2)
    

print (alpha)
print (beta)
print (A)

