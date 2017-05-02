# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:16:11 2017

@author: Amine
"""

import pandas as pd
import numpy as np
import copy
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import poisson

data = pd.read_csv("2016-2017.csv")
data_2=data.copy()
liste_equ=list(set(data.HomeTeam))

nb_journee = 30
results = pd.DataFrame(index=liste_equ)
for journee in range(21,nb_journee+1):
    liste_bm = [0] * len(liste_equ)
    liste_be = [0] * len(liste_equ)
    sub_data=data.iloc[0:(journee-1)*10]
#DataFrame but marqués à domicile et à l'extérieur
    data_bm=pd.DataFrame(index=liste_equ, columns=["bm"])
    data_be=pd.DataFrame(index=liste_equ,columns=["be"])
    #DataFrame but encaissés à domicile et à l'extérieur
    data_benc_dm=pd.DataFrame(index=liste_equ, columns=["benc_dm"])
    data_benc_ext=pd.DataFrame(index=liste_equ,columns=["benc_ext"])

    # le nombre de but marque et encaisse pour chaque équipe à domicile :
    for i in range(len(liste_equ)):
        liste_bm[i] = sub_data[sub_data.HomeTeam == liste_equ[i]]['FTHG'].sum()

    #On remplit les Dataframes de buts marqués et encaissés pour chaque équipe à domicile
    for i in liste_equ:
        data_bm.loc[i,"bm"]=sub_data[sub_data.HomeTeam == i]['FTHG'].sum()
        data_benc_dm.loc[i,"benc_dm"]=sub_data[sub_data.HomeTeam == i]['FTAG'].sum()

    # le nombre de buts marqués pour chaque équipe à l'extérieur :
    for i in range(len(liste_equ)):
        liste_be[i] = sub_data[sub_data.AwayTeam == liste_equ[i]]['FTAG'].sum()

    #On remplit les Dataframes de buts marques et encaissés pour chaque équipe à l'extérieur

    for i in liste_equ:
        data_be.loc[i, "be"]=sub_data[sub_data.AwayTeam == i]['FTAG'].sum()
        data_benc_ext.loc[i,"benc_ext"]=sub_data[sub_data.AwayTeam == i]['FTHG'].sum()


    # Parametre epsilon
    epsilon = 0.001

    # Initalisation de alpha, beta et A
    alpha_0=3
    beta_0=0.9
    A_0 = 3
    # On créée une dataframe pour nos parametres Alpha Beta et A
    alpha=pd.DataFrame(index=liste_equ,columns=["alpha_i_1","alpha_i","alpha_i+1"])
    beta = pd.DataFrame(index=liste_equ,columns=["beta_i_1","beta_i","beta_i+1"])
    A = pd.DataFrame(index=[0],columns=["A_i_1","A_i","A_i+1"])


    #Initialisation des alpha, beta et A
    for i in liste_equ:
        alpha.loc[i,"alpha_i_1"]=alpha_0
        beta.loc[i,"beta_i_1"]=beta_0
        A.loc[0,"A_i_1"]=A_0

    #mean = beta["beta_i_1"].mean()
    #beta["beta_i_1"]=beta["beta_i_1"]*1.0/mean
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

    #mean = beta["beta_i"].mean()
    #beta["beta_i"] = beta["beta_i"] *1.0/ mean

    #Calcul du premier A

    temp=0
    for i in sub_data.index:
        temp=temp+alpha.loc[sub_data.loc[i,"HomeTeam"],"alpha_i_1"]*beta.loc[sub_data.loc[i,"AwayTeam"],"beta_i_1"]
    A.loc[0,"A_i"]=(data_bm["bm"].sum())/temp
    iter =0

    list_iter=[iter]
    list_mean_beta=[]
    list_mean_beta.append(beta["beta_i"].mean())
    list_A=[]
    list_A.append(A["A_i"])

    # Algorithme de minimisation alternée :

    while abs(beta["beta_i"]-beta["beta_i_1"]).mean()>epsilon:
        temp=0
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
        for i in sub_data.index:
            temp = temp + alpha.loc[sub_data.loc[i, "HomeTeam"], "alpha_i"] * beta.loc[sub_data.loc[i, "AwayTeam"], "beta_i"]
        A.loc[0, "A_i+1"] = (data_bm["bm"].sum()) / temp

        alpha["alpha_i_1"] = alpha["alpha_i"]
        alpha["alpha_i"] = alpha["alpha_i+1"]
        beta["beta_i_1"] = beta["beta_i"]
        beta["beta_i"] = beta["beta_i+1"]
        A["A_i"]=A["A_i+1"]

    alpha.sort_values(["alpha_i"], axis=0, ascending=False,inplace=True)
    beta.sort_values(["beta_i"], axis=0, ascending=True,inplace=True)
    
    for i in liste_equ:
        results.loc[i,"alpha_" +str(journee)] = alpha.loc[i,"alpha_i"]
        results.loc[i,"beta_" + str(journee)] = beta.loc[i, "beta_i"]
        results.loc[i,"A_" +str(journee)] =A.loc[0, "A_i"]
        
print (results)    

list_equ=liste_equ
# Calcul des probas
probas_estim=pd.DataFrame(index=list(range(200,(nb_journee)*10)),columns=["Home","Draw","Away"])
for journee in range(int(len(results.columns)/3)):
    journee+=21
    A = results["A_"+str(journee)][0]
    proba_df=pd.DataFrame(index=list_equ,columns=list_equ)
    data_games= data.loc[(journee-1)*10:(journee)*10-1]
    list_games=list(range((journee-1)*10,(journee)*10))
    for game in list_games:
        i = data_games.loc[game,"HomeTeam"]
        j = data_games.loc[game,"AwayTeam"]
        list_result = [0] * 3
        proba_win_H=0
        proba_win_A=0
        proba_draw=0
        for home in range(0,11):
            for away in range(0,11):
                alpha_i = results.loc[i, "alpha_"+str(journee)]
                beta_j = results.loc[j, "beta_"+str(journee)]
                alpha_j = results.loc[j, "alpha_"+str(journee)]
                beta_i = results.loc[i, "beta_"+str(journee)]
                lambda_h = alpha_i*beta_j*A
                lambda_a = alpha_j*beta_i*A
                if home>away:
                    proba_win_H += poisson.pmf(home,lambda_h)*poisson.pmf(away,lambda_a)
                elif home == away:
                    proba_draw += poisson.pmf(home,lambda_h)*poisson.pmf(away,lambda_a)
                else:
                    proba_win_A += poisson.pmf(home,lambda_h)*poisson.pmf(away,lambda_a)
        list_result[0] = proba_win_H
        list_result[1] = proba_draw
        list_result[2] = proba_win_A
        probas_estim.loc[game,"Home"]=proba_win_H
        probas_estim.loc[game,"Draw"]=proba_draw
        probas_estim.loc[game,"Away"]=proba_win_A
        
cotes=data[["B365H","B365D","B365A"]]
proba_bookie=1/cotes
for i in (proba_bookie.index):
    somme=proba_bookie.loc[i,:].sum()
    for col in proba_bookie.columns:
        proba_bookie.loc[i,col]=proba_bookie.loc[i,col]/somme
proba_bookie=proba_bookie.loc[20*10:(nb_journee)*10-1]
cotes=cotes.loc[20*10:(nb_journee)*10-1]    

expectancy = np.array(probas_estim)/np.array(proba_bookie)-1

dummies=pd.get_dummies(pd.DataFrame(data.loc[:,'FTR'])).astype(int)
#dummies=dummies*2-1
dummies=dummies.loc[20*10:(nb_journee)*10-1]

dummies_negative = dummies-1

proba_bookie.columns=['FTR_H','FTR_D','FTR_A']
cotes.columns=['FTR_H','FTR_D','FTR_A']

bet=expectancy>0.5
bet.astype(int)
bet_df=pd.DataFrame(data=bet[0:,0:],index=list(range(20*10,(nb_journee)*10)),columns=['FTR_H','FTR_D','FTR_A'])
bet_df=bet_df.astype(int)

gagnant = (dummies*bet_df*cotes).sum()
perdant = (bet_df).sum()
pari=(gagnant-perdant).sum()