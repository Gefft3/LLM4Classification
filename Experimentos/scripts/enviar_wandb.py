import wandb 
import sys
import os
import numpy as np

"""
Para rodar, execute o comando: python3 enviar_wandb.py <tipo>
<tipo> é o tipo de classificação que deseja analisar, podendo ser 'Relevantes' ou 'Irrelevantes'
"""


run = wandb.init(
    project='ECOS da Rota'
    )

def calculate_acurracy(classificacoes, tipo):
    acurracy = 0

    correto = 'Relevante' if tipo == 'Relevantes' else 'Irrelevante'

    for classificacao in classificacoes:
        if classificacao == correto:
            acurracy += 1
    return acurracy/len(classificacoes)

tipo = sys.argv[1]

path_raiz = f'../Experimentos/Logs {tipo}'
pastas = os.listdir(path_raiz)

lista_pastas = []
for pasta in pastas:
    try: 
        lista_pastas.append(int(pasta.split('=')[1])) 
    except:
        pass

lista_pastas = sorted(lista_pastas)

for index, pasta in enumerate(lista_pastas):
    pastas[index] = f'k = {pasta}'

for pasta in pastas:
    if 'k =' in pasta:
        
        distance_mean = []
        acurracy_mean = []
        
        path_outputs = os.path.join(path_raiz, pasta)
        for arquivos in os.listdir(path_outputs):
            if 'distancias' in arquivos:
                with open(os.path.join(path_outputs, arquivos), "r") as f:
                    distancias = f.readlines()
                    distancias = [float(x.split()[1]) for x in distancias]
                    distance_mean.append(np.mean(distancias))

            if 'classificacoes' in arquivos:
                with open(os.path.join(path_outputs, arquivos), "r") as f:
                    classificacoes = f.readlines()
                    classificacoes = [x.split()[1] for x in classificacoes]
                    acurracy_mean.append(calculate_acurracy(classificacoes, tipo))

            
        if len(distance_mean) == 0:
            distance_mean = [0]
        print(f'K: {pasta}')
        print(f'Distancia média: {distance_mean}')
        print(f'Acurácia média: {acurracy_mean}')

        wandb.log({
                "distance_mean": np.mean(distance_mean),
                "acurracy_mean": np.mean(acurracy_mean)
            })