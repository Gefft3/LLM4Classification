#!/bin/bash

#Manter o padrão 'Relevantes' e 'Irrelevantes'
# tipo='Relevantes'
tipo='Irrelevantes'

#URL dos datasets de treino e teste
url_train='../datasets/relevantes/EIOS_train.csv'

# url_test='../datasets/relevantes/EIOS_test_filtrado.csv'
url_test='../datasets/irrelevantes/EIOS_test_irrelevantes_filtrado.csv'

#Tamanho máximo do k (número de documentos retornados)
k_max=(1 2 3 4 5 6 7 8)

#Tamanho máximo de tokens por prompt
max_prompt_tokens=128000


for k in "${k_max[@]}"
do
  python3 main.py $url_train $url_test $k $tipo $max_prompt_tokens
done


# python3 main.py $url_train $url_test $k_max $tipo $max_prompt_tokens

echo "Execução finalizada!" 