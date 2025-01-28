import pandas as pd
import os
import re
import gc
from tqdm import tqdm
import time
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class Response(BaseModel):
    is_in_scraped_text: bool = Field(description="Classe se o texto fornecido está no scraped_text, sendo que seu conteúdo deve ser True ou False ", required=True)

def config_model():
    prompt = PromptTemplate.from_template(
    """
Você é um classificador, você irá receber dois textos, um texto e um conteúdo do scraping, e deverá classificar se o texto informado está presente no conteúdo de scraping. Sua saída deve ser exclusivamente True ou False. No caso, se o está está presente no conteúdo scraping, a saída deve ser True, caso contrário, False. 

Tarefa: Classificar se o texto informado está presente no conteúdo de scraping.

Faça:
- Classifique se o texto informado está presente no conteúdo do scraping.
- A saída deve ser exclusivamente True ou False.
  
Não faça:
- Não adicione informações extras além da classificação e justificativa.
- Não forneça mais de uma classificação por questão.
- Não repita a questão ou o contexto na saída.
  
A saída deve seguir este formato:
response: [True ou False]

Texto: {text}

Scraping: {scraped_text}
"""
)
    llm = ChatOllama(model="llama3.2", format="json", temperature=0.1)

    structured_llm = llm.with_structured_output(Response)

    chain = prompt | structured_llm

    gc.collect()

    return prompt, chain

def ollama_response(text, scraped_text, i, path_outputs, prompt, chain):

    response = chain.invoke({'text': text, 'scraped_text': scraped_text})

    filled_prompt = prompt.format(text=text, scraped_text=scraped_text)

    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")

    with open(path_arquivo_de_prompts, "a") as f:
        f.write(f'Question {i}\n')
        f.write(f'{filled_prompt}\n')
        f.write("--------------------------------\n\n")

    return response


def extract_error_indexes(file):
    indexes = []
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r'index (\d+) - url:', line)
            if match:
                indexes.append(int(match.group(1)))
    return indexes


def get_indexes():
    root_path  = '../Resultados/scraping content/relevant'

    files = os.listdir(root_path)

    indexes = []

    errors_indexes = []

    for file in files:
        if file != 'txt_sizes.txt': 
            if file == 'erros.txt':
                errors_indexes = extract_error_indexes(root_path + '/' + file)
            else:
                indexes.append(int(file.split('.')[0]))

    indexes.sort()

    files_sorted = []

    for index in indexes:
        files_sorted.append(str(index) + '.txt')

    return files_sorted, errors_indexes

def run(dataset, files, errors, path_outputs, prompt, chain):
    
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        
        # Sleep llama to avoid timeout
        time.sleep(5)

        text = row['content']

        if i in errors or f'{i}.txt' not in files:
            continue

        scraped_text = open(f'../Resultados/scraping content/relevant/{i}.txt').read()

        response = ollama_response(text, scraped_text, i, path_outputs, prompt, chain)
        try:
            with open(os.path.join(path_outputs, "classificacoes.txt"), "a") as f:
                f.write(f'Question {i}\n{response.is_in_scraped_text}\n--------------------------------\n\n')

        except Exception as e:
            with open(os.path.join(path_outputs, "erros.txt"), "a") as f:
                f.write(f'Question {i}\n{str(e)}\n--------------------------------\n\n')
                
def main():

    # Load dataset with summary of the content
    dataset = pd.read_csv('../../datasets/relevantes/dataset_content.csv')

    # Load the model
    prompt, chain = config_model()

    # Get the indexes of the files sorted
    files, errors = get_indexes()

    # Path to save the outputs
    path_outputs = '../Resultados/filter'

    run(dataset, files, errors, path_outputs, prompt, chain)
    

if __name__ == '__main__':
    main()

