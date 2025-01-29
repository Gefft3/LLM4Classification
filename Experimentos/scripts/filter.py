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


def run(dataset_content, dataset_scraping, path_outputs, prompt, chain):
    responses = []
    output_file = os.path.join(path_outputs, 'responses.csv')
    
    # Verifica se o arquivo já existe para definir se precisa incluir cabeçalho
    write_header = not os.path.exists(output_file)
    
    for i, row in tqdm(dataset_content.iterrows(), total=dataset_content.shape[0]):
        text = row['content']
        scraped_text = dataset_scraping.loc[i, 'content']

        response = ollama_response(text, scraped_text, i, path_outputs, prompt, chain)
        responses.append(response)
        
        # Salva o resultado a cada iteração
        pd.DataFrame([response]).to_csv(output_file, mode='a', header=write_header, index=False)
        write_header = False  # Apenas na primeira iteração o cabeçalho será escrito
        
        time.sleep(0.5)
                
def main():

    # Paths 
    path_dataset = '../../datasets/relevantes'
    path_outputs = '../Resultados/filter'

    # Create output directory
    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    # Load datasets
    print("Loading datasets...")
    print("Loading dataset_content.csv...")
    dataset_content = pd.read_csv(os.path.join(path_dataset, 'dataset_content.csv'))
    print("Loading dataset_scraping.csv...")
    dataset_scraping = pd.read_csv(os.path.join(path_dataset, 'dataset_scraping.csv'))

    # Load the model
    prompt, chain = config_model()

    # Run experiment
    run(dataset_content, dataset_scraping, path_outputs, prompt, chain)
    

if __name__ == '__main__':
    main()

