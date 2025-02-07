import pandas as pd
import os
import re
import gc
from tqdm import tqdm
import time
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import tiktoken

class Response(BaseModel):
    is_in_scraped_text: bool = Field(
        description="Indicates whether the provided text is present in the 'scraped_text' dataset. "
                    "The value can be 'True' if the text is present, or 'False' if it is not.",
        required=True
    )

def config_model():
    """
    Configura o modelo para verificar se um texto está presente no conteúdo do scraping.

    Returns:
        tuple: PromptTemplate e chain configurado.
    """
    prompt = PromptTemplate.from_template(
        """
        Você é um classificador que verifica se um texto está presente no conteúdo de scraping.

        A saída deve ser um JSON com a chave "is_in_scraped_text" e um valor booleano.

        Exemplo de saída:
        {{ "is_in_scraped_text": true }}

        Texto: {text}

        Scraping: {scraped_text}
        """
    )

    llm = ChatOllama(model="llama3.2", format="json", temperature=0.1)

    structured_llm = llm.with_structured_output(Response)

    chain = prompt | structured_llm

    gc.collect()

    return prompt, chain

def split_text(text: str, model: str = "cl100k_base", max_tokens: int = 2000) -> list:
    """
    Divide um texto em segmentos de no máximo max_tokens tokens.

    Args:
        text (str): Texto a ser dividido.
        model (str): Modelo para tokenização (padrão: cl100k_base).
        max_tokens (int): Número máximo de tokens por segmento.

    Returns:
        list: Lista de segmentos de texto.
    """
    encoder = tiktoken.get_encoding(model)
    tokens = encoder.encode(text)
    
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    return [encoder.decode(chunk) for chunk in chunks]

def ollama_response(text, scraped_text, i, path_outputs, prompt, chain):
    """
    Gera a resposta do modelo de linguagem para a tarefa de classificação de texto em conteúdo de scraping.

    Args:
        text (str): Texto a ser classificado.
        scraped_text (str): Conteúdo do scraping.
        i (int): Número da questão.
        path_outputs (str): Caminho para o diretório de saída.
        prompt (PromptTemplate): Prompt para a tarefa de classificação de texto em conteúdo de scraping.
        chain: Modelo de linguagem configurado para a tarefa de classificação de texto em conteúdo de scraping.

    Returns:
        Response: Resposta do modelo de linguagem para a tarefa de classificação de texto em conteúdo de scraping.
    """

    scraped_text_splits = split_text(scraped_text)
    

    for split in scraped_text_splits:
        response = chain.invoke({'text': text, 'scraped_text': split})
        if response is not None and response.is_in_scraped_text:
            break

    filled_prompt = prompt.format(text=text, scraped_text=scraped_text)

    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")

    with open(path_arquivo_de_prompts, "a") as f:
        f.write(f'Question {i}\n')
        f.write(f'{filled_prompt}\n')
        f.write("--------------------------------\n\n")

    return response


def run(dataset_content, dataset_scraping, path_outputs, prompt, chain):
    """
    Executa o experimento de classificação de texto em conteúdo de scraping.

    Args:
        dataset_content (pd.DataFrame): Dataset com os textos a serem classificados.
        dataset_scraping (pd.DataFrame): Dataset com os conteúdos de scraping.
        path_outputs (str): Caminho para o diretório de saída.
        prompt (PromptTemplate): Prompt para a tarefa de classificação de texto em conteúdo de scraping.
        chain: Modelo de linguagem configurado para a tarefa de classificação de texto em conteúdo de scraping.
    """


    responses = []
    output_file = os.path.join(path_outputs, 'responses.csv')
    
    write_header = not os.path.exists(output_file)
    
    for i, row in tqdm(dataset_content.iterrows(), total=dataset_content.shape[0]):
        text = row['content']
        scraped_text = dataset_scraping.loc[i, 'content']

        response = ollama_response(text, scraped_text, i, path_outputs, prompt, chain)
        responses.append(response)
        
        pd.DataFrame([response]).to_csv(output_file, mode='a', header=write_header, index=False)
        write_header = False 
        
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

