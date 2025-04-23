import pandas as pd
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm
import sys
import os
import re
import gc

# Modelo de resposta estruturada para a tarefa de filtragem
class Response(BaseModel):
    news_class: str = Field(description="Classificação se o HTML contém notícia ou não", required=True)
    justify: str = Field(description="Justificativa da filtragem realizada", required=True)

# Configuração do prompt modificado para tarefa de filtragem do HTML
def config_model():
    prompt = PromptTemplate.from_template(
        """
Tarefa: Verifique se o conteúdo HTML fornecido apresenta informações de notícia (por exemplo, títulos, parágrafos informativos, elementos jornalísticos) ou se se trata apenas de código CSS/JS sem informações relevantes.

Faça:
- Se o HTML apresenta informações relacionadas a uma notícia, responda com "Notícia".
- Se o HTML não apresenta informações relevantes, responda com "Sem Informação".
- Justifique a sua resposta de forma clara e concisa, mencionando os elementos encontrados ou a ausência deles.

Formato da saída:
news_class: [Notícia ou Sem Informação]
justify: [Justificativa breve]

HTML: {html_content}
"""
    )
    llm = ChatOllama(model="llama3.2", format="json", temperature=0.1)
    structured_llm = llm.with_structured_output(Response)
    _chain = prompt | structured_llm

    gc.collect()
    return prompt, _chain

# Função para carregar os dados do dataset
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Recupera o último registro processado a partir dos logs
def verificar_dataframe(path_outputs):
    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")
    last_prompt_processed = 0

    if os.path.exists(path_arquivo_de_prompts):
        with open(path_arquivo_de_prompts, "r", encoding="utf-8") as f:
            conteudo = f.read().strip().split("--------------------------------\n\n")
            for bloco in conteudo:
                match = re.search(r"Question (\d+)", bloco.strip())
                if match:
                    last_prompt_processed = int(match.group(1))
    return last_prompt_processed

# Encapsula a chamada ao LLM passando o conteúdo HTML
def ollama_llm(html_content, i, path_outputs, prompt, _chain):
    response = _chain.invoke({'html_content': html_content})
    filled_prompt = prompt.format(html_content=html_content)
    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")
    with open(path_arquivo_de_prompts, "a", encoding="utf-8") as f:
        f.write(f'Question {i}\n')
        f.write(f'{filled_prompt}\n')
        f.write("--------------------------------\n\n")
    return response

# Processa cada registro do DataFrame
def run_test(df, path_outputs, prompt, _chain):
    path_arquivo_de_erros = os.path.join(path_outputs, "Erros.txt")
    path_arquivo_de_filtragens = os.path.join(path_outputs, "filtragens.txt")
    i = verificar_dataframe(path_outputs)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        html_content = row.get("html", "")
        short_url = row.get("short_url", "")
        expanded_url = row.get("expanded_url", "")

        try:
            response = ollama_llm(html_content, i, path_outputs, prompt, _chain)
            choice = response.news_class

            if choice in ["Notícia", "Sem Informação"]:
                with open(path_arquivo_de_filtragens, "a", encoding="utf-8") as f:
                    f.write(f"{i} | short_url: {short_url} | expanded_url: {expanded_url} | {choice}\n")
            else:
                with open(path_arquivo_de_erros, "a", encoding="utf-8") as f:
                    f.write(f"Question {i}\nHTML: {html_content}\n\nResposta LLM: {response}\n\nResposta final: {choice}\n\n------------------------------------------------\n\n")
        except Exception as e:
            with open(path_arquivo_de_erros, "a", encoding="utf-8") as f:
                f.write(f"Question {i}\nHTML: {html_content}\n\nErro (exception): {e}\n\n------------------------------------------------\n\n")
        i += 1

if __name__ == "__main__":
  
    csv_path = sys.argv[1]

    path_outputs = f'/outputs'

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    df = load_data(csv_path)
    prompt, _chain = config_model()
    ultimo = verificar_dataframe(path_outputs)
    df = df.iloc[ultimo:]

    run_test(df, path_outputs, prompt, _chain)
