import pandas as pd
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm
import tiktoken  
import sys
import os
import gc

# Modelo de resposta estruturada para a tarefa de filtragem
class Response(BaseModel):
    news_class: str = Field(description="Classificação se o HTML contém notícia ou não", required=True)
    justify: str = Field(description="Justificativa da filtragem realizada", required=True)

# Configuração do prompt
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

# Função para tokenização
def chunk_text(text, max_tokens=2000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

# Função para carregar os dados do dataset
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Chamada ao LLM para cada chunk
def ollama_llm(html_content, i, path_outputs, prompt, _chain):
    chunks = chunk_text(html_content, max_tokens=2000)
    respostas = []
    for idx, chunk in enumerate(chunks):
        try:
            response = _chain.invoke({'html_content': chunk})
            respostas.append(response)
            filled_prompt = prompt.format(html_content=chunk)
            with open(os.path.join(path_outputs, "prompts.txt"), "a", encoding="utf-8") as f:
                f.write(f'Question {i} - chunk {idx}\n')
                f.write(f'{filled_prompt}\n')
                f.write("--------------------------------\n\n")
        except Exception as e:
            respostas.append(Response(news_class="Erro", justify=str(e)))
    return respostas

# Processa cada registro do DataFrame
def run_test(df, path_outputs, prompt, _chain):
    path_arquivo_de_erros = os.path.join(path_outputs, "Erros.txt")
    path_arquivo_de_filtragens = os.path.join(path_outputs, "filtragens.txt")
    i = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        html_content = row.get("html_content", "")
        short_url = row.get("short_url", "")
        expanded_url = row.get("expanded_url", "")

        try:
            respostas = ollama_llm(html_content, i, path_outputs, prompt, _chain)

            # Agregando resultados
            classes = [r.news_class for r in respostas if isinstance(r, Response)]
            justifications = [r.justify for r in respostas if isinstance(r, Response)]

            if "Notícia" in classes:
                final_class = "Notícia"
            elif all(c == "Sem Informação" for c in classes):
                final_class = "Sem Informação"
            else:
                final_class = "Erro"

            if final_class in ["Notícia", "Sem Informação"]:
                with open(path_arquivo_de_filtragens, "a", encoding="utf-8") as f:
                    f.write(f"{i} | short_url: {short_url} | expanded_url: {expanded_url} | {final_class}\nJustificativas: {' || '.join(justifications)}\n\n")
            else:
                with open(path_arquivo_de_erros, "a", encoding="utf-8") as f:
                    f.write(f"Question {i}\nHTML: {html_content}\n\nRespostas: {respostas}\n\n------------------------------------------------\n\n")
        except Exception as e:
            with open(path_arquivo_de_erros, "a", encoding="utf-8") as f:
                f.write(f"Question {i}\nHTML: {html_content}\n\nErro (exception): {e}\n\n------------------------------------------------\n\n")
        i += 1

if __name__ == "__main__":
    csv_path = sys.argv[1]
    path_outputs = '../outputs'

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    df = load_data(csv_path)
    prompt, _chain = config_model()
    run_test(df, path_outputs, prompt, _chain)
