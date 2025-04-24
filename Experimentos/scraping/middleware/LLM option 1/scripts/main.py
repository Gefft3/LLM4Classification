import pandas as pd
import os
import sys
import gc
import tiktoken
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Modelo de resposta estruturada
class Response(BaseModel):
    is_relevant: bool = Field(description="Se o conteúdo do chunk está relacionado ao resumo.")
    justify: str = Field(description="Justificativa para a decisão de relevância.")

# Configura o modelo e prompt
def config_model():
    prompt = PromptTemplate.from_template(
        """
Você está recebendo um HTML completo de uma página e o resumo do seu conteúdo principal.

Tarefa:
- Analise se o trecho do HTML apresentado tem relação direta com o resumo fornecido.
- Relacione elementos informativos como frases, palavras-chave e tópicos principais.

Formato da saída:
is_relevant: [true ou false]
justify: [Justificativa da decisão]

Resumo:
{abstract}

Chunk de HTML:
{html_chunk}
"""
    )

    llm = ChatOllama(model="llama3.2", format="json", temperature=0.1)
    structured_llm = llm.with_structured_output(Response)
    _chain = prompt | structured_llm

    gc.collect()
    return prompt, _chain

# Chunkinização usando tokenização aproximada
def chunk_text(text, max_tokens=2000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

# Carregamento de dados
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Processamento por chunk
def process_html_chunks(i, html, abstract, _chain, path_outputs):
    chunks = chunk_text(html, max_tokens=2000)
    relevantes = []

    for idx, chunk in enumerate(chunks):
        try:
            # Salva o prompt de entrada
            with open(os.path.join(path_outputs, "prompts.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Resumo: {abstract}\nChunk:\n{chunk}\n-----------------------\n")

            # Obtém a resposta do modelo
            response = _chain.invoke({'html_chunk': chunk, 'abstract': abstract})
            is_rel = response.is_relevant
            justificativa = response.justify

            # Salva a resposta completa
            with open(os.path.join(path_outputs, "responses.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Resposta:\n{response}\n-----------------------\n")

            # Salva a classificação
            with open(os.path.join(path_outputs, "classes.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Relevant: {is_rel}\nJustificativa: {justificativa}\n-----------------------\n")

            if is_rel:
                relevantes.append((chunk, justificativa))

        except Exception as e:
            # Salva os erros
            with open(os.path.join(path_outputs, "errors.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Erro: {e}\n-----------------------\n")
    return relevantes

# Processa o CSV completo
def run(df, path_outputs, _chain):
 
    resultados_path = os.path.join(path_outputs, "relevantes.txt")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        html = row.get("html_content", "")
        abstract = row.get("abstract", "")
        short_url = row.get("short_url", "")
        expanded_url = row.get("expanded_url", "")

        relevantes = process_html_chunks(i, html, abstract, _chain, path_outputs)

        with open(resultados_path, "a", encoding="utf-8") as f:
            for idx, (chunk, justificativa) in enumerate(relevantes):
                f.write(f"[{i}] Chunk relevante {idx} | short_url: {short_url} | expanded_url: {expanded_url}\nJustificativa: {justificativa}\nChunk:\n{chunk}\n--------------------\n\n")

if __name__ == "__main__":
    csv_path = sys.argv[1] 
    path_outputs = "../outputs"

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    df = load_data(csv_path)
    prompt, _chain = config_model()
    run(df, path_outputs, _chain)
