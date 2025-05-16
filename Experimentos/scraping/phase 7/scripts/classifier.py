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
Você está recebendo um texto de uma página e o resumo do seu conteúdo principal.

Tarefa:
- Analise se o trecho do texto apresentado tem relação direta com o resumo fornecido.
- Relacione elementos informativos como frases, palavras-chave e tópicos principais.

Formato da saída:
{{
  "is_relevant": true|false,
  "justify": "Justificativa da decisão"
}}

Resumo:
{summary}

Chunk do texto:
{text_chunk}
"""
    )

    llm = ChatOllama(model="llama3.2", format="json", temperature=0.1)
    structured_llm = llm.with_structured_output(Response)
    _chain = prompt | structured_llm

    gc.collect()
    return prompt, _chain

# Chunkinização usando tokenização aproximada
def chunk_text(text, max_tokens=2000):
    if not isinstance(text, str):
        text = str(text) 
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

# Carregamento de dados
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Processamento por chunk
def process_text_chunks(i, text, summary, _chain, path_outputs):
    chunks = chunk_text(text, max_tokens=2000)
    classes = []

    for idx, chunk in enumerate(chunks):
        try:
            # Salva o prompt de entrada
            with open(os.path.join(path_outputs, "prompts.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Resumo: {summary}\nChunk:\n{chunk}\n-----------------------\n")

            # Obtém a resposta do modelo
            response = _chain.invoke({'text_chunk': chunk, 'summary': summary})
            is_rel = response.is_relevant
            justificativa = response.justify

            # Salva a resposta completa
            with open(os.path.join(path_outputs, "responses.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Resposta:\n{response}\n-----------------------\n")

            # Salva a classificação
            with open(os.path.join(path_outputs, "classes.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Relevant: {is_rel}\n-----------------------\n")

            if is_rel:
                classes.append((chunk, justificativa))

        except Exception as e:
            # Salva os erros
            with open(os.path.join(path_outputs, "errors.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{i}] Chunk {idx} | Erro: {e}\n-----------------------\n")
    return classes, len(chunks)

# Processa o CSV completo
def run(df, path_outputs, _chain):
    resultados_path = os.path.join(path_outputs, "relevantes.txt")
    total_chunks = 0
    total_rows = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row.get("html_content", "")
        if not isinstance(text, str):
            text = str(text)  # Ensure text is a string
        summary = row.get("summary", "")
        short_url = row.get("short_url", "")
        expanded_url = row.get("expanded_url", "")

        classes, num_chunks = process_text_chunks(i, text, summary, _chain, path_outputs)
        total_chunks += num_chunks
        total_rows += 1

        with open(resultados_path, "a", encoding="utf-8") as f:
            for idx, (chunk, justificativa) in enumerate(classes):
                f.write(f"[{i}] Chunk relevante {idx} | short_url: {short_url} | expanded_url: {expanded_url}\nJustificativa: {justificativa}\nChunk:\n{chunk}\n--------------------\n\n")

    # Calcula e registra a média de chunks por documento
    average = total_chunks / total_rows if total_rows > 0 else 0
    with open(os.path.join(path_outputs, "average_chunks.yml"), "w", encoding="utf-8") as f:
        f.write(f"Total documentos processados: {total_rows}\n")
        f.write(f"Total de chunks criados: {total_chunks}\n")
        f.write(f"Média de chunks por documento: {average:.2f}\n")

if __name__ == "__main__":
    # Espera caminho relativo para pasta data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", sys.argv[1])  
    class_type = sys.argv[2]
    path_outputs = os.path.join(os.path.dirname(__file__), "..", "logs")

    path_outputs = os.path.join(path_outputs, class_type)

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    df = load_data(csv_path)
    prompt, _chain = config_model()
    run(df, path_outputs, _chain)