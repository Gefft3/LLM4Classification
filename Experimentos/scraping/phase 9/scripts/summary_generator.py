import pandas as pd
import os
import gc
import time
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import tiktoken
import sys

class SummaryResponse(BaseModel):
    summary: str = Field(
        description="Resumo gerado para o texto dado",
        required=True
    )


def config_model():
    """
    Configura o modelo para gerar resumos de notícias.

    Returns:
        tuple: PromptTemplate e chain configurado para sumarização.
    """
    prompt = PromptTemplate.from_template(
        """
        Você é um gerador de resumo de notícias.

        Gere um resumo conciso que capture os pontos principais do texto e sua respectiva versão resumida.

        Texto completo: {text}
        Resumo atual: {existing_summary}

        Seu resumo final deve integrar informações de ambos e ser claro. E não deve incluir bullet points ou listas.
        """
    )

    llm = ChatOllama(model="llama3.2", format="json", temperature=0.8)
    structured_llm = llm.with_structured_output(SummaryResponse)
    chain = prompt | structured_llm
    gc.collect()

    return prompt, chain


def split_text(text: str, model: str = "cl100k_base", max_tokens: int = 2000) -> list:
    """
    Divide um texto em segmentos de no máximo max_tokens tokens.
    """
    encoder = tiktoken.get_encoding(model)
    tokens = encoder.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoder.decode(chunk) for chunk in chunks]


def generate_summary(text, existing_summary, prompt, chain):
    """
    Gera o resumo final integrando o texto original e o resumo existente.
    """
    # text_chunks = split_text(text)
    # combined_summary = None

    # for chunk in tqdm(text_chunks, desc="Processing text chunks"):
    #     response = chain.invoke({"text": chunk, "existing_summary": existing_summary})
    #     if response and response.summary:
    #         combined_summary = response.summary
    # return combined_summary
    
    response = chain.invoke({"text": text, "existing_summary": existing_summary})
    if response and response.summary:
        return response.summary
    else:
        return 'Erro ao gerar resumo'

def run(news_type, path_input, path_output, prompt, chain):
    """
    Executa a geração de resumos para cada notícia no dataset.
    """
    df = pd.read_csv(path_input)
    summaries = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.get('html_content', '')
        existing = row.get('summary', '')
        summary = generate_summary(text, existing, prompt, chain)
        summaries.append(summary)
        time.sleep(0.5)

    df['generated_summary'] = summaries
    output_dir = os.path.join("../data", news_type)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, path_output)
    df.to_csv(output_path, index=False)


def main():
    _, news_type, path_input, path_output = sys.argv

    prompt, chain = config_model()
    run(news_type, path_input, path_output, prompt, chain)


if __name__ == '__main__':
    main()
