import pandas as pd
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.document_loaders import DataFrameLoaders
from tqdm import tqdm
import numpy as np
import sys
import tiktoken
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import re
import signal
from concurrent.futures import TimeoutError as TimeoutException
import gc

class Response(BaseModel):
    news_class: str = Field(description="Classe da resposta entre Relevante ou Irrelevante", required=True)
    explain: str = Field(description="Justificativa da resposta da questão", required=True)


def timeout_handler(signum, frame):
    raise TimeoutException("Tempo limite de execução excedido (5 minutos)")


def execute_test(df, max_prompt_length, path_outputs):
    try:

        run_test(df, max_prompt_length, path_outputs)

    except Exception:
        LAST_PROMPT_PROCESSED = verificar_dataframe(path_outputs) + 1

        df = df.iloc[LAST_PROMPT_PROCESSED:]

        prompt, _chain = config_model()
        execute_test(df, max_prompt_length, path_outputs)


def verificar_dataframe(path_outputs):
    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")

    last_prompt_processed = 0

    if os.path.exists(path_arquivo_de_prompts):
        with open(path_arquivo_de_prompts, "r") as f:
            conteudo = f.read().strip().split("--------------------------------\n\n")
            for bloco in conteudo:
                match = re.search(r"Question (\d+)", bloco.strip())
                if match:
                    last_prompt_processed = int(match.group(1))

    return last_prompt_processed


def config_model():
    prompt = PromptTemplate.from_template(
    """
Tarefa: Classifique a questão fornecida como Relevante ou Irrelevante para o contexto da área de saúde epidemiológica. Pense da perspectiva de um profissional de saúde, incluindo médicos, epidemiologistas, enfermeiros, e outros profissionais da área.

Faça:
- Classifique a questão como Relevante ou Irrelevante usando a variável `news_class`.
- A justificativa deve ser clara e concisa, explicando o motivo da classificação, usando a variável `justify`.
- Considere também o contexto fornecido ao fazer a classificação.
  
Não faça:
- Não adicione informações extras além da classificação e justificativa.
- Não forneça mais de uma classificação por questão.
- Não repita a questão ou o contexto na saída.
    
Exemplos:
Questão: 'Tenho dores pulmonares afetadas pelo cigarro.'
Saída esperada: 
news_class: Relevante 
justify: O texto trata de um problema respiratório diretamente relacionado à área da saúde.

Questão: 'Quais são os melhores livros de autoajuda?'
Saída esperada: 
news_class: Irrelevante 
justify: A questão não se relaciona diretamente com o contexto de saúde ou cuidados médicos.
  
A saída deve seguir este formato:
news_class: [Relevante ou Irrelevante]
justify: [Justificativa clara e objetiva]

Questão: {question}

Contexto: {context}
"""
)
    llm = ChatOllama(model="llama3.1", format="json", temperature=0.1)

    structured_llm = llm.with_structured_output(Response)

    _chain = prompt | structured_llm

    gc.collect()

    return prompt, _chain 

def load_data(url_train, url_test):
    train = pd.read_csv(url_train)
    test = pd.read_csv(url_test)
    return train, test

def format_text(text):
    text = text.split("$")
    return {"classe": text[0], "justificativa": text[1]}

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query,k=K_MAX))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs


def ollama_llm(question, context, i, path_outputs):

    response = _chain.invoke({'question': question, 'context': context})

    filled_prompt = prompt.format(question=question, context=context)

    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")

    with open(path_arquivo_de_prompts, "a") as f:
        f.write(f'Question {i}\n')
        f.write(f'{filled_prompt}\n')
        f.write("--------------------------------\n\n")

    return response

    
def rag_chain(question, max_prompt_length, i, path_outputs):    
    retrieved_docs = retriever.invoke(question)

    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens_question = len(encoding.encode(question))
    prompt_tokens = num_tokens_question + 318
    
    formatted_context = ""
    distancias = []
    for doc in retrieved_docs:
        if len(encoding.encode(doc.page_content)) + prompt_tokens < max_prompt_length:
            prompt_tokens += len(encoding.encode(doc.page_content))
            formatted_context += "\n\n" + doc.page_content
            distancias.append(doc.metadata["score"]) 

    return ollama_llm(question, formatted_context, i, path_outputs), distancias

def run_test(df, max_prompt_length, path_outputs):
    
    path_arquivo_de_erros = os.path.join(path_outputs, "Erros.txt")
    path_arquivo_de_classificacoes = os.path.join(path_outputs, "classificacoes.txt")
    path_distancias_raiz = os.path.join(path_outputs, "distancias")

    # if not os.path.exists(path_distancias_raiz):
    #     os.makedirs(path_distancias_raiz)

    i = LAST_PROMPT_PROCESSED

    # signal.signal(signal.SIGALRM, timeout_handler)

    for text in tqdm(df['text']):

        path_arquivo_de_distancias = os.path.join(path_distancias_raiz, f"{i}.txt")

        try:

            response, distancias = rag_chain(text, max_prompt_length, i, path_outputs) 

            # with open(path_arquivo_de_distancias, "w") as f:
            #     for j, distancia in enumerate(distancias):
            #         f.write(f"{j} {distancia}\n")

            choice = response.news_class

            if choice == "Relevante" or choice == "Irrelevante":
                with open(path_arquivo_de_classificacoes, "a") as f:
                    f.write(f"{i} {choice}\n")
                    
            else:
                with open(path_arquivo_de_erros, "a") as f:
                    f.write(f"Question {i}\n")
                    f.write(f"Texto de entrada: {text}\n\n")
                    f.write(f"Resposta LLM: {response}\n\n")
                    f.write(f"Resposta final: {choice}\n\n")
                    f.write("------------------------------------------------\n\n")
                pass
                
        except Exception as e:
            with open(path_arquivo_de_erros, "a") as f:
                f.write(f"Question {i}\n")
                f.write(f"Texto de entrada: {text}\n\n")
                f.write(f"Erro (exception): {e}\n\n")
                f.write("------------------------------------------------\n\n")
            pass

        # except TimeoutException:
        #     raise Exception
        

        i += 1

if __name__ == "__main__":
    
    
    K_MAX = int(sys.argv[3])
    tipo = sys.argv[4]
    max_prompt_length = int(sys.argv[5])

    path_outputs = f'Logs {tipo}/k = {K_MAX}'

    df_train, df_test = load_data(sys.argv[1], sys.argv[2])

    if not os.path.exists(path_outputs):
            os.makedirs(path_outputs)

    #Carregando os documentos de treino
    # loader = DataFrameLoader(df_train, page_content_column="text")
    # docs = loader.load()

    #Criando instanciando modelo de embedding e criando a vectorstore
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma(collection_name='v_db', persist_directory="./chroma_db", embedding_function=embeddings)
    
    #Criando o modelo de recuperação
    # retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': k_max})

    LAST_PROMPT_PROCESSED = verificar_dataframe(path_outputs) 
    print(LAST_PROMPT_PROCESSED)

    df_test = df_test.iloc[LAST_PROMPT_PROCESSED:]
    prompt, _chain = config_model()
    run_test(df_test, max_prompt_length, path_outputs)
    # execute_test(df_test, max_prompt_length, path_outputs)