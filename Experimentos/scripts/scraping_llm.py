import pandas as pd
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import gc
import ollama  

def fetch_page_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        main_content = soup.find("main") or soup.find("article")   
        
        if main_content:
            text = main_content.get_text(separator="\n").strip()
        else:
            text = soup.get_text(separator="\n").strip()

        return text if text else "Conteúdo vazio ou indisponível."
    except Exception as e:
        return f"Erro ao acessar o link: {e}"


def call_ollama(content, model):
    prompt = f"""
Você é um assistente especializado em gerar resumos detalhados.

Tarefa: Seu objetivo é fornecer um resumo claro e completo, destacando os principais pontos, argumentos e informações relevantes do conteúdo. O resumo deve ser informativo e bem estruturado, com foco nas ideias principais. 

Restrições:
- O limite mínimo do resumo é de 350 palavras e o máximo é de 400.
- Mesmo que haja poucas informações, foque no principal conteúdo. 
- Caso não haja informações suficientes para gerar um resumo, você pode informar que o conteúdo é vazio ou indisponível.
- O resumo deve ser escrito em terceira pessoa, ou de forma impessoal.
- Retorne apenas o resumo, sem nenhuma mensagem adicional.
- O resumo deve estar em português brasileiro.

Texto:
{content}
"""
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"Erro ao acessar o Ollama: {e}"

def clear_page(page_content):
    return page_content.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\xa0", " ").strip()

def summarize_web_content(url, model):
    page_content = fetch_page_content(url)
    
    if "Erro" in page_content or "Conteúdo vazio" in page_content:
        return page_content  
    
    page_content = clear_page(page_content)
    
    summary = call_ollama(page_content, model=model)
    return summary


def main():
    url_dataset = '../datasets/relevantes/dataset_links.csv'
    dataset = pd.read_csv(url_dataset)
    dataset_output = pd.DataFrame(columns=['link', 'summary'])

    model_name = "llama3.1"  

    for url in tqdm(dataset['link']):
        try:
            response = summarize_web_content(url, model_name)
    
            df_aux = pd.DataFrame({'link': [url], 'summary': [response]})
            dataset_output = pd.concat([dataset_output, df_aux])
            dataset_output.to_csv('../datasets/relevantes/dataset_links_summary.csv', index=False)
        except Exception as e:
            print(f'Erro: {e}')


if __name__ == "__main__":
    main()