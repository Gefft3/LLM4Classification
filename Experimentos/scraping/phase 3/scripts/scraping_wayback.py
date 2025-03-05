import requests
import pandas as pd
from tqdm import tqdm
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG_DIR = "../logs"
SUCCESS_PATH = "../data/retrieved_html_wayback_success.csv"
FAIL_PATH = "../data/retrieved_html_wayback_fail.csv"

os.makedirs(LOG_DIR, exist_ok=True)
ERROR_LOG_PATH = os.path.join(LOG_DIR, "errors.txt")

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)
session.mount('http://', adapter)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def log_error(message, index=None, url=None):
    """Registra erros em um arquivo de log."""
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Index: {index}, URL: {url}, Error: {message}\n")

def get_latest_wayback_url(url, retries=3, wait_time=5, index=None):
    """Obtém a versão mais recente arquivada de uma URL na Wayback Machine."""
    wayback_api = f"https://web.archive.org/cdx/search/cdx?url={url}&output=json&fl=timestamp,original&collapse=digest&limit=1&filter=statuscode:200&sort=reverse"
    
    for attempt in range(retries):
        try:
            response = session.get(wayback_api, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if len(data) > 1:
                        timestamp = data[1][0]
                        latest_url = f"https://web.archive.org/web/{timestamp}/{url}"
                        return latest_url
                except ValueError:
                    log_error("Erro ao processar JSON da Wayback Machine", index, url)
            elif attempt == retries - 1:
                log_error(f"Erro na API da Wayback Machine: Status {response.status_code}", index, url)
        except requests.exceptions.RequestException as e:
            log_error(f"Erro de conexão com Wayback Machine: {str(e)}", index, url)
            time.sleep(wait_time)
    return None

def get_html_content(url, index=None):
    """Obtém o conteúdo HTML da versão arquivada da URL."""
    wayback_url = get_latest_wayback_url(url, index=index)
    if wayback_url:
        try:
            response = session.get(wayback_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            log_error(f"Erro ao baixar HTML da Wayback Machine: {str(e)}", index, url)
    return None

def save_to_csv(path, url, html_content, retrieved_sucess):
    """Salva os resultados no CSV correspondente."""
    if retrieved_sucess:
        new_row = pd.DataFrame({'expanded_url': [url], 'html_content': [html_content]})
        new_row.to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8')
    else:
        new_row = pd.DataFrame({'expanded_url': [url]})
        new_row.to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8')

def main():
    path_dataset = "../../phase 2 /data/unique_expanded_urls.csv"
    
    if os.path.exists(ERROR_LOG_PATH):
        os.remove(ERROR_LOG_PATH)
    
    try:
        dataset = pd.read_csv(path_dataset)
    except Exception as e:
        log_error(f"Erro ao ler dataset: {str(e)}")
        return
    
    with tqdm(total=len(dataset), desc="Processando URLs") as pbar:
        for index, url in enumerate(dataset['expanded_url']):
            try:
                html_content = get_html_content(url, index=index)
                
                if html_content:
                    save_to_csv(SUCCESS_PATH, url, html_content, True)
                else:
                    save_to_csv(FAIL_PATH, url, "", False)
                    log_error("Erro ao obter conteúdo HTML, sem contéudo", index, url)
            
            except Exception as e:
                log_error(f"Erro crítico ao processar URL: {str(e)}", index, url)
            
            time.sleep(1)
            pbar.update(1)

if __name__ == "__main__":
    main()
