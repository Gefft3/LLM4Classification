import requests
import pandas as pd 
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

# Definir o path do arquivo de erros (no mesmo diretório do dataset de saída)
PATH_ERRORS = "../../datasets/relevantes/errors.txt"

# Função para registrar mensagens de erro no arquivo de log
def log_error(message):
    with open(PATH_ERRORS, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

# Configurar retentativas com backoff exponencial
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_latest_snapshot(url):
    wayback_api = f"https://archive.org/wayback/available?url={url}"
    try:
        response = session.get(
            wayback_api,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        # Verifica se o snapshot foi encontrado
        if not result or 'archived_snapshots' not in result or not result['archived_snapshots'].get('closest'):
            log_error(f"[get_latest_snapshot] Nenhuma snapshot encontrada para: {url}. Dados retornados: {result}")
            return None
        return result
    except Exception as e:
        log_error(f"[get_latest_snapshot] Erro ao buscar snapshot para {url}: {str(e)}")
        return None

def get_snapshot_html(url):
    snapshot_data = get_latest_snapshot(url)
    
    if snapshot_data is None:
        log_error(f"[get_snapshot_html] Retorno None em get_latest_snapshot para a URL: {url}")
        return None
        
    if 'archived_snapshots' in snapshot_data and snapshot_data['archived_snapshots'].get('closest'):
        snapshot_url = snapshot_data['archived_snapshots']['closest']['url']
        try:
            response = session.get(
                snapshot_url,
                headers=HEADERS,
                timeout=15
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            log_error(f"[get_snapshot_html] Erro ao buscar conteúdo do snapshot para {url} (snapshot URL: {snapshot_url}): {str(e)}")
            return None
    else:
        log_error(f"[get_snapshot_html] Estrutura inesperada dos dados para {url}: {snapshot_data}")
        return None

if __name__ == "__main__":
    path_dataset = "../../datasets/relevantes/expanded_links.csv"
    path_output = "../../datasets/relevantes/scraping_content.csv"

    # Limpa o arquivo de erros, se existir, para evitar duplicidade de logs
    if os.path.exists(PATH_ERRORS):
        os.remove(PATH_ERRORS)

    try:
        dataset = pd.read_csv(path_dataset)
    except Exception as e:
        log_error(f"[Main] Erro ao ler o dataset ({path_dataset}): {str(e)}")
        raise

    # Se o arquivo de output não existir, cria-o com header
    if not os.path.exists(path_output):
        pd.DataFrame(columns=['expanded_url', 'html_content']).to_csv(path_output, index=False)

    for url in tqdm(dataset['expanded_url'], desc="Processando URLs"):
        try:
            html_content = get_snapshot_html(url)
            # Se o conteúdo for None, registra o ocorrido
            if html_content is None:
                log_error(f"[Main] html_content retornou None para a URL: {url}")
            
            df_aux = pd.DataFrame({
                'expanded_url': [url],
                'html_content': [html_content]
            })
            
            # Escreve os dados em modo append; o header só é escrito se o arquivo for novo
            df_aux.to_csv(
                path_output,
                mode='a',
                header=False,
                index=False
            )
            
            # Delay para evitar sobrecarga no servidor
            time.sleep(1)
            
        except Exception as e:
            log_error(f"[Main] Erro ao processar a URL {url}: {str(e)}")
            continue
