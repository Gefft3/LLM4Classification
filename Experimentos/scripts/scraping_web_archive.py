import requests
import pandas as pd 
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        return response.json()
    except Exception as e:
        print(f"Error fetching snapshot: {e}")
        return None

def get_snapshot_html(url):
    snapshot_data = get_latest_snapshot(url)
    
    if not snapshot_data:
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
            print(f"Error fetching content: {e}")
            return None
    return None

if __name__ == "__main__":
    path_dataset = "../../datasets/relevantes/expanded_links.csv"
    path_output = "../../datasets/relevantes/scraping_content.csv"
    path_errors = "../../datasets/relevantes/scraping_errors.txt"

    dataset = pd.read_csv(path_dataset)
    output_dataset = pd.DataFrame(columns=['expanded_url', 'html_content'])

    for url in tqdm(dataset['expanded_url']):
        try:
            html_content = get_snapshot_html(url)
            
            df_aux = pd.DataFrame({
                'expanded_url': [url],
                'html_content': [html_content]
            })
            
            # Append mode para melhor performance
            df_aux.to_csv(
                path_output,
                mode='a',
                header=not pd.io.common.file_exists(path_output),
                index=False
            )
            
            # Delay para evitar sobrecarga
            time.sleep(1)
            
        except Exception as e:
            error_msg = f"Erro: {url} - {str(e)}\n"
            with open(path_errors, 'a', encoding='utf-8') as f:
                f.write(error_msg)
            continue