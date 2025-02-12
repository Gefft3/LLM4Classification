import requests
import pandas as pd 
from tqdm import tqdm

def get_latest_snapshot(url):
    wayback_api = f"https://archive.org/wayback/available?url={url}"
    response = requests.get(wayback_api).json()
    
    if 'archived_snapshots' in response and 'closest' in response['archived_snapshots']:
        snapshot_url = response['archived_snapshots']['closest']['url']
        return snapshot_url
    else:
        return None

def get_snapshot_html(url):
    snapshot_url = get_latest_snapshot(url)
    if snapshot_url:
        response = requests.get(snapshot_url)
        return response.text
    return None

if __name__ == "__main__":
    
    path_dataset = "../../datasets/relevantes/expanded_links.csv"
    path_output = "../../datasets/relevantes/scraping_content.csv"
    path_errors = "../../datasets/relevantes/scraping_errors"

    dataset = pd.read_csv(path_dataset)

    output_dataset = pd.DataFrame(columns=['expanded_url', 'html_content'])

    for url in tqdm(dataset['expanded_url']):
        try: 
            html_content = get_snapshot_html(url)
            
            df_aux = pd.DataFrame({'expanded_url': [url], 'html_content': [html_content]})
            output_dataset = pd.concat([output_dataset, df_aux], ignore_index=True)
            output_dataset.to_csv(path_output, index=False)
        except Exception as e:
            with open(path_errors, 'a') as f:
                f.write(f"Erro: {url} - {e}\n")

