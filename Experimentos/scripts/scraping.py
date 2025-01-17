import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import tqdm
import os

def create_driver():
    """Cria e retorna uma instância do driver configurada para desativar JavaScript."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Executa sem interface gráfica
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Configuração para desabilitar o JavaScript
    prefs = {"profile.managed_default_content_settings.javascript": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def fetch_html_with_retries(driver, url, retries=3, wait_time=5):
    """Tenta buscar o HTML de uma URL com repetição em caso de falha."""
    for attempt in range(retries):
        try:
            driver.set_page_load_timeout(60)  # Define timeout de carregamento
            driver.get(url)
            return driver.page_source
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait_time)  # Aguarda antes de tentar novamente
            else:
                raise e

def main():
    dataset_path = '../../datasets/irrelevantes/dataset_links.csv'
    dataset = pd.read_csv(dataset_path)

    os.makedirs('./scraping_content', exist_ok=True)

    driver = create_driver()

    try:
        for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
            url = row['link']
            try:
                html_content = fetch_html_with_retries(driver, url)
                with open(f'./scraping_content_irrelevants/{index}.txt', 'w') as f:
                    f.write(html_content)
            except Exception as e:
                with open('./scraping_content_irrelevants/erros.txt', 'a') as f:
                    f.write('------------------------------\n')
                    f.write(f'index {index} - url: {url}\n')
                    f.write(f'erro: {e}\n')
    finally:
        driver.quit()

if __name__ == '__main__':
    main()
