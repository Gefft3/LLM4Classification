import os
import time
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Diretórios e arquivos
LOG_DIR = "../logs"
SUCCESS_PATH = "../data/retrieved_html_selenium_success.csv"
FAIL_PATH = "../data/retrieved_html_selenium_fail.csv"

os.makedirs(LOG_DIR, exist_ok=True)
ERROR_LOG_PATH = os.path.join(LOG_DIR, "errors.txt")

def create_driver():
    """Cria e retorna uma instância do driver configurada para desativar JavaScript."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Desativa JavaScript para evitar bloqueios
    prefs = {"profile.managed_default_content_settings.javascript": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def log_error(message, index=None, url=None):
    """Registra erros em um arquivo de log."""
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Index: {index}, URL: {url}, Error: {message}\n")

def fetch_html_with_retries(driver, url, retries=3, wait_time=5):
    """Tenta buscar o HTML de uma URL com repetição em caso de falha."""
    for attempt in range(retries):
        try:
            driver.set_page_load_timeout(60)
            driver.get(url)
            return driver.page_source
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                raise e

def save_to_csv(path, url, html_content, retrieved_success):
    """Salva os resultados no CSV correspondente."""
    if retrieved_success:
        new_row = pd.DataFrame({'expanded_url': [url], 'html_content': [html_content]})
    else:
        new_row = pd.DataFrame({'expanded_url': [url]})

    new_row.to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8')

def main():
    dataset_path = '../../phase 3/data/retrieved_html_wayback_fail.csv'

    if os.path.exists(ERROR_LOG_PATH):
        os.remove(ERROR_LOG_PATH)

    try:
        dataset = pd.read_csv(dataset_path)
    except Exception as e:
        log_error(f"Erro ao ler dataset: {str(e)}")
        return

    driver = create_driver()

    try:
        with tqdm(total=len(dataset), desc="Processando URLs") as pbar:
            for index, row in dataset.iterrows():
                url = row['expanded_url']
                try:
                    html_content = fetch_html_with_retries(driver, url)

                    if html_content:
                        save_to_csv(SUCCESS_PATH, url, html_content, True)
                    else:
                        save_to_csv(FAIL_PATH, url, "", False)
                        log_error("Erro ao obter conteúdo HTML, sem conteúdo", index, url)

                except Exception as e:
                    save_to_csv(FAIL_PATH, url, "", False)
                    log_error(f"Erro crítico ao processar URL: {str(e)}", index, url)

                time.sleep(1)
                pbar.update(1)
    finally:
        driver.quit()

if __name__ == '__main__':
    main()
