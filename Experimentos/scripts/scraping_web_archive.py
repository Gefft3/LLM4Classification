import requests
import pandas as pd 
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

PATH_ERRORS = "../../datasets/relevantes/errors.txt"

def format_error_message(message, index=None, url=None):
    if index is not None and url is not None:
        return f"Index: {index}, URL: {url}, Error: {message}\n"
    return message + "\n\n"

def log_error(message, index=None, url=None):
    with open(PATH_ERRORS, 'a', encoding='utf-8') as f:
        f.write(format_error_message(message, index, url))

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

def create_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = '/usr/bin/google-chrome-stable'
    
    prefs = {"profile.managed_default_content_settings.javascript": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def get_html_content_with_selenium(driver, url, retries=3, wait_time=5, index=None):
    for attempt in range(retries):
        try:
            time.sleep(1)
            driver.get(url)
            return driver.page_source
        except Exception as e:
            if "no such window: target window already closed" in str(e):
                driver = create_driver()
            if attempt == retries - 1:
                log_error(f"[Selenium] Erro ao baixar {url}: {str(e)}", index, url)
            time.sleep(wait_time)
    return None

def get_latest_wayback_url(url, retries=3, wait_time=5, index=None):
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
                    log_error(f"[Wayback] Erro ao processar JSON: {response.text}", index, url)
            elif attempt == retries - 1:
                log_error(f"[Wayback] Erro ao acessar API: Status {response.status_code}", index, url)
        except requests.exceptions.RequestException as e:
            if "Max retries exceeded" in str(e) or "Connection refused" in str(e):
                log_error(f"[Wayback] Erro de conexão: {str(e)}", index, url)
                break
            if attempt == retries - 1:
                log_error(f"[Wayback] Erro na API: {str(e)}", index, url)
            time.sleep(wait_time)
    return None

def get_html_content(url, driver, index=None):
    wayback_url = get_latest_wayback_url(url, index=index)
    html_content = None
    from_web_archive = False
    
    if wayback_url:
        try:
            response = session.get(wayback_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            html_content = response.text
            from_web_archive = True
        except Exception as e:
            pass
    
    if not html_content:
        html_content = get_html_content_with_selenium(driver, url, index=index)
    
    return html_content, from_web_archive

def save_to_csv(path_output, url, html_content, from_web_archive):
    new_row = pd.DataFrame({
        'expanded_url': [url],
        'html_content': [html_content],
        'from WebArchive': [from_web_archive]
    })
    new_row.to_csv(
        path_output,
        mode='a',
        header=False,
        index=False,
        encoding='utf-8'
    )

def main():
    path_dataset = "../../datasets/relevantes/expanded_links.csv"
    path_output = "../../datasets/relevantes/scraping_content.csv"

    if os.path.exists(PATH_ERRORS):
        os.remove(PATH_ERRORS)

    try:
        dataset = pd.read_csv(path_dataset)
    except Exception as e:
        log_error(f"[Main] Erro ao ler dataset: {str(e)}")
        raise

    if not os.path.exists(path_output):
        pd.DataFrame(columns=['expanded_url', 'html_content', 'from WebArchive']).to_csv(path_output, index=False)

    driver = create_driver()

    try:
        for index, url in tqdm(enumerate(dataset['expanded_url']), desc="Processando URLs"):
            try:
                html_content, from_web_archive = get_html_content(url, driver, index=index)
                
                if html_content is None:
                    log_error(f"[Main] Falha em todos os métodos para: {url}", index, url)
                    continue
                
                save_to_csv(path_output, url, html_content, from_web_archive)
                
                time.sleep(1)
                
            except Exception as e:
                log_error(f"[Main] Erro crítico com {url}: {str(e)}", index, url)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()