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

def log_error(message):
    with open(PATH_ERRORS, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

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

def get_html_content_with_selenium(driver, url, retries=3, wait_time=5):
    for attempt in range(retries):
        try:
            time.sleep(1)
            driver.get(url)
            return driver.page_source
        except Exception as e:
            if "no such window: target window already closed" in str(e):
                driver = create_driver()
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                log_error(f"[Selenium] Erro ao baixar {url}: {str(e)}")
                return None

def get_latest_wayback_url(url):
    wayback_api = f"http://web.archive.org/cdx/search/cdx?url={url}&output=json&fl=timestamp,original&collapse=digest&limit=1&filter=statuscode:200&sort=reverse"
    response = requests.get(wayback_api)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 1:
            timestamp = data[1][0]
            latest_url = f"http://web.archive.org/web/{timestamp}/{url}"
            return latest_url
    return None

def get_snapshot_html(url):
    snapshot_url = get_latest_wayback_url(url)
    if not snapshot_url:
        return None
    try:
        response = session.get(snapshot_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e:
        log_error(f"[HTML] Erro ao baixar {snapshot_url}: {str(e)}")
        return None

def get_html_content(url, driver):
    html_content = get_snapshot_html(url)
    if html_content:
        return html_content, True
    html_content = get_html_content_with_selenium(driver, url)
    if html_content:
        return html_content, False
    return None, None

if __name__ == "__main__":
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

    for url in tqdm(dataset['expanded_url'], desc="Processando URLs"):
        try:
            html_content, from_web_archive = get_html_content(url, driver)
            if html_content:
                pd.DataFrame({
                    'expanded_url': [url],
                    'html_content': [html_content],
                    'from WebArchive': [from_web_archive]
                }).to_csv(
                    path_output,
                    mode='a',
                    header=False,
                    index=False
                )
            else:
                log_error(f"[Main] Erro geral com {url}: Não foi possível obter o conteúdo HTML")
            time.sleep(1)
        except Exception as e:
            log_error(f"[Main] Erro geral com {url}: {str(e)}")
    
    driver.quit()