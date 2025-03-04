import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import time

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

def expand_short_url(driver, short_url, retries=3, wait_time=5):
    for attempt in range(retries):
        try:
            time.sleep(1)
            driver.get(short_url)
            return driver.current_url
        except Exception as e:
            if "no such window: target window already closed" in str(e):
                driver = create_driver()
            if attempt < retries - 1:
                time.sleep(wait_time)  
                continue
            else:
                raise e

def main():
    links_path = '../../../datasets/relevantes/dataset_links.csv'
    output_csv_path = '../result/expanded_url_selenium.csv'
    error_log_path = '../logs/error.txt'

    dataset_links = pd.read_csv(links_path)
    expanded_links = []

    driver = create_driver()

    try:
        for i, row in tqdm(dataset_links.iterrows(), total=dataset_links.shape[0]):
            short_url = row['links']
            try:
                expanded_url = expand_short_url(driver, short_url)
                expanded_links.append({'index': i, 'short_url': short_url, 'expanded_url': expanded_url})
                expanded_links_df = pd.DataFrame(expanded_links)
                expanded_links_df.to_csv(output_csv_path, index=False)
            except Exception as e:
                if "no such window: target window already closed" in str(e):
                    driver = create_driver()
                with open(error_log_path, 'a') as error_file:
                    error_file.write(f"Error: {e}\n")
                    error_file.write(f"Index: {i}\n")
                    error_file.write(f"------------------------\n")
    finally:
        driver.quit()  

if __name__ == '__main__':
    main()