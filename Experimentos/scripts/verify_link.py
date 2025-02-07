import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

def expand_short_url(short_url):
    options = Options()
    options.add_argument('--headless') 
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.binary_location = '/usr/bin/google-chrome-stable' 
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(short_url)
        expanded_url = driver.current_url
    except Exception:
        expanded_url = None
    finally:
        driver.quit()
    
    return expanded_url

def main():
    links_path = './datasets/relevantes/dataset_links.csv'
    output_csv_path = './datasets/relevantes/expanded_links.csv'
    error_log_path = './datasets/relevantes/error_log.txt'

    dataset_links = pd.read_csv(links_path)
    expanded_links = []

    for i, row in tqdm(dataset_links.iterrows(), total=dataset_links.shape[0]):
        short_url = row['links']
        try:
            expanded_url = expand_short_url(short_url)
            expanded_links.append({'index': i, 'short_url': short_url, 'expanded_url': expanded_url})
            expanded_links_df = pd.DataFrame(expanded_links)
            expanded_links_df.to_csv(output_csv_path, index=False)
        except Exception as e:
            with open(error_log_path, 'a') as error_file:
                error_file.write(f"Error: {e}\n")
                error_file.write(f"{i}\n")
                error_file.write(f"------------------------\n")
            
if __name__ == '__main__':
    main()