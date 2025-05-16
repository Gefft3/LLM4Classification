import pandas as pd
import re
from collections import defaultdict
import sys

def load_data(text_sample_path, classification_results_path):
    df_text_sample = pd.read_csv(text_sample_path)
    with open(classification_results_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return df_text_sample, lines

def parse_classification_results(lines):
    pattern = re.compile(r"\[(\d+)\] Chunk \d+ \| Relevant: (True|False)")
    resultados = defaultdict(list)
    for line in lines:
        match = pattern.search(line)
        if match:
            indice, valor = match.groups()
            resultados[int(indice)].append(valor == "True")
    return {indice: any(valores) for indice, valores in resultados.items()}

def filter_texts(df_text_sample, relevancy_map):
    filtered_texts = []
    for i, row in df_text_sample.iterrows():
        if relevancy_map.get(i, False):
            filtered_texts.append(df_text_sample.iloc[i])
    filtered_df = pd.DataFrame(filtered_texts).reset_index(drop=True)
    return filtered_df

def save_filtered_texts(filtered_df, output_path):
    filtered_df.to_csv(output_path, index=False)

def main(text_sample_path, classification_results_path, output_path):
    df_text_sample, lines = load_data(text_sample_path, classification_results_path)
    relevancy_map = parse_classification_results(lines)
    filtered_df = filter_texts(df_text_sample, relevancy_map)
    save_filtered_texts(filtered_df, output_path)
    print(f"Filtered texts saved to {output_path}")

if __name__ == "__main__":
    
    text_sample_path = sys.argv[1]
    classification_results_path = sys.argv[2]
    output_path = sys.argv[3]

    main(text_sample_path, classification_results_path, output_path)

   


