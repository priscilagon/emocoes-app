import pandas as pd
import numpy as np
import os

def pre_processar_dados(caminho_entrada, caminho_saida):
    """
    Realiza o pré-processamento e balanceamento do dataset de emoções.
    Combina Undersampling das classes majoritárias (como 'neutro') e 
    Oversampling das classes minoritárias para tentar igualar as frequências.
    """
    print(f"Carregando dados de: {caminho_entrada}")
    df = pd.read_csv(caminho_entrada, sep=';')
    df = df.dropna(subset=['texto'])
    
    emocoes = df.columns.drop('texto')
    frequencias = df[emocoes].sum()
    
    print("\nFrequência de cada emoção antes do balanceamento:")
    print(frequencias.sort_values(ascending=False))
    
    # Alvo para a quantidade de exemplos de cada emoção (em torno de 5000)
    alvo = 5000 
    
    # Para lidar com multi-labels sem remover emoções raras por acidente, 
    # identificamos qual é a emoção mais rara presente em cada texto.
    freq_dict = frequencias.to_dict()
    
    def emocao_mais_rara_freq(row):
        emocoes_presentes = [e for e in emocoes if row[e] == 1]
        if not emocoes_presentes:
            return float('inf')
        return min([freq_dict[e] for e in emocoes_presentes])
        
    print("\nCalculando raridade das amostras (isso pode levar alguns segundos)...")
    df['freq_emocao_rara'] = df.apply(emocao_mais_rara_freq, axis=1)
    
    dados_balanceados = []
    
    # 1. Undersampling (Redução) das amostras onde TODAS as emoções são muito frequentes
    df_majoritario = df[df['freq_emocao_rara'] > alvo]
    df_minoritario = df[df['freq_emocao_rara'] <= alvo]
    
    # Reduzindo o dataframe majoritário
    # Como 'neutro' tem 31k, vamos pegar apenas uma fração para baixar para perto do alvo
    df_majoritario_under = df_majoritario.sample(n=alvo, random_state=42, replace=False)
    dados_balanceados.append(df_majoritario_under)
    
    # 2. Oversampling (Superamostragem) das emoções raras
    # Vamos agrupar as amostras pela emoção mais rara que contêm e duplicá-las
    for emocao in emocoes:
        count = frequencias[emocao]
        if count <= alvo:
            # Pegar amostras onde essa é a emoção mais rara
            amostras = df_minoritario[df_minoritario['freq_emocao_rara'] == freq_dict[emocao]]
            qtd_amostras = len(amostras)
            
            if qtd_amostras > 0:
                fator = int(alvo / count)
                # O fator não deve ser menor que 1
                fator = max(1, fator)
                
                # Adiciona as cópias
                dados_balanceados.append(pd.concat([amostras] * fator, ignore_index=True))
                
    # Concatenar todos os dados gerados
    df_balanceado = pd.concat(dados_balanceados, ignore_index=True)
    
    # Remover a coluna auxiliar
    df_balanceado = df_balanceado.drop(columns=['freq_emocao_rara'])
    
    # Embaralhar os dados
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    frequencias_pos = df_balanceado[emocoes].sum()
    print("\nFrequência de cada emoção após o balanceamento:")
    print(frequencias_pos.sort_values(ascending=False))
    
    print(f"\nTamanho original do dataset: {len(df)} linhas")
    print(f"Tamanho após balanceamento: {len(df_balanceado)} linhas")
    
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_balanceado.to_csv(caminho_saida, sep=';', index=False)
    print(f"\nDados balanceados salvos com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    caminho_entrada = "data/AnaliseDeEmocoes_PT-br.csv"
    caminho_saida = "data/AnaliseDeEmocoes_PT-br_balanceado.csv"
    pre_processar_dados(caminho_entrada, caminho_saida)
