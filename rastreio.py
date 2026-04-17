import cv2
import csv
import os

# 1. Configurações Iniciais
pasta_frames = 'frames' # Substitua pelo nome da pasta onde estão suas im2, im3...
caminho_template = 'template.jpg'
num_total_frames = 300 # Ajuste conforme o tamanho do seu vídeo

# Carrega o template em tons de cinza, conforme exigido
template = cv2.imread(caminho_template, cv2.IMREAD_GRAYSCALE)

if template is None:
    raise ValueError("Erro: Não foi possível carregar o template. Verifique o caminho.")

# 2. Dicionário com os métodos exigidos no trabalho 
metodos = {
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
}

# Dicionário para armazenar as linhas de dados antes de salvar no CSV
# Estrutura: {'TM_CCOEFF': [['im2', min, max], ['im3', min, max]...], ...}
resultados_csv = {nome: [] for nome in metodos.keys()}

print("Iniciando o Template Matching...")

# 3. Loop principal iterando sobre os quadros (im2 até im300)
for i in range(2, num_total_frames + 1):
    nome_arquivo = f'im{i}.jpg'
    caminho_frame = os.path.join(pasta_frames, nome_arquivo)
    
    # Carrega a imagem atual em tons de cinza
    frame = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
    
    if frame is None:
        print(f"Aviso: Quadro {nome_arquivo} não encontrado. Pulando...")
        continue

    # 4. Aplica cvMatchTemplate para cada método
    for nome_metodo, flag_metodo in metodos.items():
        
        # A operação matemática de convolução/casamento
        res = cv2.matchTemplate(frame, template, flag_metodo)
        
        # Encontra os valores globais mínimos e máximos na matriz de resultado
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Formata os valores para manter a consistência e armazena
        resultados_csv[nome_metodo].append([f'im{i}', min_val, max_val])

# 5. Geração dos arquivos CSV 
print("Exportando dados para CSV...")

for nome_metodo, dados in resultados_csv.items():
    nome_arquivo_csv = f'{nome_metodo}.csv'
    
    with open(nome_arquivo_csv, mode='w', newline='') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(['Quadro (imagem)', 'min_val', 'max_val']) 
        escritor.writerows(dados)

print("Processamento concluído com sucesso! Os arquivos CSV foram gerados.")