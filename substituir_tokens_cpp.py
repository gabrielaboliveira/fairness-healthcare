import os
import re

# Caminho base do projeto
base_dir = r"D:\UFRGS\Mestrado\fairnes_treatment\fairgbm"

# Arquivos com extensão alvo
extensoes_alvo = [".h", ".hpp", ".cpp"]

# Regex para substituição segura de tokens lógicos
substituicoes = [
    (r"\bnot\b", "!"),
    (r"\bor\b", "||"),
    (r"\band\b", "&&"),
]

def substituir_tokens_em_arquivo(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    conteudo_original = conteudo

    for regex, substituto in substituicoes:
        conteudo = re.sub(regex, substituto, conteudo)

    if conteudo != conteudo_original:
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        print(f"✅ Tokens substituídos em: {caminho_arquivo}")

def aplicar_substituicoes():
    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensoes_alvo):
                caminho_arquivo = os.path.join(root, file)
                substituir_tokens_em_arquivo(caminho_arquivo)

if __name__ == "__main__":
    aplicar_substituicoes()
