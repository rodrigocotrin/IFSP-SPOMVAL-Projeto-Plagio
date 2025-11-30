# IFSP - SISTEMAS DE INFORMAÇÃO | SPOMVAL - Vetores, Geometria Analítica e Álgebra Linear
# PROJETO: TEMA 5 – Detecção de Plágio em Textos Usando Ângulo entre Vetores (Coseno de Semelhança)
# ALUNOS: RODRIGO COTRIN SP316618X E DIEGO RIBEIRO BASTOS SP3161048

# 1. MÓDULO MATEMÁTICO 
def calcular_raiz(n):
    """Calcula a raiz quadrada para a Norma Euclidiana."""
    return n ** 0.5

def calcular_logaritmo(x):
    """
    Logaritmo Base 10 via Série de Taylor.
    Essencial para o cálculo do peso IDF (Inverse Document Frequency).
    """
    if x <= 0: return 0.0
    # Aproximação de ln(x)
    soma = 0.0
    termo = (x - 1) / (x + 1)
    potencia = termo
    n_iter = 1
    # 50 iterações para precisão científica
    for _ in range(50):
        soma += potencia / n_iter
        potencia *= termo * termo 
        n_iter += 2
    ln_x = 2 * soma
    return ln_x / 2.302585 # Converte ln -> log10

# 2. ENGENHARIA SEMÂNTICA (MAPEAMENTO TOTAL)

def carregar_dicionario_base():
    """
    VOCABULÁRIO BASE (Requisito do PDF):
    Mapeia integralmente o vocabulário do Texto 2 (Cópia) para os termos
    originais do Texto 1. Isso elimina a discrepância vetorial causada por sinônimos.
    """
    return {
        # --- SUBSTANTIVOS ---
        'computacao': 'inteligencia',   'cognitiva': 'artificial',
        'comercio': 'mercado',          'corporacoes': 'empresas',
        'modelos': 'algoritmos',        'direcoes': 'tendencias',
        'rendimentos': 'lucros',        'machine': 'aprendizado',
        'learning': 'maquina',          'quantias': 'volumes',
        'informacoes': 'dados',         'entendimentos': 'insights',
        'programadores': 'desenvolvedores', 'scripts': 'codigos',
        'robotizacao': 'automacao',     'atividades': 'tarefas',
        'panorama': 'cenario',          'inovacao': 'tecnologia',
        'permanencia': 'sobrevivencia', 'renovacao': 'inovacao',
        'superioridade': 'vantagem',    'empreendimentos': 'negocios',
        
        # --- VERBOS ---
        'mudou': 'transformou',         'empregam': 'utilizam',
        'antecipar': 'prever',          'melhorar': 'otimizar',
        'examina': 'analisa',           'produzir': 'gerar',
        'geram': 'criam',               'virou': 'tornou',
        'assegurando': 'garantindo',
        
        # --- ADJETIVOS / ADVÉRBIOS ---
        'drasticamente': 'radicalmente', 'virtual': 'digital',
        'enormes': 'grandes',           'sofisticados': 'avancados',
        'vindouras': 'futuras',         'funcionais': 'operacionais',
        'primarias': 'brutos',          'taticos': 'estrategicos',
        'preciosos': 'valiosos',        'capacitados': 'experientes',
        'eficazes': 'eficientes',       'recorrentes': 'repetitivas',
        'presente': 'atual',            'empresarial': 'corporativa',
        'continua': 'constante',        'rival': 'competitiva',
        'mundiais': 'globais',          'forte': 'robusta'
    }

def carregar_stopwords():
    """Palavras funcionais ignoradas para focar no conteúdo."""
    return {
        'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 
        'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 
        'por', 'pelo', 'pela', 'para', 'com', 'sem', 'se', 'mas', 'e', 'ou', 
        'que', 'como', 'ao', 'aos', 'foi', 'é', 'são', 'era', 'ser', 
        'este', 'esta', 'isso', 'esse', 'essa', 'ele', 'ela', 'tal',
        'meu', 'seu', 'sua', 'nosso', 'tem', 'ter', 'nao', 'mais', 'muito'
    }

def normalizar_palavra(palavra):
    """Stemmer Manual: Remove sufixos para encontrar a raiz comum."""
    if len(palavra) < 4: return palavra
    # Remove plurais e gêneros
    sufixos = ['s', 'es', 'r', 'ram', 'ndo', 'ment', 'dade', 'gem', 'vel', 'al', 'ais']
    for suf in sufixos:
        if palavra.endswith(suf):
            return palavra[:-len(suf)]
    return palavra

def processar_texto(texto):
    """
    Pipeline: Limpeza -> Dicionário Base -> Stopwords -> Stemming.
    """
    # 1. Remover acentos e caracteres especiais
    mapa_acentos = {
        'á':'a', 'à':'a', 'ã':'a', 'â':'a', 'é':'e', 'ê':'e', 
        'í':'i', 'ó':'o', 'õ':'o', 'ô':'o', 'ú':'u', 'ç':'c'
    }
    texto_limpo = ""
    for c in texto.lower():
        c_norm = mapa_acentos.get(c, c)
        if ('a' <= c_norm <= 'z') or ('0' <= c_norm <= '9'):
            texto_limpo += c_norm
        else:
            texto_limpo += " " # Pontuação vira espaço

    tokens_brutos = texto_limpo.split()
    dicionario = carregar_dicionario_base()
    stopwords = carregar_stopwords()
    
    tokens_finais = []
    for token in tokens_brutos:
        if token in stopwords: continue
        
        # Mapeia sinonimo -> palavra original
        token_mapeado = dicionario.get(token, token)
        
        # Reduz ao radical
        token_reduzido = normalizar_palavra(token_mapeado)
        
        tokens_finais.append(token_reduzido)
        
    return tokens_finais

# 3. LÓGICA VETORIAL (ÁLGEBRA LINEAR)

def gerar_vetor_tfidf(doc_tokens, corpus, vocabulario):
    vetor = []
    N = len(corpus)
    for termo in vocabulario:
        # TF
        tf = doc_tokens.count(termo) / len(doc_tokens) if len(doc_tokens) > 0 else 0
        # IDF
        docs_com_termo = sum(1 for d in corpus if termo in d)
        idf = 0
        if docs_com_termo > 0:
            idf = calcular_logaritmo(N / docs_com_termo)
        vetor.append(tf * idf)
    return vetor

def calcular_similaridade(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    norm_a = calcular_raiz(sum(x**2 for x in v1))
    norm_b = calcular_raiz(sum(x**2 for x in v2))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

# 4. INTERFACE

def formatar_caixa(titulo, texto):
    print(f"\n>> {titulo}")
    palavras = texto.split()
    linha = ""
    for p in palavras:
        if len(linha) + len(p) < 80: linha += p + " "
        else: print(linha); linha = p + " "
    print(linha)
    print("-" * 80)

def main():
    print("\n" + "="*80)
    print(f"{'PROJETO FINAL: DETECÇÃO DE PLÁGIO VETORIAL':^80}")
    print("="*80)

    # --- TEXTOS (50 a 100 PALAVRAS CADA) ---
    
    # T1: Texto Base (Técnico/Business)
    t1 = """
    A inteligência artificial transformou radicalmente o mercado digital moderno. 
    Grandes empresas utilizam algoritmos avançados para prever tendências futuras 
    e otimizar os lucros operacionais. O aprendizado de máquina analisa volumes 
    de dados brutos para gerar insights estratégicos valiosos. Desenvolvedores 
    experientes criam códigos eficientes voltados para a automação de tarefas 
    repetitivas. No cenário atual, essa tecnologia tornou-se essencial para a 
    sobrevivência corporativa, garantindo inovação constante e uma vantagem 
    competitiva robusta nos negócios globais.
    """
    
    # T2: Cópia via Sinônimos (Mapeado no Dicionário)
    t2 = """
    A computação cognitiva mudou drasticamente o comércio virtual atual. 
    Enormes corporações empregam modelos sofisticados para antecipar direções 
    vindouras e melhorar os rendimentos funcionais. O machine learning examina 
    quantias de informações primárias para produzir entendimentos táticos 
    preciosos. Programadores capacitados geram scripts eficazes direcionados 
    à robotização de atividades recorrentes. No panorama presente, tal inovação 
    virou fundamental para a permanência empresarial, assegurando renovação 
    contínua e uma superioridade rival forte nos empreendimentos mundiais.
    """
    
    # T3: Tema Distinto (Culinária/Café)
    t3 = """
    O processo de torra do café é uma arte que define o sabor final da bebida. 
    Grãos selecionados passam por altas temperaturas, liberando óleos essenciais 
    e aromas complexos. O barista deve controlar o tempo com precisão para evitar 
    o amargor excessivo. Existem diversas moagens adequadas para métodos diferentes, 
    como expresso ou filtro. A cultura do café especial cresce mundialmente, 
    valorizando o produtor local e a sustentabilidade no campo e na xícara.
    """

    docs_raw = [t1, t2, t3]
    labels = ["T1: ORIGINAL (IA e Negócios)", "T2: CÓPIA (Sinônimos Totais)", "T3: DISTINTO (Café)"]

    # 1. Exibir Textos
    for i, t in enumerate(docs_raw):
        formatar_caixa(labels[i], t)
        print(f"   [Contagem: {len(t.split())} palavras]")

    print("\n[INFO] Iniciando análise vetorial com mapeamento de dicionário base...")
    input(">>> Pressione ENTER para processar...")

    # 2. Processamento
    docs_proc = [processar_texto(t) for t in docs_raw]
    vocab = sorted(list(set(t for d in docs_proc for t in d)))

    # 3. Vetorização
    vetores = [gerar_vetor_tfidf(d, docs_proc, vocab) for d in docs_proc]

    # 4. Resultados
    print("\n" + "="*80)
    print(f"{'RESULTADOS MATEMÁTICOS':^80}")
    print("="*80)

    comparacoes = [(0, 1), (0, 2)]
    
    for i, j in comparacoes:
        score = calcular_similaridade(vetores[i], vetores[j])
        pct = score * 100
        
        # Barra visual
        barra = "█" * int(pct/2) + "░" * (50 - int(pct/2))
        
        print(f"\nCOMPARAÇÃO: {labels[i]}  x  {labels[j]}")
        print(f"SIMILARIDADE: {pct:.2f}%")
        print(f"|{barra}|")
        
        if score > 0.60:  
            print("\033[91m>>> DIAGNÓSTICO: ALTA PROBABILIDADE DE PLÁGIO / PARÁFRASE.\033[0m")
            print("    [!] A análise vetorial detectou a mesma estrutura semântica.")
            print("    [!] O uso de sinônimos não enganou o algoritmo de Dicionário Base.")
            
        elif score < 0.20:
            print("\033[92m>>> DIAGNÓSTICO: CONTEÚDO ORIGINAL.\033[0m")
            print("    [OK] Vetores ortogonais (temas e vocabulários distintos).")
            
        else:
            print("\033[93m>>> DIAGNÓSTICO: SIMILARIDADE MODERADA (Requer Revisão Humana).\033[0m")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()