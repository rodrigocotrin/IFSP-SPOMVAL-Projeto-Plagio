# Detecção de Plágio em Textos Usando Ângulo entre Vetores (Cosine Similarity)

## Descrição do Projeto
Este projeto implementa um sistema de detecção de plágio baseado em Álgebra Linear e Modelagem Vetorial de Documentos. A solução utiliza TF-IDF, transformação semântica via dicionário base, vetorização em espaço de alta dimensionalidade e cálculo de similaridade pelo coseno do ângulo entre vetores.

O sistema identifica tanto cópia direta quanto paráfrase baseada em sinônimos, aplicando técnicas de Processamento de Linguagem Natural (PLN) e Geometria Analítica.

Trabalho desenvolvido no contexto do curso de Bacharelado em Sistemas de Informação do Instituto Federal de São Paulo (IFSP), como parte dos requisitos da disciplina SPOMVAL – Vetores, Geometria Analítica e Álgebra Linear, sob orientação da professora Josceli Maria Tenoria.

---

## Funcionalidades Principais
- Processamento textual com limpeza, normalização e remoção de stopwords  
- Mapeamento semântico via dicionário de sinônimos (engenharia semântica)  
- Vetorização usando TF-IDF com cálculo de logaritmos via série de Taylor  
- Similaridade entre documentos por Cosseno de Semelhança  
- Relatórios com palavras mais relevantes (Top 5 por TF-IDF)  
- Diagnóstico automático de plágio ou originalidade  
- Casos de teste estruturados com entrada e saída esperada  

---

## Estrutura do Código
- `processar_texto()`: pipeline completo de preparação textual  
- `gerar_vetor_tfidf()`: construção do vetor TF-IDF  
- `calcular_similaridade()`: cálculo da métrica do cosseno  
- `carregar_dicionario_base()`: módulo de equivalência semântica  
- `main()`: execução integral do sistema via CLI  

---

## Casos de Teste
O projeto inclui dois casos de teste principais:

### 1. Documento original × cópia por sinônimos  
**Resultado esperado:** similaridade alta (> 60%) e diagnóstico de plágio.

### 2. Documento original × texto de tema distinto  
**Resultado esperado:** similaridade próxima de zero e diagnóstico de originalidade.

---

## Requisitos
- Python 3.8 ou superior  
- Nenhuma biblioteca externa é necessária (implementação totalmente em Python padrão)
