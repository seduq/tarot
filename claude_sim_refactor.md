# Análise de Memória Melhorada - RIS-MCTS

## Visão Geral

O `ris_mcts_example.py` foi aprimorado para fornecer análise de memória muito mais detalhada e precisa do algoritmo RIS-MCTS. Esta versão utiliza múltiplas ferramentas e técnicas de medição para obter insights profundos sobre o comportamento de memória.

## Melhorias Implementadas

### 1. Múltiplas Ferramentas de Medição

#### Tracemalloc (Python Built-in)
- **Uso**: Medição precisa de alocações Python
- **Métricas**: Pico de memória, memória atual durante execução
- **Precisão**: Alta para objetos Python

#### PSUtil (System-level)
- **Uso**: Medição de memória do processo no sistema
- **Métricas**: RSS (Resident Set Size), memória total do processo
- **Precisão**: Inclui memória de bibliotecas C/C++

#### Garbage Collection Stats
- **Uso**: Monitoramento de coletas de lixo
- **Métricas**: Número de coletas por geração (Gen0, Gen1, Gen2)
- **Insight**: Indica pressão de memória e fragmentação

### 2. Medições Temporais Precisas

```python
# ANTES de cada chamada RIS-MCTS
gc.collect()  # Força limpeza para baseline limpo
memory_before = process.memory_info().rss / 1024 / 1024

# DURANTE a execução
tracemalloc.start()
# ... execução do RIS-MCTS ...
current_mem, peak_mem = tracemalloc.get_traced_memory()

# APÓS a execução
memory_after = process.memory_info().rss / 1024 / 1024
memory_growth = memory_after - memory_before
```

### 3. Estatísticas Detalhadas Coletadas

#### Por Jogo
- Baseline de memória
- Memória final
- Crescimento total de memória
- Estatísticas por chamada do RIS-MCTS

#### Por Chamada RIS-MCTS
- Memória antes da chamada
- Memória após a chamada
- Crescimento de memória
- Pico de memória (Tracemalloc)
- Tempo de execução
- Iterações tentadas

### 4. Análise Estatística Avançada

#### Medidas de Tendência Central
- Média, mediana, mínimo, máximo
- Desvio padrão para dispersão
- Análise de quartis

#### Detecção de Outliers
- Método IQR (Interquartile Range)
- Identificação de comportamentos anômalos
- Percentual de outliers

#### Análise de Correlação
- Correlação entre tempo de execução e uso de memória
- Identificação de padrões de crescimento

#### Análise de Tendência Temporal
- Comparação entre primeiro e último quartil
- Detecção de vazamentos de memória
- Padrões de degradação

## Saída de Exemplo

### Console (Verbose)
```
DETAILED MEMORY ANALYSIS:
Baseline memory: 45.23 MB
Final memory: 78.45 MB
Total memory growth: 33.22 MB
Tracemalloc peak: 156.78 MB
PSUtil peak: 89.12 MB
Memory growth from RIS-MCTS: 28.65 MB
GC collections: {'gen0': 45, 'gen1': 3, 'gen2': 1}
Number of memory measurements: 23
Average memory growth per RIS-MCTS call: 1.25 MB
Maximum memory growth per RIS-MCTS call: 4.56 MB
```

### Arquivo CSV Expandido
```csv
Game,Win,Legal Actions,Chance Outcomes,Game Length,Invalid Actions,TraceMallocPeakMB,PSUtilPeakMB,MemoryGrowthMB,TotalMemoryGrowthMB,ElapsedTimeMCTS,GC_Gen0,GC_Gen1,GC_Gen2,RIS_Position_0,RIS_Position_1,...
```

### Arquivo TXT Detalhado
```
DETAILED MEMORY ANALYSIS:
Average Tracemalloc peak per game: 145.67 MB
Average PSUtil peak per game: 78.23 MB
Average memory growth from RIS-MCTS per game: 12.45 MB
Average total memory growth per game: 15.67 MB
Total memory measurements taken: 2340
Average GC Gen0 collections per game: 23.4
Average GC Gen1 collections per game: 2.1
Average GC Gen2 collections per game: 0.3
```

## Interpretação dos Resultados

### Indicadores de Performance
- **Tracemalloc Peak < 200MB**: Bom uso de memória Python
- **Memory Growth < 5MB per call**: Comportamento normal
- **GC Gen2 collections < 5 per game**: Baixa fragmentação

### Sinais de Alerta
- **Crescimento consistente de memória**: Possível vazamento
- **Picos de memória > 500MB**: Uso excessivo
- **Muitas coletas Gen2**: Fragmentação alta
- **Correlação alta tempo-memória**: Ineficiência algorítmica

### Otimizações Sugeridas
1. **Memória crescente**: Implementar cache com limite de tamanho
2. **Picos altos**: Reduzir árvore de busca ou usar poda mais agressiva
3. **Muitas coletas GC**: Reutilizar objetos, evitar alocações desnecessárias

## Uso Prático

### Análise de Performance
```python
# Executar com análise detalhada
python ris_mcts_example.py

# Resultados salvos em:
# - ris_mcts_results_N_M.csv (dados tabulares)
# - ris_mcts_results_N_M_X.txt (análise detalhada)
```

### Monitoramento Contínuo
- Use os dados CSV para gráficos de tendência
- Monitore médias móveis de uso de memória
- Estabeleça alertas para outliers

### Benchmarking
- Compare diferentes configurações de iterações
- Avalie trade-off entre qualidade e recursos
- Otimize para ambientes com restrições de memória

## Limitações e Considerações

### Precisão
- Tracemalloc pode ter overhead (~10-20%)
- PSUtil inclui memória de bibliotecas externas
- GC forçado pode afetar performance natural

### Interpretação
- Crescimento de memória nem sempre indica vazamento
- Alguns picos são esperados em algoritmos de busca
- Correlações podem ser espúrias

### Ambiente
- Resultados variam entre sistemas operacionais
- Memória disponível afeta comportamento do GC
- Outros processos podem influenciar medições

## Conclusão

Esta versão aprimorada fornece insights profundos sobre o comportamento de memória do RIS-MCTS, permitindo:
- Identificação precoce de problemas de memória
- Otimização baseada em dados
- Monitoramento de performance em produção
- Análise comparativa de diferentes configurações
