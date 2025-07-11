# SolarChem QA-RAG Benchmark & Dataset

This repository contains the benchmark and dataset for [SolarQA](https://github.com/oeg-upm/solar-qa) application. The SolarChem QA-RAG Benchmark is designed specifically for factual question answering benchmark based on solar chemistry academic papers. We intent to build an open-source benchmark & dataset for testing the performance of QA system when it comes to narrow-domain and in-depth questions. 

The SolarChem QA-RAG Benchmark addresses the critical need for domain-specific evaluation tools in scientific literature processing. It aims to:

- Build an open-source benchmark for testing QA system performance on narrow-domain, in-depth questions
- Provide rigorous assessment capabilities for LLM-driven QA systems processing scientific content
- Enable comprehensive evaluation of retrieval-augmented generation (RAG) strategies in solar chemistry

## SolarChemQA Annotation Pipeline

The SolarChem Annotation Pipeline is the evidence extraction component that processes solar chemistry research papers to identify and extract relevant experimental information. This pipeline serves as the foundation for creating high-quality training and evaluation data. This pipeline connects with the broader SolarQA ecosystem, providing the annotated evidence base necessary for effective question-answering system development and evaluation. The annotation pipeline extracts evidence for seven critical experimental parameters:

- Catalyst: A substance that increases the rate of a reaction without modifying the overall standard Gibbs energy change in the reaction
- Co-Catalyst: A substance or agent that brings about catalysis in conjunction with one or more others
- Light Source: A light source is an optical subsystem that provides light for use in a distant area using a delivery system (e.g., fiber optics).
- Lamp: A devide that generates heat, light, or any other form of radiation
- Reactor Type: The condition of the reactor devide that performs the process of a photocatalysis reaction
- Reaction Medium: The medium used for the reaction of the experiment execution
- Operation Mode: The condition whether the operation is perfomed in batch-mode or continuous-mode

Details for all seven experimental parameters is given at [SolarChem Ontology](https://solarchem.github.io/solarchem-ontology/docs/release/1.1.0/core/index-en.html)

## SolarChem Generation Pipeline

The SolarChem Generation Pipeline represents the complete end-to-end system that takes extracted paper data and generates all seven experimental parameters from the paper. The pipeline first enables the SolarChem Annotation Pipeline to generate all evidence (extract quota from the paper) with corresponding experimental parameters. And then the pipeline incorporates large language models to infer all seven experimental parameters based on the evidence and the generated possible answers.

Word Flow:

![wordflow~](/img/agent_anno.png)

## SolarChemQA Dataset

SolarChemQA provides a novel question answering dataset curated from solar chemistry literature, designed to rigorously assess the capabilities of LLM-driven QA systems in processing domain-specific scientific content. 

### SolarChemQA-Dataset-json

SolarChemQA provides a novel question answering dataset curated from solar chemistry literature, designed to rigorously assess the capabilities of LLM-driven QA systems in processing domain-specific scientific content. The dataset includes 574 domain expert annotations across 82 solar chemistry research papers, structured around seven critical experimental parameters. Together with 289 annotated source sentences with positive and negative examples across 21 papers.

- Query Sub-Dataset: Standardized queries for the seven experimental parameter categories
- Paper Extraction Sub-Dataset: Structured content extracted from research papers
- Annotation Sub-Dataset: 574 expert-validated annotations and 289 supporting textual sentences

### SolarChemQA-Dataset-Updated

The SolarChemQA-Dataset-Updated directory contains the dataset under further developing, with the aim to expand the scope of the dataset.

Dataset Structure:
```project
SolarChemQA-Dataset-Updated/
├── extracted_paper/                    # Raw extracted paper contents (623 papers)
├── domain_expert_annotated_sources/    # Expert-validated annotations
└── [Additional structured data directories]
```

## SolarChemQA Evaluation: The Benchmark

This repository also provides comprehensive benchmark tasks and evaluation methodologies for the SolarChemQA dataset. The benchmark is designed to evaluate LLM-driven QA systems across three critical dimensions in solar chemistry literature.

### Task 1: Information Retrieval Evaluation
Assesses the effectiveness of different retrieval strategies in accessing relevant information from solar chemistry papers.

Evaluation Target:
- Chunking methodologies: Naive, Recursive, and Semantic Segmentation
- Retrieval strategies: Naive, Hybrid, and Reranking approaches
- Performance at identifying relevant experimental contexts

Evaluation Metric: Normalized Discounted Cumulative Gain (NDCG)

Arguments:
- input_file_dir: Path of raw extracted paper contents folder
- gt_file_dir: Path of the source annotation folder
- res_file_dir: Path for saving the evaluated result folder
- chunk_size: The maximum number of characters or tokens allowed in a single chunk
-overlap: Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.

Usage
```python
python Eval_IR.py --input_file_dir ".../SolarChemQA-Dataset-Updated/extracted_paper" \
                  --gt_file_dir ".../SolarChemQA-Dataset-Updated/domain_expert_annotated_sources/" \
                  --res_file_dir ".../SolarChem-Evaluation/results/" \
                  --chunk_size "1028" \
                  --overlap "128"
```

### Task 2: RAG Method Evaluation
Measures the effectiveness of different Retrieval-Augmented Generation strategies for providing accurate answers.


Evaluation Target:
- Nine combinations of chunking-retrieval methods
- Answer quality across all seven experimental parameter categories
- Semantic fidelity and lexical precision of generated answers


Metrics:
- Semantic similarity (cosine similarity between embeddings)
- Lexical matching (partial_ratio algorithm)


Arguments:
- input_file_dir: Path of raw extracted paper contents folder
- res_file_dir: Path for saving the evaluated result folder
- chunk_type: Naive, Recursive, Semantic.
- rank_type: Naive, Rerank, Hybrid.
- chunk_size: The maximum number of characters or tokens allowed in a single chunk
-overlap: Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.
```python
python Eval_RAG.py --input_file_dir ".../SolarChemQA-Dataset-Updated/extracted_paper" \
                   --res_file_dir ".../SolarChem-Evaluation/results/" \
                   --chunk_type "Semantic" \
                   --rank_type "Hybrid" \
                   --chunk_size "1028" \
                   --overlap "128"
```

### Task 3: LLM Performance Evaluation
Compares the capabilities of various LLMs in understanding and answering questions about solar chemistry experiments.


Evaluation Target:
- Performance of API-based models: gemini-2.0-flash, gemini-2.5-flash, deepseek-v3, deepseek-r1, qwen3-plus, qwen2.5-max
- Performance of locally-run models: deepseek-r1-32b, qwen-30b, gemma3-27b
- Ability to capture experimental settings from scientific literature


Metrics:
- Semantic similarity (cosine similarity between embeddings)
- Lexical matching (partial_ratio algorithm)


Arguments:
- input_file_dir: Path of raw extracted paper contents folder
- res_file_dir: Path for saving the evaluated result folder
- chunk_type: Naive, Recursive, Semantic.
- rank_type: Naive, Rerank, Hybrid.
- chunk_size: The maximum number of characters or tokens allowed in a single chunk
-overlap: Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.
```bash
python Eval_RAG.py --input_file_dir ".../SolarChemQA-Dataset-Updated/extracted_paper" \
                   --res_file_dir ".../SolarChem-Evaluation/results/" \
                   --chunk_type "Semantic" \
                   --rank_type "Hybrid" \
                   --chunk_size "1028" \
                   --overlap "128"
```

## Benchmark Results on SolarChemQA-Dataset-Json

Our benchmark results are available in the results/ directory, including:
- Comparative analysis of retrieval strategies
- Performance metrics for RAG methodologies
- LLM performance rankings
  
### Task 1:

| Question Category | Chunking Method |      |     | Retrieval Method |           |           |
|-------------------|----------------|------|-----|------------------|-----------|-----------|
|                   | Naive          | Recursive | Semantic         | Naive     | Hybrid    | Reranking |
| Catalyst          | 0.3913         | 0.3141    | **0.4056**       | 0.3994    | **0.4250**| 0.2216    |
| Co-Catalyst       | 0.2605         | 0.2424    | **0.3141**       | 0.2456    | **0.3168**| 0.1714    |
| Light Source      | 0.3142         | 0.2884    | **0.3784**       | 0.3356    | **0.3856**| 0.3273    |
| Lamp              | 0.3416         | 0.2997    | **0.4026**       | 0.3385    | **0.4736**| 0.4381    |
| Reactor Type      | 0.2758         | 0.2597    | **0.3110**       | 0.2801    | **0.3640**| 0.3045    |
| Reaction Medium   | 0.2643         | 0.2612    | **0.3164**       | 0.2689    | **0.3621**| 0.2316    |
| Operation Mode    | 0.2421         | 0.2798    | **0.3056**       | 0.2957    | **0.3432**| 0.3750    |
| Average           | 0.2985         | 0.2779    | **0.3477**       | 0.3091    | **0.3815**| 0.2836    |


### Task 2:

| Chunking-Retrieval | Metric | Catalyst | Co-Catalyst | Light Source | Lamp | Reactor Type | Reaction Medium | Operation Mode | Average |
|--------------------|--------|----------|-------------|--------------|------|--------------|-----------------|----------------|---------|
| Naive-Naive        | cos_sim | 0.6830 | 0.6218 | 0.7059 | 0.9338 | 0.5661 | 0.8339 | 0.8258 | 0.7386 |
|                    | ratio   | 0.3902 | 0.3415 | 0.4024 | 0.8415 | 0.0854 | 0.5732 | 0.5610 | 0.4565 |
| Naive-Hybrid       | cos_sim | 0.7016 | 0.6221 | 0.7280 | 0.9296 | 0.5639 | 0.8233 | 0.7570 | 0.7322 |
|                    | ratio   | 0.4756 | 0.3049 | 0.4268 | 0.8049 | 0.0732 | 0.5610 | 0.3780 | 0.4321 |
| Naive-Rerank       | cos_sim | 0.6593 | 0.6428 | 0.7222 | 0.7990 | 0.5704 | 0.7786 | 0.7253 | 0.6997 |
|                    | ratio   | 0.4268 | 0.3780 | 0.4390 | 0.5122 | 0.0488 | 0.4512 | 0.2927 | 0.3641 |
| Recursive-Naive    | cos_sim | 0.6649 | 0.6051 | 0.7013 | 0.9419 | 0.5839 | 0.7889 | 0.8189 | 0.7293 |
|                    | ratio   | 0.3780 | 0.3293 | 0.3537 | 0.8659 | 0.0976 | 0.5122 | 0.5366 | 0.4390 |
| Recursive-Hybrid   | cos_sim | 0.6834 | 0.6191 | 0.7299 | 0.9307 | 0.5707 | 0.8281 | 0.7079 | 0.7243 |
|                    | ratio   | 0.5000 | 0.3049 | 0.4634 | 0.8293 | 0.0854 | 0.5732 | 0.2439 | 0.4286 |
| Recursive-Rerank   | cos_sim | 0.6807 | 0.6170 | 0.7361 | 0.7967 | 0.5769 | 0.7647 | 0.6762 | 0.6926 |
|                    | ratio   | 0.4146 | 0.3293 | 0.4878 | 0.5122 | 0.0610 | 0.3902 | 0.1585 | 0.3362 |
| Semantic-Naive     | cos_sim | 0.6868 | 0.6215 | 0.7188 | 0.8769 | 0.5546 | 0.8622 | 0.8208 | 0.7345 |
|                    | ratio   | 0.4390 | 0.3293 | 0.4268 | 0.6829 | 0.0732 | 0.6220 | 0.5610 | 0.4477 |
| Semantic-Hybrid    | cos_sim | 0.6901 | 0.6308 | 0.6708 | 0.8998 | 0.5639 | 0.8786 | 0.7632 | 0.7282 |
|                    | ratio   | 0.5366 | 0.4512 | 0.3902 | 0.7561 | 0.0854 | 0.6585 | 0.4146 | 0.4703 |
| Semantic-Rerank    | cos_sim | 0.6640 | 0.6449 | 0.7370 | 0.8083 | 0.5607 | 0.8400 | 0.7495 | 0.7149 |
|                    | ratio   | 0.4268 | 0.3780 | 0.5122 | 0.5244 | 0.0366 | 0.5854 | 0.3415 | 0.4007 |

Task 3:
![leaderboard~](/img/llm_performance_solarchemqa.png)
