# SolarChemQA Benchmark & Dataset

[![DOI](https://zenodo.org/badge/925259194.svg)](https://doi.org/10.5281/zenodo.20843928)

This repository contains the benchmark and dataset for [SolarQA](https://github.com/oeg-upm/solar-qa) application. The SolarChemQA Benchmark is designed specifically for factual question answering benchmark based on solar chemistry academic papers. We intent to build an open-source benchmark & dataset for testing the performance of QA system when it comes to narrow-domain and in-depth questions. 

The SolarChemQA Benchmark addresses the critical need for domain-specific evaluation tools in scientific literature processing. It aims to:

- Build an open-source benchmark for testing QA system performance on narrow-domain, in-depth questions
- Provide rigorous assessment capabilities for LLM-driven QA systems processing scientific content
- Enable comprehensive evaluation of retrieval-augmented generation (RAG) strategies in solar chemistry


## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"
```

The OpenRouter key is used by the generation pipeline and by the qwen3-embedding-8b encoder in the evaluation notebooks. Task 1 runs its embeddings through a local [Ollama](https://ollama.com) server instead, so it additionally needs:

```bash
ollama pull qwen3-embedding:8b
```

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

The implementation is under `src/generation`. A single run is configured in `config.yaml` (model, RAG strategy, prompts and queries), and writes one `evidences_<id>.json` and one `annotated_annotation_<id>.json` per paper:

```bash
cd src/generation
python generate.py --config config.yaml
```

Sweeps over RAG configurations or models reuse the same base config through overlay files:

```bash
python batch.py --batch batch_rag.yaml    # 9 chunking-retrieval combinations (Task 2)
python batch.py --batch batch_eval.yaml   # 8 LLMs under Naive-Naive (Task 3)
```

Both batch files write into the `result/` folders under `src/evaluation`, and papers with an existing output file are skipped, so re-running against the released results costs nothing.




## SolarChemQA Dataset


SolarChemQA provides a novel question answering dataset curated from solar chemistry literature, designed to rigorously assess the capabilities of LLM-driven QA systems in processing domain-specific scientific content. 

### Dataset Statistics

The foundational corpus contains 1,096 solar chemistry papers (1979-2022) from the Art Leaf dataset, each annotated by a domain expert with the seven experimental parameters, for 7,672 annotations in total. Filtering for openly available papers that report a single photocatalytic experiment leaves 294 papers with 2,058 annotations, which is the filtered SolarChemQA dataset used for the benchmark. From these, 150 papers (1,050 annotations) are sampled for the LLM evaluation, and 21 papers are further annotated with 291 sentence-level evidences (217 positive, 74 negative) for the retrieval evaluation, validated independently by at least two annotators.

- Query Sub-Dataset: Standardized queries for the seven experimental parameter categories (see `queries` in `src/generation/config.yaml`)
- Paper Extraction Sub-Dataset: Structured content extracted from research papers
- Annotation Sub-Dataset: Expert-validated annotations and the supporting textual sentences


<!-- SolarChemQA provides a novel question answering dataset curated from solar chemistry literature, designed to rigorously assess the capabilities of LLM-driven QA systems in processing domain-specific scientific content. 

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
``` -->


## SolarChemQA Evaluation: The Benchmark



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


Usage:

Step 1: generate the predictions for the nine RAG configurations, with gemma-4-26b-a4b-it as the fixed LLM. The released predictions are already under `result/` and existing files are skipped, so this step only costs API calls if you delete them first.

```bash
cd src/generation
python batch.py --batch batch_rag.yaml
```

Step 2: score the predictions against the ground truth.

```bash
cd ../evaluation/Task-2-RAG-methods
python task2_evaluation.py --results_dir ./result-150/ --ground_truth ./ground_truth.json --vecs_cache ./vecs.pkl --json_out ./task2_evaluation.json
```

Run all cells. The notebook embeds every answer with qwen3-embedding-8b (cached in `src/evaluation/embedding_cache.pkl`), prints the per-category table.


### Task 3: LLM Performance Evaluation
Compares the capabilities of various LLMs in understanding and answering questions about solar chemistry experiments.


Evaluation Target:
- Performance of API-based models: gemini-2.0-flash, gemini-2.5-flash, deepseek-v3, deepseek-r1, qwen3-plus, qwen2.5-max
- Performance of locally-run models: deepseek-r1-32b, qwen-30b, gemma3-27b
- Ability to capture experimental settings from scientific literature


Metrics:
- Semantic similarity (cosine similarity between embeddings)
- Lexical matching (partial_ratio algorithm)


Usage:

Step 1: generate the predictions of the eight LLMs, all under the Naive-Naive RAG configuration selected in Task 2. As in Task 2, the released predictions are already under `result/` and are skipped on re-run.

```bash
cd src/generation
python batch.py --batch batch_eval.yaml
```

Step 2: score the predictions and produce the figures.

```bash
cd ../evaluation/Task-3-LLM-performance
python task3_evaluation.py --results_dir ./result-150/ --ground_truth ./ground_truth.json --vecs_cache ./vecs.pkl --json_out ./task3_evaluation.json --output_dir figures/
```

Run all cells. The notebook prints the per-category tables and writes `task3_metrics.csv`, the leaderboard figure and the heatmaps into `figures/`. It shares the embedding cache and the scoring rules with the Task 2 notebook, so the gemma-4-26b-a4b-it Naive-Naive values match between the two tasks.


## Benchmark Results

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


### Task 2: RAG Configuration Evaluation

Table：Performance evaluation of different RAG strategies across experiment parameter categories under the fixed LLM (gemma-4-26b-a4b-it) and embedding model (qwen3-embedding-8b)

| RAG Config | Metric | Catalyst | Co-Catalyst | Light Source | Lamp | Reactor Type | Reaction Medium | Operation Mode | Average |
|---|---|---|---|---|---|---|---|---|---|
| Naive-Naive | cos_sim | 0.6271 | 0.6249 | 0.6338 | 0.8562 | 0.7078 | 0.9390 | 0.8782 | 0.7524 |
| Naive-Naive | ratio | 0.6486 | 0.4577 | 0.5571 | 0.8693 | 0.5219 | 0.7123 | 0.6302 | 0.6282 |
| Naive-Hybrid | cos_sim | 0.6176 | 0.6187 | 0.6475 | 0.8419 | 0.6836 | 0.9243 | 0.8910 | 0.7464 |
| Naive-Hybrid | ratio | 0.6557 | 0.4335 | 0.5608 | 0.8410 | 0.5005 | 0.7093 | 0.6186 | 0.6170 |
| Naive-Rerank | cos_sim | 0.6291 | 0.6175 | 0.6768 | 0.7587 | 0.5791 | 0.8829 | 0.7874 | 0.7045 |
| Naive-Rerank | ratio | 0.6837 | 0.4434 | 0.5265 | 0.7742 | 0.2590 | 0.6542 | 0.4817 | 0.5461 |
| Recursive-Naive | cos_sim | 0.6199 | 0.6232 | 0.6494 | 0.8468 | 0.6711 | 0.9477 | 0.8779 | 0.7480 |
| Recursive-Naive | ratio | 0.6846 | 0.4605 | 0.5833 | 0.8632 | 0.4553 | 0.7291 | 0.6484 | 0.6321 |
| Recursive-Hybrid | cos_sim | 0.5942 | 0.6026 | 0.6307 | 0.8428 | 0.6843 | 0.9139 | 0.8663 | 0.7335 |
| Recursive-Hybrid | ratio | 0.6676 | 0.4659 | 0.5388 | 0.8410 | 0.4747 | 0.7037 | 0.6080 | 0.6142 |
| Recursive-Rerank | cos_sim | 0.6337 | 0.6027 | 0.6627 | 0.7132 | 0.5815 | 0.8754 | 0.7984 | 0.6954 |
| Recursive-Rerank | ratio | 0.6507 | 0.4633 | 0.5819 | 0.7280 | 0.2543 | 0.6371 | 0.4960 | 0.5445 |
| Semantic-Naive | cos_sim | 0.6025 | 0.5938 | 0.6442 | 0.8184 | 0.6896 | 0.9329 | 0.8910 | 0.7389 |
| Semantic-Naive | ratio | 0.6393 | 0.4153 | 0.5999 | 0.8371 | 0.5122 | 0.7351 | 0.6607 | 0.6285 |
| Semantic-Hybrid | cos_sim | 0.6518 | 0.6156 | 0.6608 | 0.8217 | 0.6766 | 0.9341 | 0.8702 | 0.7472 |
| Semantic-Hybrid | ratio | 0.6533 | 0.4366 | 0.5951 | 0.8451 | 0.4827 | 0.6958 | 0.6165 | 0.6179 |
| Semantic-Rerank | cos_sim | 0.6191 | 0.6106 | 0.6674 | 0.7370 | 0.6329 | 0.8939 | 0.8126 | 0.7105 |
| Semantic-Rerank | ratio | 0.6183 | 0.4505 | 0.5746 | 0.7440 | 0.3641 | 0.6155 | 0.5220 | 0.5556 |

Task 3: LLM Benchmark

Leaderboard:

![leaderboard~](/img/llm_performance_solarchemqa.png)

Table: Per-category performance of the evaluated LLMs under the Naive-Naive RAG configuration with the qwen3-embedding-8b encoder

| Model | Metric | Catalyst | Co-Catalyst | Light Source | Lamp | Reactor Type | Reaction Medium | Operation Mode | Average |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-plus | cos_sim | 0.6666 | 0.6183 | 0.7124 | 0.8391 | 0.7230 | 0.9479 | 0.8974 | 0.7721 |
| qwen3.6-plus | ratio | 0.6725 | 0.4338 | 0.5943 | 0.8595 | 0.5666 | 0.7574 | 0.6812 | 0.6522 |
| qwen3.6-27b | cos_sim | 0.6941 | 0.6245 | 0.7108 | 0.8311 | 0.6982 | 0.9449 | 0.8827 | 0.7695 |
| qwen3.6-27b | ratio | 0.6828 | 0.4641 | 0.5707 | 0.8385 | 0.5006 | 0.7197 | 0.6500 | 0.6324 |
| qwen3.6-flash | cos_sim | 0.6126 | 0.6101 | 0.7084 | 0.8415 | 0.6933 | 0.9429 | 0.8904 | 0.7570 |
| qwen3.6-flash | ratio | 0.5363 | 0.4482 | 0.4995 | 0.8453 | 0.4953 | 0.7303 | 0.6674 | 0.6032 |
| gemma-4-26b-a4b-it | cos_sim | 0.6128 | 0.6288 | 0.6539 | 0.8453 | 0.7211 | 0.9365 | 0.8888 | 0.7553 |
| gemma-4-26b-a4b-it | ratio | 0.6269 | 0.4428 | 0.5815 | 0.8513 | 0.5729 | 0.7172 | 0.6461 | 0.6341 |
| gemma-4-31b-it | cos_sim | 0.6146 | 0.6172 | 0.7286 | 0.8324 | 0.6618 | 0.9446 | 0.8446 | 0.7491 |
| gemma-4-31b-it | ratio | 0.7324 | 0.4038 | 0.6101 | 0.8323 | 0.4357 | 0.7377 | 0.5913 | 0.6205 |
| qwen3.6-35b-a3b | cos_sim | 0.5735 | 0.6048 | 0.7021 | 0.8271 | 0.7002 | 0.9444 | 0.8914 | 0.7491 |
| qwen3.6-35b-a3b | ratio | 0.5066 | 0.4054 | 0.5183 | 0.8431 | 0.5212 | 0.7357 | 0.6846 | 0.6021 |
| deepseek-v4-pro | cos_sim | 0.5535 | 0.6045 | 0.7590 | 0.8038 | 0.6831 | 0.8781 | 0.8546 | 0.7338 |
| deepseek-v4-pro | ratio | 0.4480 | 0.3715 | 0.6179 | 0.8028 | 0.4661 | 0.5957 | 0.6350 | 0.5624 |
| deepseek-v4-flash | cos_sim | 0.5570 | 0.6104 | 0.7445 | 0.8103 | 0.6694 | 0.8610 | 0.8256 | 0.7255 |
| deepseek-v4-flash | ratio | 0.3916 | 0.3675 | 0.5339 | 0.8363 | 0.4416 | 0.6001 | 0.5701 | 0.5344 |
