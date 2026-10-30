from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever

from sklearn.metrics import average_precision_score
from fuzzywuzzy import fuzz

import numpy as np
import argparse
import json
import math
import time
import os


def dcg(relevance_scores):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

def ndcg(ranked_list, k=None):
    if not ranked_list:
        return 0.0

    if k is not None:
        ranked_list = ranked_list[:k]

    dcg_value = dcg(ranked_list)

    ideal_ranking = sorted(ranked_list, reverse=True)
    idcg_value = dcg(ideal_ranking)

    if idcg_value == 0:
        return 0.0

    return dcg_value / idcg_value

def MAP(ranked_list):
    if not ranked_list:
        return 0.0
    R = sum(ranked_list)
    
    if R == 0:
        return 0.0
    
    sum_precision = 0.0
    relevant_count = 0
    
    for index, item in enumerate(ranked_list):
        if item == 1:
            relevant_count += 1
            precision_at_k = relevant_count / (index + 1)
            sum_precision += precision_at_k
    
    AP = sum_precision / R
    return AP

query_index = {
    "catalyst": "what is the catalyst used in the experiment?",
    "co-catalyst": "what is the co-catalyst used in the experiment?",
    "light source": "what is the light source used in the experiment?",
    "lamp": "what is the lamp used in the experiment?",
    "reaction medium": "what is the reaction medium used in the experiment?",
    "reactor type": "what is the reactor type for the experiment?",
    "operation mode": "what is the operation mode for the experiment?"
}

class Chunker:
    def __init__(self, input_path, ground_truth_file, chunk_size, overlap):
        self.ground_truth_path = ground_truth_file
        self.input_data = self.get_text(input_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embeddings = OllamaEmbeddings(model="avr/sfr-embedding-mistral:latest")
    
    def get_text(self, input_path):
        title_list = ["Article_Title", "Abstract", "Experimental", "Results_and_discussion", "Conclusions"]
        f = open(input_path, "rb")
        data = json.load(f)
        text = ""
        for item in data:
            if item["title"] in title_list:
                text += item["title"]
                text += "\n"
                text += item["content"]
                text += "\n"
        return text
    
    def get_chunks(self, chunk_type):
        if chunk_type == "Naive":
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separator='', 
                strip_whitespace=False
            )
        elif chunk_type == "Recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
                is_separator_regex=False,
            )
        # elif chunk_type == "Semantic":
        else:
            number_of_chunk = int(math.ceil(len(self.input_data) / self.chunk_size))
            min_chunk_size = int(math.ceil(self.chunk_size / 2))
            text_splitter = SemanticChunker(self.embeddings, number_of_chunks=number_of_chunk, min_chunk_size=min_chunk_size)
        
        texts = text_splitter.split_text(self.input_data)

        documents = [Document(page_content=x) for x in texts]
        return documents

    def ranking(self, chunks, rank_type, query):
        # self.chunks = self.get_chunks()
        retriever = FAISS.from_documents(chunks, self.embeddings).as_retriever(search_kwargs={"k": len(chunks)})
        if rank_type == "Naive":
            retri_docs = retriever.invoke(query)
            retri_text = [x.page_content for x in retri_docs]
        elif rank_type == "Rerank":
            compressor = FlashrankRerank()
            # compressor = CohereRerank(model="rerank-english-v3.0")
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            compressed_docs = compression_retriever.invoke(query)
            retri_text = [x.page_content for x in compressed_docs]
        else:
            keyword_retriever = BM25Retriever.from_documents(chunks)
            ensemble_retriever = EnsembleRetriever(retrievers=[retriever,keyword_retriever],weights=[0.5, 0.5])
            retri_docs = ensemble_retriever.invoke(query)
            retri_text = [x.page_content for x in retri_docs]
        retri_index = {k: retri_text[k] for k in range(len(retri_text))}
        return retri_index
    
    def eval_rank(self):
        res = {}
        
        with open(self.ground_truth_path, "rb") as f:
            annotation_full = json.load(f)
        annotation_paras = annotation_full["annotation"]
        annotation = {}
        for cate, anno in annotation_paras.items():
            temp = []
            for anno_item in anno:
                if "vote" in list(anno_item.keys()):
                    if anno_item["vote"] == "1":
                        temp.append(anno_item["source"])
            annotation[cate] = temp
        # for para in annotation_paras:
        #     for anno in para["annotations"]:
        #         if anno["category"] in annotation.keys():
        #             annotation[anno["category"]].append(anno["source"])
        #         else:
        #             annotation[anno["category"]] = [anno["source"]]

        for query_type, query in query_index.items():
            if query_type not in annotation.keys():
                res[query_type] = []
                for chunk_type in ["Naive", "Recursive", "Semantic"]:
                    for rank_type in ["Naive", "Rerank", "Hybrid"]:
                        temp_dict = {"chunk_type": chunk_type, "rank_type": rank_type, "ndcg": 0.0, "AP": 0.0}
                        res[query_type].append(temp_dict)
            else:
                res[query_type] = []
                evidences = annotation[query_type]
                for chunk_type in ["Naive", "Recursive", "Semantic"]:
                    chunks = self.get_chunks(chunk_type)
                    for rank_type in ["Naive", "Rerank", "Hybrid"]:
                        sorted_chunks = self.ranking(chunks, rank_type, query)
                        temp = []
                        for chunk_id, chunk_item in sorted_chunks.items():
                            flag = len(temp)
                            for evidence in evidences:
                                ratio = fuzz.partial_ratio(evidence.lower(), chunk_item.lower())
                                if ratio > 80:

                                    temp.append(1)
                                    break
                            if len(temp) == flag:
                                temp.append(0)
                        
                        temp_dict = {"chunk_type": chunk_type, "rank_type": rank_type, "ndcg": ndcg(temp), "AP": MAP(temp)}
                        res[query_type].append(temp_dict)

        return res
    


def get_parser():
    parser = argparse.ArgumentParser(description="Solar Annotation Pipeline")
    parser.add_argument('--input_file_dir', default="/home/jovyan/SolarChemEval/data/extracted_paper/", help="Path of raw extracted paper contents folder", type=str)
    parser.add_argument('--gt_file_dir', default = "/home/jovyan/SolarChemEval/data/domain_expert_anno/", help="Path of the source annotation", type=str)
    parser.add_argument('--res_file_dir', default = "/home/jovyan/SolarChemEval/new_ir_result/", help="Path for saving the evaluated result per paper", type=str)
    parser.add_argument('--chunk_size', default = 1024, help="The maximum number of characters or tokens allowed in a single chunk", type=int)
    parser.add_argument('--overlap', default = 128, help="Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.", type=int)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    input_file_list = os.listdir(args_dict["input_file_dir"])
    gt_file_list = os.listdir(args_dict["gt_file_dir"])
    for gt_file in gt_file_list:
        file_index = gt_file.split("_")[-1].split(".")[0]
        input_file_name = "paper_" + file_index + ".json"
        output_file_name = "result_" + file_index + ".json"
        res_file_path = args_dict["res_file_dir"] + output_file_name
        if os.path.exists(res_file_path):
            pass
        else:
            if input_file_name in input_file_list:
                input_file_path = args_dict["input_file_dir"] + input_file_name
                gt_file_path = args_dict["gt_file_dir"] + gt_file
                chunker = Chunker(input_file_path, gt_file_path, 1024, 128)
                res = chunker.eval_rank()
                with open(res_file_path, "w") as f:
                    json.dump(res, f)

main()