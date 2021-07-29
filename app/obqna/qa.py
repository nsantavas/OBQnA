import os
from typing import Any, Dict, List

import torch
from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForQuestionAnswering,
        DPRQuestionEncoder,
        DPRContextEncoder,
        pipeline,
)

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from obqna.searcher import Searcher


class QuestionAnswering:
    def __init__(self, searcher_type: str = "faiss"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        context_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
        question_model_name = "facebook/dpr-question_encoder-single-nq-base"
        answer_model_name = "deepset/roberta-base-squad2"

        self.context_tokenizer = AutoTokenizer.from_pretrained(context_model_name)
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name).to(self.device)

        self.question_tokenizer = AutoTokenizer.from_pretrained(question_model_name)
        self.question_model = DPRQuestionEncoder.from_pretrained(question_model_name).to(self.device)

        answer_model = AutoModelForQuestionAnswering.from_pretrained(answer_model_name)
        answer_tokenizer = AutoTokenizer.from_pretrained(answer_model_name)

        self.nlp = pipeline("question-answering", model=answer_model, tokenizer=answer_tokenizer)
        self.searcher = Searcher(searcher_type)


    def vectorize(self, corpus: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
        corpus = [n["text"] for n in corpus]
        vectors = []

        with torch.no_grad():
            for i in tqdm(range(0, len(corpus), batch_size)):
                encoded_dict = self.context_tokenizer(
                    corpus[i:i+batch_size],
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt").to("cuda")

                outputs = self.context_model(**encoded_dict)
                vectors.extend(outputs[0].detach().to("cpu").numpy())

        lookup = pd.DataFrame({"passages": corpus, "vectors": vectors})
        return lookup

    def save(self, dataframe: pd.DataFrame, path="context.pickle") -> None:
        dataframe.to_pickle(path)

    def vectorize_question(self, question: str) -> np.array:
        vector = self.question_tokenizer(question, return_tensors="pt")[
            "input_ids"].to("cuda")
        output = self.question_model(
            vector).pooler_output.detach().to("cpu").numpy()
        return output

    def prepare(self,
                corpus: pd.DataFrame,
                save_dataset: bool = True,
                vectorized_corpus_path: str = "context.pickle") -> None:
        if os.path.isfile(vectorized_corpus_path):
            self.dataframe = pd.read_pickle(vectorized_corpus_path)
        else:
            self.dataframe = self.vectorize(corpus, batch_size=16)
            if save_dataset:
                self.save(self.dataframe, vectorized_corpus_path)

        self.searcher.passages_indexing(self.dataframe)
        print("Preparation completed")

    def ask(self, question: str) -> Dict[str, Any]:
        q = self.vectorize_question(question)
        indices = self.searcher.rank_passages(q)
        context = " ".join([self.dataframe["passages"][n] for n in indices])
        qa_input = {
            "question": question,
            "context": context
        }
        result = self.nlp(qa_input)

        return result
