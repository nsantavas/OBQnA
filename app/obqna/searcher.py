from abc import ABC, abstractmethod
from typing import List

import annoy
import faiss
import numpy as np
import pandas as pd
import scann


class SearcherBase(ABC):
    @abstractmethod
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        return NotImplemented

    @abstractmethod
    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        return NotImplemented
        

class Searcher(SearcherBase):
    def __init__(self, searcher_type: str = "faiss"):
        if searcher_type == "faiss":
            self.searcher = SearcherFaiss()
        elif searcher_type == "annoy":
            self.searcher = SearcherAnnoy()
        elif searcher_type == "scann":
            self.searcher = SearcherScaNN()
        else:
            raise Exception("Provided Searcher Type not Implemented!")

    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        self.searcher.passages_indexing(dataframe)
        print("Indexing completed")

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        return self.searcher.rank_passages(vectorized_question, sorted_first)


class SearcherFaiss(SearcherBase):
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        vectors = np.array(list(dataframe["vectors"]))
        self.total_vectors = len(vectors)
        self.searcher = faiss.IndexFlatIP(vectors.shape[1])
        self.searcher.add(vectors)

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        _, indices = self.searcher.search(vectorized_question, self.total_vectors)
        return indices[0][:sorted_first]


class SearcherAnnoy(SearcherBase):
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        f = dataframe["vectors"][0].shape[0]
        self.total_vectors = len(dataframe["vectors"])
        self.searcher = AnnoyIndex(f, "angular")
        for i, v in enumerate(dataframe["vectors"]):
            self.searcher.add_item(i, v)
        self.searcher.build(500)
        # self.searcher.save("passages.ann")

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        indices = self.searcher.get_nns_by_vector(vectorized_question[0], self.total_vectors)
        return indices[:sorted_first]


class SearcherScaNN(SearcherBase):
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        vectors = np.array(list(dataframe["vectors"]))
        self.searcher = scann.scann_ops_pybind.builder(
            vectors, 10, "dot_product"
        ).score_brute_force().build()

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        indices, _ = self.searcher.search(vectorized_question[0], final_num_neighbors=sorted_first)
        return indices
