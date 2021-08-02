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
        """ Creates indices for the vectors of the passages

        :param dataframe: DataFrame containing passages and their vectors
        :type dataframe: pd.DataFrame
        """        
        return NotImplemented

    @abstractmethod
    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        """ Runs vector similarity search to retrieve the most relevant passages to a question

        :param vectorized_question: Vector of question
        :type vectorized_question: np.array
        :param sorted_first: Num of passages indices to return, defaults to 10
        :type sorted_first: int, optional
        :return: Passages indices
        :rtype: List[int]
        """        
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
        """ Creates indices for the vectors of the passages

        :param dataframe: DataFrame containing passages and their vectors
        :type dataframe: pd.DataFrame
        """        
        self.searcher.passages_indexing(dataframe)
        print("Indexing completed")

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        """ Runs vector similarity search to retrieve the most relevant passages to a question

        :param vectorized_question: Vector of question
        :type vectorized_question: np.array
        :param sorted_first: Num of passages indices to return, defaults to 10
        :type sorted_first: int, optional
        :return: Passages indices
        :rtype: List[int]
        """        
        return self.searcher.rank_passages(vectorized_question, sorted_first)


class SearcherFaiss(SearcherBase):
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        """ Creates indices for the vectors of the passages

        :param dataframe: DataFrame containing passages and their vectors
        :type dataframe: pd.DataFrame
        """        
        vectors = np.array(list(dataframe["vectors"]))
        self.total_vectors = len(vectors)
        self.searcher = faiss.IndexFlatIP(vectors.shape[1])
        self.searcher.add(vectors)

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        """ Runs vector similarity search to retrieve the most relevant passages to a question

        :param vectorized_question: Vector of question
        :type vectorized_question: np.array
        :param sorted_first: Num of passages indices to return, defaults to 10
        :type sorted_first: int, optional
        :return: Passages indices
        :rtype: List[int]
        """        
        _, indices = self.searcher.search(vectorized_question, self.total_vectors)
        return indices[0][:sorted_first]


class SearcherAnnoy(SearcherBase):
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        """ Creates indices for the vectors of the passages

        :param dataframe: DataFrame containing passages and their vectors
        :type dataframe: pd.DataFrame
        """        
        f = dataframe["vectors"][0].shape[0]
        self.total_vectors = len(dataframe["vectors"])
        self.searcher = annoy.AnnoyIndex(f, "angular")
        for i, v in enumerate(dataframe["vectors"]):
            self.searcher.add_item(i, v)
        self.searcher.build(500)
        # self.searcher.save("passages.ann")

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        """ Runs vector similarity search to retrieve the most relevant passages to a question

        :param vectorized_question: Vector of question
        :type vectorized_question: np.array
        :param sorted_first: Num of passages indices to return, defaults to 10
        :type sorted_first: int, optional
        :return: Passages indices
        :rtype: List[int]
        """        
        indices = self.searcher.get_nns_by_vector(vectorized_question[0], self.total_vectors)
        return indices[:sorted_first]


class SearcherScaNN(SearcherBase):
    def passages_indexing(self, dataframe: pd.DataFrame) -> None:
        """ Creates indices for the vectors of the passages

        :param dataframe: DataFrame containing passages and their vectors
        :type dataframe: pd.DataFrame
        """        
        vectors = np.array(list(dataframe["vectors"]))
        self.searcher = scann.scann_ops_pybind.builder(
            vectors, 10, "dot_product"
        ).score_brute_force().build()

    def rank_passages(self, vectorized_question: np.array, sorted_first: int = 10) -> List[int]:
        """ Runs vector similarity search to retrieve the most relevant passages to a question

        :param vectorized_question: Vector of question
        :type vectorized_question: np.array
        :param sorted_first: Num of passages indices to return, defaults to 10
        :type sorted_first: int, optional
        :return: Passages indices
        :rtype: List[int]
        """        
        indices, _ = self.searcher.search(vectorized_question[0], final_num_neighbors=sorted_first)
        return indices