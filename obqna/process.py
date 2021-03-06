import os
from typing import List, Dict

import nltk.data
import pandas as pd
import re
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from nltk.tokenize.punkt import PunktSentenceTokenizer
from pandarallel import pandarallel
from tika import parser as tikaparser


class PDFParser:
    def __init__(self, books_directory: str = "books/"):
        self.directory = books_directory

    def parse(self) -> List[str]:
        """ Parses the raw pdf(s) provided in the books_directory

        :return: A list wiht the parsed text of the pdf(s)
        :rtype: List[str]
        """
        corpus = []
        for book in os.listdir(self.directory):
            temp = tikaparser.from_file(self.directory + book)["content"]
            temp = " ".join(temp.split("Chapter I"))
            corpus.append(temp)
        return corpus

    def clean(self, corpus: List[str]) -> pd.DataFrame:
        """ Applies basic text cleaning.

        :param corpus: A list wiht the parsed text of the pdf(s)
        :type corpus: List[str]
        :return: A DataFrame with a single column named "text" containing the cleaned input
        :rtype: pd.DataFrame
        """
        corpus = [strip_multiple_whitespaces(n) for n in corpus]
        corpus = [n.encode("ascii", "ignore").decode() for n in corpus]
        corpus = pd.DataFrame({"text": corpus})
        return corpus


class Passages:
    def __init__(self):
        self.nb_workers: int = os.cpu_count()
        self.seg: PunktSentenceTokenizer = nltk.data.load(
            "tokenizers/punkt/PY3/english.pickle"
        )

        pattern_sub: re.Pattern = re.compile("\\{2}+")
        pattern_sub1: re.Pattern = re.compile('"')
        pattern_sub2: re.Pattern = re.compile("'")
        self.pattern_find: re.Pattern = re.compile(r"\w+")
        self.patterns: List[re.Pattern] = [pattern_sub, pattern_sub1, pattern_sub2]

        pandarallel.initialize(nb_workers=self.nb_workers)

    def chunker(self, text: str) -> List[List[str]]:
        """ Brakes a text into passages for parallelization

        :param text: A single text
        :type text: str
        :return: A List containing chunks of the input text, where chunks are lists of sentences
        :rtype: List[List[str]]
        """
        for pat in self.patterns:
            text = pat.sub("", str(text))
        text = text.encode("ascii", "ignore").decode()

        segmented = self.seg.tokenize(text)
        chunks_n = len(segmented) // (self.nb_workers - 1)
        chunks = [
            segmented[i : i + chunks_n] for i in range(0, len(segmented), chunks_n)
        ]

        return chunks

    def combine(
        self, data: List[str], lim: int = 60, upper_lim: int = None
    ) -> List[str]:
        """ Combines A list of sentences to passages of approximate num of words close to `lim`

        :param data: A list containing sentences
        :type data: List[str]
        :param lim: The lim that num of words should approximate, defaults to 60
        :type lim: int, optional
        :param upper_lim: The max lim that num of words should not exceed, defaults to None
        :type upper_lim: int, optional
        :return: List of passages
        :rtype: List[str]
        """
        data = {s: len(self.pattern_find.findall(s)) for s in data}
        upper_lim = upper_lim or int(lim * 1.2)
        passages = []
        temp = []
        temp_value = 0
        for key, value in data.items():
            if not temp:
                if value > lim:
                    passages.append(key)
                else:
                    temp.append(key)
                    temp_value = value
            elif temp_value + value > upper_lim:
                passages.append(" ".join(temp))
                temp = []
                temp_value = 0
                if value > lim:
                    passages.append(key)
                else:
                    temp.append(key)
                    temp_value = value
            elif temp_value + value > lim:
                temp.append(key)
                passages.append(" ".join(temp))
                temp = []
                temp_value = 0
            else:
                temp.append(key)
                temp_value += value

        return passages

    def df2passages(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """ Transforms DataFrame containing text(s) to a list of dicts containing passages of the text(s)

        :param df: DataFrame containing all the text(s)
        :type df: pd.DataFrame
        :return: List of dicts containing passages of the text(s)
        :rtype: List[Dict[str, str]]
        """
        df["text"] = df["text"].parallel_apply(self.chunker)
        df = df.explode("text").reset_index(drop=True)
        df["text"] = df["text"].parallel_apply(self.combine)
        df = df.explode("text").reset_index(drop=True)

        corpus = df.to_dict("records")
        corpus = [{k: str(v) for k, v in corp.items()} for corp in corpus]

        return corpus
