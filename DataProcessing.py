import os
from typing import List, Dict

import nltk.data
import pandas as pd
import re
import tika
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from nltk.tokenize import sent_tokenize
from pandarallel import pandarallel
from tika import parser as tikaparser


class PDFParser:
    def __init__(self, books_directory='books/'):
        self.directory = books_directory

    def parse(self):
        corpus = []
        for book in os.listdir(self.directory):
            temp = tikaparser.from_file(self.directory+book)['content']
            temp = " ".join(temp.split('Chapter I'))
            corpus.append(temp)
        return corpus

    def clean(self, corpus):

        corpus = [strip_multiple_whitespaces(n) for n in corpus]
        corpus = [n.encode("ascii", "ignore").decode() for n in corpus]
        corpus = pd.DataFrame({'text': corpus})
        return corpus


class Passages:
    def __init__(self):
        self.nb_workers = os.cpu_count()
        pandarallel.initialize(nb_workers=self.nb_workers)
        self.seg = nltk.data.load("tokenizers/punkt/PY3/english.pickle")

        pattern_sub = re.compile("\\{2}+")
        pattern_sub1 = re.compile("\"")
        pattern_sub2 = re.compile("\'")
        self.pattern_find = re.compile(r'\w+')
        self.patterns = [pattern_sub, pattern_sub1, pattern_sub2]

    def chunker(self, text: str) -> List[List[str]]:
        for pat in self.patterns:
            text = pat.sub('', str(text))
        text = text.encode("ascii", "ignore").decode()

        segmented = self.seg.tokenize(text)
        chunks_n = len(segmented)//(self.nb_workers - 1)
        chunks = [segmented[i:i+chunks_n] for i in range(0, len(segmented), chunks_n)]

        return chunks

    def combine(self, data: List[str], lim: int = 60, upper_lim: int = None) -> List[str]:
        data = {s: len(self.pattern_find.findall(s)) for s in data}
        upper_lim = upper_lim or int(lim*1.2)
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

    def df2passages(self, df):
        """ Transforms DataFrame with content to list of dictionary with the content in passages.
        Args:
            df (DataFrame): [description]
        Returns:
            List[Dict[str, Any]]: [description]
        """
        df['text'] = df['text'].parallel_apply(self.chunker)
        df = df.explode('text').reset_index(drop=True)
        df["text"] = df["text"].parallel_apply(self.combine)
        df = df.explode("text").reset_index(drop=True)

        return df.to_dict('records')
