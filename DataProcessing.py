import os
from typing import List, Dict

import pandas as pd
import tika
from tika import parser as tikaparser
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import re
import pysbd

from pandarallel import pandarallel


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
        self.seg = pysbd.Segmenter(language="en", clean=False)

        pattern_sub = re.compile("\\{2}+")
        pattern_sub1 = re.compile("\"")
        pattern_sub2 = re.compile("\'")
        self.pattern_find = re.compile(r'\w+')
        self.patterns = [pattern_sub, pattern_sub1, pattern_sub2]

    def chunker(self, text: str) -> List[Dict[str, int]]:
        for pat in self.patterns:
            text = pat.sub('', str(text))
        text = text.encode("ascii", "ignore").decode()

        segmented = self.seg.segment(text)
        chunks_n = len(segmented)//(self.nb_workers - 1)
        chunks = [segmented[i:i+chunks_n] for i in range(0, len(segmented), chunks_n)]
        chunks_length = [{s: len(self.pattern_find.findall(s)) for s in segmented}
                         for segmented in chunks]

        return chunks_length

    def combine(data: Dict[str, int], lim: int = 60) -> List[str]:
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
            elif temp_value + value > lim+20:
                passages.append(" ".join(temp))
                temp = []
                temp_value = 0
                if value > lim:
                    passages.append(key)
                else:
                    temp.append(key)
                    temp_value = value
                continue
            elif temp_value + value > lim:
                temp.append(key)
                passages.append(" ".join(temp))
                temp = []
                temp_value = 0
            else:
                temp.append(key)
                temp_value = 0
        
        return passages

    def split(self, segmented: List[str], lim=60):
        """ Splits a passage to smaller passages with a number of tokens close to lim.
        Args:
            segmented: List[str]: A chunk of sentences.
            lim (int, optional): Defaults to 60.
        Returns:
            List[str]:
        """
        passage_list = []
        i = 0
        if segmented:
            while True:
                s = segmented[i]
                text = s
                res = len(self.pattern_find.findall(s))  # Count tokens
                flag = True

                # Concat until tokens are close to lim
                while res < lim:
                    flag = False
                    i += 1
                    if i >= len(segmented):
                        break
                    text = s
                    s += segmented[i]
                    res = len(self.pattern_find.findall(s))

                # Check if it didn't passed through the second while
                if flag:
                    i += 1

                # Check the last occurance
                if i >= len(segmented):
                    if res < lim or flag:
                        passage_list.append(s.strip())
                        break
                    passage_list.append(text.strip())
                    passage_list.append(segmented[i-1].strip())
                    break

                passage_list.append(text.strip())
        else:
            passage_list.append('z')

        return passage_list

    def df2passages(self, df):
        """ Transforms DataFrame with content to list of dictionary with the content in passages.
        Args:
            df (DataFrame): [description]
        Returns:
            List[Dict[str, Any]]: [description]
        """
        df['text'] = df['text'].parallel_map(self.split)
        data = df.explode('text').reset_index(drop=True)

        return data.to_dict('records')
