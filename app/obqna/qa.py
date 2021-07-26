from DataProcessing import PDFParser, Passages

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, DPRQuestionEncoder, DPRContextEncoder, pipeline

import faiss

from tqdm import tqdm
import pandas as pd
import numpy as np


class QuestionAnswering:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        context_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
        question_model_name = "facebook/dpr-question_encoder-single-nq-base"
        answer_model_name = "deepset/roberta-base-squad2"

        self.context_tokenizer = AutoTokenizer.from_pretrained(
            context_model_name)
        self.context_model = DPRContextEncoder.from_pretrained(
            context_model_name).to(self.device)

        self.question_tokenizer = AutoTokenizer.from_pretrained(
            question_model_name)
        self.question_model = DPRQuestionEncoder.from_pretrained(
            question_model_name).to(self.device)

        answer_model = AutoModelForQuestionAnswering.from_pretrained(
            answer_model_name)
        answer_tokenizer = AutoTokenizer.from_pretrained(answer_model_name)

        self.nlp = pipeline('question-answering',
                            model=answer_model, tokenizer=answer_tokenizer)

    def vectorize(self, corpus, batch_size=16):

        corpus = [n['text'] for n in corpus]
        vectors = []

        with torch.no_grad():
            for i in tqdm(range(0, len(corpus), batch_size)):

                encoded_dict = self.context_tokenizer(
                    corpus[i:i+batch_size],
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt').to('cuda')

                outputs = self.context_model(**encoded_dict)
                vectors.extend(outputs[0].detach().to('cpu').numpy())

        lookup = pd.DataFrame({"passages": corpus, "vectors": vectors})
        return lookup

    def save(self, dataframe, path='context.csv'):
        dataframe.to_csv(path, index=False)

    def passages_indexing(self, dataframe):
        vectors = np.array([n for n in dataframe['vectors']])
        self.total_vectors = len(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        print('Indexing completed')

    def rank_passages(self, vectorized_question, sorted_first=10):
        _, indices = self.index.search(vectorized_question, self.total_vectors)
        return indices[0][:sorted_first]

    def vectorize_question(self, question):
        vector = self.question_tokenizer(question, return_tensors='pt')[
            'input_ids'].to('cuda')
        output = self.question_model(
            vector).pooler_output.detach().to('cpu').numpy()
        return output

    def prepare(self, corpus, save_dataset=True):

        self.dataframe = self.vectorize(corpus, batch_size=16)
        if save_dataset:
            self.save(self.dataframe)

        self.passages_indexing(self.dataframe)

        print('Preparation completed')

    def ask(self, question):

        q = self.vectorize_question(question)
        indices = self.rank_passages(q)
        context = " ".join([self.dataframe.iloc[int(n)]
                           ['passages'] for n in indices])
        QA_input = {
            'question': question,
            'context': context
        }
        result = self.nlp(QA_input)

        return result
