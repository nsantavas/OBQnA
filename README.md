<h1 align="center">
    <p>OBQnA</p>
</h1>
<h2 align="center">
    <p>An OpenBook Question 'n' Answer System</p>
</h2>

Introduction
------------
`OBQnA` is a high-level OO Python package which aims to provide an easy and intuitive way of creating a OpenBook Question ‘n’ Answer system. <p>
The package parses PDF files using <a href="https://github.com/chrismattmann/tika-python">Apache Tika</a>, splits the corpus into passages and calculates their corresponding Dense vector representation exploiting a Transformer NLP model. For each question asked, the system performs a Dense Passage Retrieval, using an efficient similarity search library (<a href="https://github.com/facebookresearch/faiss">Faiss</a>, <a href="https://github.com/google-research/google-research/tree/master/scann">ScaNN</a> or <a href="https://github.com/spotify/annoy">Annoy</a>) and extracts the answer from the retrieved passages.

-------------

Install
-------
To install simply do ``pip install -r requirements.tx``

* note: If you want to use GPU please install CUDA

----------

Python code example
------------


<h3 align="center">We are using J. R. R. Tolkien's Lord Of The Rings Trilogy and the Hobbit for the following example.</p></h3>

<p align="center">
  <img src="images/lotr.png" height="250">
</p>

For more detailed explanation please read the <a href="https://nsantavas.github.io/OBQnA/">Documentation</a>.

------------

<h3>Parsing PDFs and performing some basic text cleaning</p></h3>

``` python
from obqna.process import PDFParser, Passages
from obqna.qa import QuestionAnswering

parser = PDFParser("../books/") # Path of PDFs
books = parser.parse()
books = parser.clean(books)
```

<h3>Splitting the corpus into passages</p></h3>

``` python
passages = Passages()
corpus = passages.df2passages(books)
```


<h3>Calculate the vector represantation of each passage and store the corresponding indices</p></h3>

``` python
searcher_type = "scann" # other choices: "faiss", "annoy"
qna = QuestionAnswering(searcher_type)
qna.prepare(corpus)
```

<h3>Ask questions</p></h3>

``` python
questions = [
    "Who is Galadriel?",
    "Who is Isildur?",
    "Who is Boromir's father?",
    "Who is Aragorn?",
    "Was the ring destroyed?",
    "What language is on the One Ring inscription?"
]

for question in questions:
    print(f"Question: {question}: ")
    results = qna.ask(question)
    print(f"Answer: {results['answer']}")
    print(10*'-')

  ```

```
Question: Who is Galadriel?: 
Answer: The Lady of Lorien
----------
Question: Who is Isildur?: 
Answer: Elendils son
----------
Question: Who is Boromir's father?: 
Answer: Lord Denethor
----------
Question: Who is Aragorn?: 
Answer: Heir of Isildur
----------
Question: Was the ring destroyed?: 
Answer: it perished from the world in the ruin of his first realm
----------
Question: What language is on the One Ring inscription?: 
Answer: Black Speech
----------
```
