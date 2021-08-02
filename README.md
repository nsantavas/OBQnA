<h1 align="center">
    <p>OBQnA</p>
</h1>
<h2 align="center">
    <p>An OpenBook Question 'n' Answer System</p>
</h2>

Install
-------
To install simply do ``pip install -r requirements.tx``

* note: If you want to use GPU please install cuda

<p>

Python code example
-------------------
<p>

``` python
from app.obqna.process import PDFParser, Passages
from app.obqna.qa import QuestionAnswering

parser = PDFParser("../books/")
books = parser.parse()
books = parser.clean(books)

passages = Passages()
corpus = passages.df2passages(books)

searcher_type = "scann" # other choices: "faiss", "annoy"
qna = QuestionAnswering(searcher_type)
qna.prepare(corpus)

questions = [
    "Who is Galadriel?",
    "Who is Gandalf?",
    "Was the ring destroyed?",
]

for question in questions:
    print(f"{question}: ")
    print(qna.ask(question))

  ```

```
Who is Galadriel?: 
{'score': 0.002381378784775734, 'start': 1352, 'end': 1370, 'answer': 'The Lady of Lorien'}
Who is Gandalf?: 
{'score': 0.24408744275569916, 'start': 0, 'end': 16, 'answer': 'Gandalf the Grey'}
Was the ring destroyed?: 
{'score': 0.19746769964694977, 'start': 2126, 'end': 2183, 'answer': 'it perished from the world in the ruin of his first realm'}
```
