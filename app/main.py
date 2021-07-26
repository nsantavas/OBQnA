from fastapi import FastAPI
from obqna.process import Passages, PDFParser
from obqna.qa import QuestionAnswering

app = FastAPI()

passages = Passages()
parser = PDFParser()
qa = QuestionAnswering()

@app.get("/", status_code=200)
async def root():
    return {"message": "OK"}

@app.get("/", status_code=200)
async def corpus_parser():
    """Parse Corpus and split to passages

    Returns:
        [type]: [description]
    """
    return {"message": "OK"}

@app.get("/", status_code=200)
async def question_answering():
    """QA Endpoint

    Returns:
        [type]: [description]
    """
    return {"message": "OK"}