from fastapi import FastAPI
from pydantic import BaseModel

from cleartext2 import predict_weighted

app = FastAPI()


class SelectionInContext(BaseModel):
    before: str
    selection: str
    after: str


@app.post("/")
async def root(context: SelectionInContext):
    pipeline = predict_weighted.Pipeline(top_k=5, likelihood_weight=0, frequency_weight=1)
    top_tok, _ = pipeline.predict(context.before, context.selection, context.after)
    return top_tok
