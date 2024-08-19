from cleartext2 import predict_weighted


def test_smoke():
    pipeline = predict_weighted.Pipeline()
    tok, score = pipeline.predict("a", "b", "c")
    assert isinstance(tok, str)
    assert isinstance(score, float)
