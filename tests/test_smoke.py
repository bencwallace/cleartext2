from cleartext2 import predict_weighted


def test_smoke():
    pipeline = predict_weighted.Pipeline(top_k=10, likelihood_weight=0, frequency_weight=1)
    tok, score = pipeline.predict("a", "b", "c")
    assert isinstance(tok, str)
    assert isinstance(score, float)
