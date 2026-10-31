from app.schemas import Chunk, Paper, RetrievedChunk
from app.verification.span_match import span_score, verify_claim, verify_free_claim

PAPER = Paper(paper_id="p1", title="T", authors=["A. Author"], year=2022, filename="t.pdf")
CHUNK = Chunk(chunk_id="p1-p3-c0", paper_id="p1", page=3,
              text="The LSTM achieves an RMSE of 0.82 dB on the test set. Generalization to other sites was not evaluated.")


def test_verbatim_quote_scores_one_even_with_case_and_whitespace_noise():
    assert span_score("the  LSTM achieves an rmse of 0.82 dB", CHUNK.text) == 1.0


def test_fabricated_quote_is_rejected_and_never_becomes_a_citation():
    vc = verify_claim("LSTM reaches 99% accuracy", "The LSTM reaches 99 percent accuracy on every site.", PAPER, [CHUNK])
    assert vc.status == "unverified"
    assert vc.supporting == []


def test_verified_claim_carries_page_and_source_sentence_not_model_paraphrase():
    vc = verify_claim("LSTM RMSE", "LSTM achieves an RMSE of 0.82 dB", PAPER, [CHUNK], key_terms=["LSTM", "RMSE"])
    assert vc.status == "supported"
    sp = vc.supporting[0]
    assert sp.page == 3 and sp.year == 2022 and sp.authors == ["A. Author"]
    assert sp.quote == "The LSTM achieves an RMSE of 0.82 dB on the test set."  # real source sentence


def test_free_claim_contradicted_by_opposite_polarity_passage():
    p = Paper(paper_id="d", title="D", authors=["X"], year=2024, filename="d.pdf")
    c = Chunk(chunk_id="d-p1-c0", paper_id="d", page=1,
              text="The Transformer does not improve over the LSTM in heavy rain and fails to generalize to a held-out month.")
    vc = verify_free_claim("The Transformer improves over the LSTM in heavy rain", [RetrievedChunk(chunk=c, score=1)], {"d": p})
    assert vc.status == "contradicted" and vc.contradicting
