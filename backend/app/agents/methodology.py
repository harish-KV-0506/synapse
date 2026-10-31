"""Methodology Agent: per-paper objective, methods, datasets, metrics, results, limitations.

Every extracted item is a VerifiedClaim whose citation is a verbatim sentence from the paper.
- Heuristic mode: lexicon + pattern matching over the paper's own sentences (always cited, by construction).
- LLM mode: the model proposes {claim, quote}; the quote is span-matched against the paper, and discarded
  as evidence if it is not really there.
"""
from __future__ import annotations

import re

from ..llm import llm
from ..retrieval.hybrid import corpus
from ..schemas import Chunk, Paper, PaperProfile, VerifiedClaim
from ..verification.span_match import split_sentences, verify_claim
from .state import WorkflowState, traced

ALGORITHMS = {
    "LSTM": r"\blstm\b|long short-term memory", "GRU": r"\bgru\b|gated recurrent", "CNN": r"\bcnn\b|convolutional neural",
    "Transformer": r"\btransformers?\b|attention mechanism", "Random Forest": r"random forests?",
    "XGBoost": r"\bxgboost\b|gradient boost", "SVM/SVR": r"\bsvr?m?\b|support vector", "ANN/MLP": r"\bann\b|\bmlp\b|multilayer perceptron|artificial neural network",
    "Linear/Logistic Regression": r"(?:linear|logistic) regression", "k-NN": r"\bk-?nn\b|nearest neighbou?rs?",
    "Gaussian Process": r"gaussian process", "Bayesian methods": r"\bbayesian (?!optimi[sz]ation)\w+|\bbayesian$", "Reinforcement Learning": r"reinforcement learning|\bdqn\b",
    "Autoencoder": r"auto-?encoders?", "GAN": r"\bgans?\b|generative adversarial", "ARIMA": r"\barima\b",
    "Physics-informed / hybrid": r"physics-informed|physics-based hybrid|hybrid (?:model|approach|forecast)", "ITU-R baseline": r"itu-r\s*p\.?\s*\d+",
    "Ensemble": r"\bensembles?\b", "Decision Tree": r"decision trees?",
}
METRICS = {
    "RMSE": r"\brmse\b|root mean square", "MAE": r"\bmae\b|mean absolute error", "MAPE": r"\bmape\b",
    "R²": r"\br\s?2\b|r-squared|r²|coefficient of determination", "Accuracy": r"\baccuracy\b", "F1": r"\bf1(?:-score)?\b",
    "Precision/Recall": r"\bprecision\b|\brecall\b", "AUC": r"\bauc\b|\broc\b", "Correlation": r"correlation coefficient|pearson",
    "Outage probability": r"outage (?:probability|rate)", "BER": r"\bber\b|bit error",
    "CDF/CCDF": r"\bc?cdf\b", "Event-level / peak error": r"peak (?:error|event|attenuation)|event-level",
}
DATASET_PAT = re.compile(r"\b(data ?sets?|measurements?|beacon|campaign|recorded|collected|corpus|benchmark|observations?|"
                         r"gauge|disdrometer|radar|reanalysis|era5|logs?)\b", re.I)
OBJECTIVE_PAT = re.compile(r"\b(this (?:paper|study|work|article)|we|our (?:work|study))\b.{0,40}\b(aims?|propos\w+|presents?|investigat\w+|"
                           r"develops?|introduces?|evaluates?|explores?|examines?|addresses|focus\w*)\b", re.I)
LIMIT_PAT = re.compile(r"\b(limitation|limited to|limited by|only (?:one|a single|a few|two)|single (?:site|location|station)|"
                       r"not (?:evaluated|tested|validated|considered|investigated|addressed)|future work|remains? (?:unclear|unexplored|open)|"
                       r"does not generali[sz]e|fails? to|short(?:er)? (?:duration|period)|small(?:er)? (?:sample|dataset)|"
                       r"lack of|insufficient|constrain\w+|cannot)\b", re.I)
RESULT_PAT = re.compile(r"\b(achiev\w+|outperform\w*|improv\w+|reduc\w+|reach\w+|obtain\w+|yield\w*|result\w* in|error of|accuracy of)\b", re.I)
NUM_PAT = re.compile(r"\d")


def _sentences(chunks: list[Chunk]) -> list[tuple[str, Chunk]]:
    seen, out = set(), []
    for c in sorted(chunks, key=lambda c: (c.page, c.chunk_id)):
        for s in split_sentences(c.text):
            key = re.sub(r"\W+", "", s.lower())
            if key not in seen and len(s.split()) >= 6:
                seen.add(key)
                out.append((s, c))
    return out


def _lexicon_claims(label: str, lex: dict[str, str], sents, paper, chunks, verb: str, max_items: int = 6) -> list[VerifiedClaim]:
    out: list[VerifiedClaim] = []
    for name, pat in lex.items():
        rx = re.compile(pat, re.I)
        hits = [(s, c) for s, c in sents if rx.search(s)]
        if not hits:
            continue
        # prefer sentences from method/results sections, then earliest page
        hits.sort(key=lambda sc: (0 if (sc[1].section or "").lower().startswith(("method", "experiment", "result", "data")) else 1, sc[1].page))
        s, _ = hits[0]
        out.append(verify_claim(f"{verb} {name}", s, paper, chunks, key_terms=[name.split("/")[0].split(" ")[0]],
                                note=f"Mentioned in {len(hits)} passage(s)."))
    out.sort(key=lambda v: v.confidence, reverse=True)
    return out[:max_items]


def _heuristic_profile(paper: Paper, chunks: list[Chunk]) -> PaperProfile:
    sents = _sentences(chunks)
    prof = PaperProfile(paper_id=paper.paper_id, title=paper.title, authors=paper.authors, year=paper.year)
    # objective: abstract/intro sentence matching objective pattern
    cand = [(s, c) for s, c in sents if OBJECTIVE_PAT.search(s)]
    cand.sort(key=lambda sc: (sc[1].page, 0 if (sc[1].section or "").lower() in ("abstract", "introduction") else 1))
    if cand:
        s, _ = cand[0]
        prof.objective = verify_claim(s, s, paper, chunks, key_terms=[])
    prof.methods = _lexicon_claims("methods", ALGORITHMS, sents, paper, chunks, "Uses")
    prof.metrics = _lexicon_claims("metrics", METRICS, sents, paper, chunks, "Evaluates with")
    ds = [(s, c) for s, c in sents if DATASET_PAT.search(s) and (NUM_PAT.search(s) or "dataset" in s.lower())]
    ds.sort(key=lambda sc: (0 if (sc[1].section or "").lower().startswith(("data", "method", "experiment")) else 1, sc[1].page))
    prof.datasets = [verify_claim(s, s, paper, chunks, key_terms=[]) for s, _ in ds[:3]]
    res = [(s, c) for s, c in sents if RESULT_PAT.search(s) and NUM_PAT.search(s)]
    res.sort(key=lambda sc: (0 if (sc[1].section or "").lower().startswith(("result", "conclusion", "abstract")) else 1, sc[1].page))
    prof.results = [verify_claim(s, s, paper, chunks, key_terms=[]) for s, _ in res[:3]]
    lim = [(s, c) for s, c in sents if LIMIT_PAT.search(s)]
    lim.sort(key=lambda sc: (0 if (sc[1].section or "").lower().startswith(("limit", "discussion", "conclusion", "future")) else 1, -sc[1].page))
    prof.limitations = [verify_claim(s, s, paper, chunks, key_terms=[]) for s, _ in lim[:4]]
    return prof


def _llm_profile(paper: Paper, chunks: list[Chunk]) -> PaperProfile:
    ctx = "\n\n".join(f"[p.{c.page}] {c.text}" for c in sorted(chunks, key=lambda c: c.page)[:40])[:24000]
    out = llm.json(
        "You extract structured facts from ONE research paper. For every item give a short `claim` and a `quote` copied "
        "VERBATIM from the provided text (one sentence). Never invent content; omit an item if the text does not support it.",
        f"Paper: {paper.title}\n\nTEXT:\n{ctx}\n\nReturn JSON: {{\"objective\": {{\"claim\":\"\",\"quote\":\"\"}}, "
        "\"methods\": [{\"claim\":\"\",\"quote\":\"\"}], \"datasets\": [...], \"metrics\": [...], \"results\": [...], \"limitations\": [...]}")
    out = out or {}

    def conv(item):
        return verify_claim(item.get("claim", ""), item.get("quote"), paper, chunks, key_terms=[]) if isinstance(item, dict) else None

    prof = PaperProfile(paper_id=paper.paper_id, title=paper.title, authors=paper.authors, year=paper.year)
    prof.objective = conv(out.get("objective") or {})
    for f in ("methods", "datasets", "metrics", "results", "limitations"):
        setattr(prof, f, [v for v in (conv(i) for i in (out.get(f) or [])[:6]) if v])
    return prof


def methodology_agent(state: WorkflowState) -> WorkflowState:
    with traced(state, "Methodology Agent") as t:
        profiles: list[PaperProfile] = []
        for pid in state.get("ranked_papers", []):
            paper = corpus.papers[pid]
            chunks = corpus.chunks_for_paper(pid)
            prof = None
            if state.get("mode") == "llm":
                try:
                    prof = _llm_profile(paper, chunks)
                except Exception as e:
                    state.setdefault("limitations", []).append(f"LLM extraction failed for '{paper.title}' ({e}); used heuristics.")
            profiles.append(prof or _heuristic_profile(paper, chunks))
        state["profiles"] = profiles
        n = sum(len(p.methods) + len(p.datasets) + len(p.metrics) + len(p.results) + len(p.limitations) + (1 if p.objective else 0)
                for p in profiles)
        t["summary"] = f"Extracted {n} cited facts from {len(profiles)} papers (objective, methods, datasets, metrics, results, limitations)."
        if state.get("mode") == "heuristic":
            state.setdefault("limitations", []).append(
                "Heuristic extraction mode (no LLM key): facts are pattern-matched sentences, not paraphrased understanding; "
                "recall is limited to the built-in method/metric lexicons.")
    return state
