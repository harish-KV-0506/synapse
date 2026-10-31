"""Generate a SYNTHETIC demo corpus (5 short PDFs) so SYNAPSE can be demoed offline.

IMPORTANT: these are fabricated for software testing only. Titles, authors, numbers and findings are NOT real
publications and must never be cited. Each PDF carries a watermark line saying so.
Usage: python scripts/make_sample_pdfs.py ../sample_papers
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

NOTICE = "SYNTHETIC DEMO DOCUMENT - fictional content for software testing. Not a real publication."

PAPERS = [
    {
        "file": "demo_lstm_rain_fade.pdf",
        "title": "Demo Study A: LSTM Prediction of Rain Attenuation on a Ka-band Satellite Link",
        "authors": "A. Rao, B. Iyer", "year": 2021,
        "sections": [
            ("Abstract", "This paper proposes an LSTM model to predict rain attenuation on a Ka-band satellite link. We evaluate the model on a single-site beacon dataset and show that it outperforms the ITU-R P.618 baseline in RMSE."),
            ("Introduction", "Rain fade is the dominant impairment for Ka-band satellite communication. Accurate attenuation forecasts enable adaptive coding and modulation. This study aims to improve short-term forecasting using recurrent neural networks."),
            ("Data", "The dataset consists of beacon measurements collected over 18 months at a single ground station, sampled every 1 second. Rain gauge records were collected in parallel and aligned with the beacon signal. The data were split chronologically into training and test sets."),
            ("Methodology", "We use a two-layer LSTM with 64 hidden units, trained with the Adam optimizer. Input features include rain rate, elevation angle, and past attenuation values. The model is compared with the ITU-R P.618 empirical model and a linear regression baseline."),
            ("Results", "The LSTM achieves an RMSE of 0.82 dB, which improves on the ITU-R P.618 baseline RMSE of 1.54 dB. Correlation coefficient reaches 0.93 on the test set. Performance degrades for attenuation events above 15 dB where training samples are scarce."),
            ("Limitations", "The evaluation is limited to a single site, and generalization to other locations was not evaluated. Peak attenuation events are rare in the dataset, so event-level errors remain high."),
        ],
    },
    {
        "file": "demo_xgboost_multisite.pdf",
        "title": "Demo Study B: Gradient Boosting for Multi-site Rain Attenuation Estimation",
        "authors": "C. Menon, D. Patel, E. Singh", "year": 2022,
        "sections": [
            ("Abstract", "We investigate gradient boosting with XGBoost for rain attenuation estimation across five ground stations in tropical and temperate climates. Cross-site validation shows moderate performance loss on unseen locations."),
            ("Introduction", "Satellite link planning requires attenuation statistics for many locations. This work examines whether a single machine learning model can serve multiple sites without retraining."),
            ("Data", "The dataset combines beacon measurements from five stations, covering 3 years in total. Reanalysis humidity and temperature were added as features. Rain gauge data were available at three of the five sites."),
            ("Methodology", "An XGBoost regressor is trained using leave-one-site-out cross-validation to test transferability across unseen sites. Hyperparameters were tuned by Bayesian optimization. We compare against a random forest and the ITU-R P.618 model."),
            ("Results", "XGBoost reaches an RMSE of 1.05 dB on seen sites and 1.61 dB on held-out sites, improving over the ITU-R P.618 model, which obtains an RMSE of 2.10 dB on held-out sites. MAE is reduced by 31 percent relative to the random forest."),
            ("Limitations", "Extreme events above 20 dB are underrepresented, and the model degrades under heavy rain. Uncertainty estimates were not produced; only point predictions are available."),
        ],
    },
    {
        "file": "demo_physics_hybrid.pdf",
        "title": "Demo Study C: A Physics-Informed Hybrid Model for Link Margin Forecasting",
        "authors": "F. Kumar, G. Bose", "year": 2023,
        "sections": [
            ("Abstract", "This study introduces a hybrid forecasting approach that combines the ITU-R rain model with a gated recurrent unit (GRU) residual learner. Probabilistic prediction intervals are produced via quantile regression."),
            ("Introduction", "Pure data-driven models can violate physical constraints. We develop a hybrid model that embeds physics-based attenuation estimates into a neural network, and we address predictive uncertainty explicitly."),
            ("Data", "We use a multi-year beacon dataset of 4 years from two stations at Ka-band and Ku-band. Seasonal variations were studied through year-round evaluation."),
            ("Methodology", "The hybrid approach adds a GRU residual term to the ITU-R P.618 estimate and is trained with a quantile loss for 90 percent prediction intervals. Evaluation uses RMSE, outage probability, and calibration of prediction intervals."),
            ("Results", "The hybrid model achieves an RMSE of 0.71 dB and reduces outage probability error by 24 percent compared with the physics baseline. Prediction interval coverage reaches 89 percent for the nominal 90 percent interval."),
            ("Limitations", "Real-time on-board deployment was not investigated, and inference latency was not measured. Only two stations were used, which limits conclusions about generalization to other regions."),
        ],
    },
    {
        "file": "demo_cnn_transformer.pdf",
        "title": "Demo Study D: Transformer Models for Satellite Link Quality Prediction",
        "authors": "H. Das, I. Nair", "year": 2024,
        "sections": [
            ("Abstract", "This paper evaluates a Transformer model for predicting link quality (SNR) under rain. Results show that the Transformer does not improve over a simple LSTM in heavy rain conditions."),
            ("Introduction", "Attention-based architectures have shown strong results in sequence modelling. We examine whether they offer benefits for satellite signal-to-noise ratio forecasting during heavy rain."),
            ("Data", "The dataset is a public benchmark of SNR logs recorded at one Ku-band terminal over 8 months. The dataset is publicly available and the code is open source on GitHub."),
            ("Methodology", "We train a Transformer encoder with 4 attention heads and compare it with an LSTM and a CNN. Metrics are RMSE, MAE, and R2. Feature importance was analysed with SHAP for interpretability."),
            ("Results", "The Transformer obtains an RMSE of 1.12 dB, which is worse than the LSTM at 0.98 dB during heavy rain. The Transformer does not improve peak attenuation prediction and fails to generalize to a held-out month."),
            ("Limitations", "Only one terminal was used, so site generalization was not tested. Inference cost of the Transformer is higher, but real-time performance was not evaluated."),
        ],
    },
    {
        "file": "demo_ensemble_survey.pdf",
        "title": "Demo Study E: Ensemble Learning for Rain Fade Mitigation: An Empirical Comparison",
        "authors": "J. Reddy, K. Joshi", "year": 2020,
        "sections": [
            ("Abstract", "We compare random forest, support vector regression, and an ANN ensemble for rain fade prediction using measurements from two Ku-band links. The ensemble improves accuracy over individual learners."),
            ("Introduction", "Fade mitigation techniques need fast, accurate predictions. Ensemble methods can reduce variance. This work aims to quantify the benefit of ensembling for short-term attenuation prediction."),
            ("Data", "Measurements were recorded over 12 months on two Ku-band links using a beacon receiver and a disdrometer. The dataset includes 2.1 million samples."),
            ("Methodology", "A random forest, SVR and a multilayer perceptron are combined via stacking. Evaluation uses RMSE, MAPE and the coefficient of determination R2 with k-fold cross-validation."),
            ("Results", "The ensemble achieves an RMSE of 0.95 dB, improving on the best single learner by 12 percent. R2 reaches 0.91. Seasonal differences were observed, with larger errors in the monsoon months."),
            ("Limitations", "Cross-validation was performed randomly rather than by time or site, which may overestimate performance. Extreme rain events remain poorly predicted, and the study covers only one year."),
        ],
    },
]


def make(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("t", parent=ss["Title"], fontSize=15, leading=19, alignment=0)
    h2 = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12, spaceBefore=6, spaceAfter=2)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10.5, leading=14)
    small = ParagraphStyle("s", parent=ss["BodyText"], fontSize=8, textColor="#666666")
    for p in PAPERS:
        doc = SimpleDocTemplate(str(out / p["file"]), pagesize=A4, title=p["title"], author=p["authors"],
                                leftMargin=50, rightMargin=50, topMargin=55, bottomMargin=55)
        flow = [Paragraph(p["title"], h1), Paragraph(f"{p['authors']} - {p['year']}", body),
                Paragraph(NOTICE, small), Spacer(1, 10)]
        for h, text in p["sections"]:
            flow += [Paragraph(h, h2), Paragraph(text, body), Spacer(1, 6)]
        doc.build(flow)
    print(f"Wrote {len(PAPERS)} synthetic PDFs to {out}")


if __name__ == "__main__":
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "../sample_papers"))
