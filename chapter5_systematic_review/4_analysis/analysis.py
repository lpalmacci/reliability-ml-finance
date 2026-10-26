"""Reproduce every count, percentage and table reported in the review.

Usage (from the root of the replication package):
    python 4_analysis/analysis.py

Requires only the Python 3 standard library. Reads
    3_extraction/coding_sheet.csv
    2_screening/scopus_records.csv, 2_screening/fulltext_assessment.csv
and writes
    5_tables/results.md                 all numbers quoted in the text
    5_tables/table_S1_included_studies.csv / .tex
    5_tables/table_S2_reference_set.csv / .tex
    5_tables/table_S3_fulltext_exclusions.csv / .tex
    5_tables/main_tables.tex            Tables 3.2, 4.1-4.3, 6.1-6.2 of the chapter
Classification rules for metric families, fidelity and utility are defined
below (RULES) and documented in 3_extraction/codebook.md.
"""
import csv, os, re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(BASE, *a)
rows = list(csv.DictReader(open(P("3_extraction", "coding_sheet.csv"), encoding="utf-8")))
inc = [r for r in rows if r["role"] == "included"]
ref = [r for r in rows if r["role"] == "reference_set"]
assert len(inc) == 137 and len(ref) == 15
N = len(inc)
pct = lambda k, n=N: round(100 * k / n) if n else 0
out = []
def say(*a): out.append(" ".join(str(x) for x in a))

# ------------------------------------------------------------------ RULES
def rx(p): return lambda r: bool(re.search(p, r["metrics"], re.I))
METRIC_FAMILIES = [  # (label, rule) -- order of Table 6.1
    ("Stylized-fact diagnostics", lambda r: r["stylized_facts_tested"] != "none"),
    ("Distances between distributions", rx(r"\bDB\b|\bKS\b|MMD|Wasserstein|\bW1?\b|DTW|\bKL\b|\bJS\b|FID|EMD|Fr[eé]chet|surface distance|precision/recall|marginal")),
    ("Model-based scores", rx(r"\bMB\b|discriminat|predictive score|\bTSTR\b|DA score")),
    ("Path-space distances", rx(r"PATH|Sig-|sig-MMD|RS-W1")),
    ("Adapted and causal distances", rx(r"CAUS|AW2|adapted|causal W")),
    ("Probabilistic scoring rules", rx(r"PROB|CRPS|NLL|log-lik|likelihood|perplexity")),
    ("Point-forecast errors", rx(r"RMSE|\bMAE\b|\bMSE\b|MAPE|accuracy|\bR2\b|\bF1\b")),
    ("Risk metrics", rx(r"RISK|VaR|\bES\b|CVaR")),
    ("Decision metrics", rx(r"\bDEC\b|Sharpe|profit|PnL|P&L|ROI|hedg|utility|Calmar|drawdown|MDD")),
    ("Visualisation", rx(r"\bVIS\b|t-SNE|\bPCA\b")),
    ("Efficiency", rx(r"\bEFF\b|time|scalab")),
]
MF = dict(METRIC_FAMILIES)
def fidelity(r):  # stylized facts or any distance between real and generated data
    return MF["Stylized-fact diagnostics"](r) or MF["Distances between distributions"](r) or \
           MF["Path-space distances"](r) or MF["Adapted and causal distances"](r)
def utility(r):   # a risk measure or the outcome of a decision computed with generated data
    return MF["Risk metrics"](r) or MF["Decision metrics"](r)

DOMAINS = [("synthetic_data_augmentation", "SD"), ("forecasting", "FC"), ("microstructure_lob", "MS"),
           ("risk_pricing", "RP"), ("portfolio_strategy", "PS"), ("evaluation_study", "EV")]
FAMILIES = [("adversarial", "Adversarial"), ("diffusion", "Diffusion and score-based"), ("vae_autoencoder", "VAE and autoencoder"),
            ("hybrid", "Hybrid"), ("energy_based", "Energy-based and probabilistic circuits"),
            ("autoregressive", "Autoregressive (RNN, Transformer, SSM)"), ("multiple_families", "Several families compared"),
            ("distance_minimising", "Distance-minimising (signature, OT)"), ("normalizing_flow", "Normalizing flows"),
            ("evaluation_method", "Evaluation method")]
Q = ["Q1_data", "Q2_code", "Q3_classical_benchmark", "Q4_deep_benchmark", "Q5_temporal_oos", "Q6_stylized_facts", "Q7_runs_uncertainty"]
classical = lambda r: r["Q3_classical_benchmark"] in ("Y", "P")

# ------------------------------------------------------------------ PRISMA flow
sc = list(csv.DictReader(open(P("2_screening", "scopus_records.csv"), encoding="utf-8")))
ft = list(csv.DictReader(open(P("2_screening", "fulltext_assessment.csv"), encoding="utf-8")))
say("## PRISMA flow")
say("Scopus records:", len(sc), "| retained:", sum(x["screening_decision"] == "retained" for x in sc),
    "| of which second pass:", sum(x["note"] == "retained at second-pass check" for x in sc))
db = [x for x in ft if x["channel"] == "databases"]; om = [x for x in ft if x["channel"] == "other_methods"]
say("Database reports sought:", len(db), "| not retrieved:", sum(x["outcome"] == "not retrieved" for x in db),
    "| excluded:", dict(Counter(x["code"] for x in db if x["outcome"] == "excluded")),
    "| included:", sum(x["outcome"] == "included" for x in db))
say("Other-methods reports:", len(om), "| excluded:", dict(Counter(x["code"] for x in om if x["outcome"] == "excluded")),
    "| included:", sum(x["outcome"] == "included" for x in om))
inc_reports = [x for x in ft if x["outcome"] == "included"]
say("Included reports:", len(inc_reports), "| distinct studies:", len(set(x["study_id"] for x in inc_reports)))

# ------------------------------------------------------------------ Section 4
say("\n## Corpus profile (Section 4)")
yr = Counter(int(r["year"]) for r in inc)
say("Years:", dict(sorted(yr.items())), "| <=2019:", sum(v for k, v in yr.items() if k <= 2019),
    "| 2020-21:", yr[2020] + yr[2021], "| >=2024:", sum(v for k, v in yr.items() if k >= 2024))
fam = Counter(r["family"] for r in inc)
say("Families:", {k: f"{v} ({pct(v)}%)" for k, v in fam.most_common()})
PERIODS = [("2017--2019", 2017, 2019), ("2020--2021", 2020, 2021), ("2022", 2022, 2022), ("2023", 2023, 2023),
           ("2024", 2024, 2024), ("2025--2026", 2025, 2026)]
grp = lambda f: f if f in ("adversarial", "diffusion", "vae_autoencoder") else "other"
tab_years = []
for lab, a, b in PERIODS:
    ss = [r for r in inc if a <= int(r["year"]) <= b]; c = Counter(grp(r["family"]) for r in ss)
    tab_years.append((lab, c["adversarial"], c["diffusion"], c["vae_autoencoder"], c["other"], len(ss)))
peer = Counter(r["peer_reviewed"] for r in inc)
say("Peer reviewed:", dict(peer), f"({pct(peer['yes'])}%)")
say("ICAIF studies:", sum(bool(re.search(r"ICAIF|International Conference on AI in Finance", r["venue"])) for r in inc),
    "| Quantitative Finance:", sum(r["venue"].startswith("Quantitative Finance") for r in inc))
mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0
qs = lambda ss: mean([float(r["Q_score"]) for r in ss])
say("Mean Q:", qs(inc), "| peer reviewed:", qs([r for r in inc if r["peer_reviewed"] == "yes"]),
    "| not peer reviewed:", qs([r for r in inc if r["peer_reviewed"] == "no"]))
say("Mean Q by period: <=2021", qs([r for r in inc if int(r["year"]) <= 2021]), "| 2022-23", qs([r for r in inc if 2022 <= int(r["year"]) <= 2023]),
    "| >=2024", qs([r for r in inc if int(r["year"]) >= 2024]))
ac = Counter(a.strip() for r in inc for a in r["asset_class"].split(";") if a.strip())
say("Asset classes:", dict(ac.most_common()))
say("Frequencies:", dict(Counter(r["frequency"] for r in inc).most_common()))
say("S&P 500 in data:", sum(bool(re.search(r"S&P|SPX", r["data"])) for r in inc), "| LOBSTER:", sum("LOBSTER" in r["data"] for r in inc),
    "| Google 'Stocks' dataset: corpus", sum(bool(re.search(r"Google|\bStocks\b|GOOG", r["data"])) for r in inc),
    "reference set", sum(bool(re.search(r"Google|Stocks|stockv|Stock\b", r["data"])) for r in ref))
tab_fd = []
for f, lab in FAMILIES:
    c = [sum(r["family"] == f and r["domain"] == d for r in inc) for d, _ in DOMAINS]
    tab_fd.append((lab, *c, sum(c)))
tab_q = []
LABQ = ["Data public or fully described", "Code released", "Classical benchmark", "Deep generative benchmark",
        "Temporal out-of-sample split", "Stylized-fact diagnostics", "Several runs or uncertainty"]
for q, lab in zip(Q, LABQ):
    vals = [pct(sum(r[q] == "Y" for r in inc))]
    for d, _ in DOMAINS[:5]:
        ss = [r for r in inc if r["domain"] == d]; vals.append(pct(sum(r[q] == "Y" for r in ss), len(ss)))
    tab_q.append((f"{q.split('_')[0]} {lab}", *vals))
q5 = Counter(r["Q5_temporal_oos"] for r in inc)
say("Q5 (temporal OOS):", dict(q5), "| applicable:", N - q5["NA"])
say("Q3 classical benchmark Y/P:", sum(r["Q3_classical_benchmark"] == "Y" for r in inc), sum(r["Q3_classical_benchmark"] == "P" for r in inc))
say("Studies with Q_score = 7:", [r["study_id"] for r in inc if float(r["Q_score"]) >= 7])

# ------------------------------------------------------------------ Section 3 / 6.2: stylized facts
say("\n## Stylized facts (Table 3.2, Section 6.2)")
sf = Counter(s for r in inc for s in r["stylized_facts_tested"].split(";") if s.startswith("SF"))
for k in [f"SF{i}" for i in range(1, 12)]: say(k, sf[k], f"({pct(sf[k])}%)")
none = sum(r["stylized_facts_tested"] == "none" for r in inc); say("None tested:", none, f"({pct(none)}%)")
for d, ab in DOMAINS[:5]:
    ss = [r for r in inc if r["domain"] == d]
    say(ab, "SF tested:", sum(r["stylized_facts_tested"] != "none" for r in ss), "/", len(ss),
        f"({pct(sum(r['stylized_facts_tested'] != 'none' for r in ss), len(ss))}%)")

# ------------------------------------------------------------------ Section 6.1 / 6.3
say("\n## Metric families (Table 6.1)")
tab_m = [(lab, sum(f(r) for r in inc), sum(f(r) for r in ref)) for lab, f in METRIC_FAMILIES]
for t in tab_m: say(t[0], "corpus", t[1], f"({pct(t[1])}%)", "| reference set", t[2], "/ 15")
say("\n## Fidelity and utility (Table 6.2)")
tab_c = []
for d, ab in DOMAINS[:5]:
    ss = [r for r in inc if r["domain"] == d]; n = len(ss)
    f = sum(map(fidelity, ss)); u = sum(map(utility, ss)); b = sum(fidelity(r) and utility(r) for r in ss)
    tab_c.append((ab, n, pct(f, n), pct(u, n), pct(b, n), pct(sum(map(classical, ss)), n)))
f = sum(map(fidelity, inc)); u = sum(map(utility, inc)); b = sum(fidelity(r) and utility(r) for r in inc)
tab_c.append(("All", N, pct(f), pct(u), pct(b), pct(sum(map(classical, inc)))))
for t in tab_c: say(t)
say("Fidelity:", f, "| utility:", u, "| both:", b, "| fidelity only:", f - b, f"({pct(f-b)}%)", "| utility only:", u - b, f"({pct(u-b)}%)",
    "| neither:", N - f - u + b, f"({pct(N-f-u+b)}%)")
mb = MF["Model-based scores"]
b2 = sum((fidelity(r) or mb(r)) and utility(r) for r in inc)
say("Robustness, counting model-based scores as fidelity: both =", b2)
say("Classical benchmark (Y or P):", sum(map(classical, inc)), f"({pct(sum(map(classical, inc)))}%)")
for lab, val in [("peer-reviewed studies only", "yes"), ("preprints only", "no")]:
    ss = [r for r in inc if r["peer_reviewed"] == val]; n = len(ss)
    say(f"Robustness, {lab} (n = {n}): fidelity", f"{pct(sum(map(fidelity, ss)), n)}%", "| utility", f"{pct(sum(map(utility, ss)), n)}%",
        "| both", f"{pct(sum(fidelity(r) and utility(r) for r in ss), n)}%")

# ------------------------------------------------------------------ write results and LaTeX
os.makedirs(P("5_tables"), exist_ok=True)
open(P("5_tables", "results.md"), "w", encoding="utf-8").write("# Numbers reported in the review\n\n" + "\n".join(out) + "\n")

def tex_escape(s):
    return (s.replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
             .replace("^", "\\^{}").replace("~", "\\~{}").replace("$", "\\$"))
L = []
L.append("% Generated by 4_analysis/analysis.py -- do not edit by hand.\n")
L.append("% Table 4.1: studies by period and family\n" + "\n".join(" & ".join(map(str, t)) + " \\\\" for t in tab_years))
L.append("% Table 4.2: family x domain\n" + "\n".join(" & ".join(map(str, t)) + " \\\\" for t in tab_fd))
L.append("% Table 4.3: quality by domain (% fully satisfied)\n" + "\n".join(" & ".join(map(str, t)) + " \\\\" for t in tab_q))
L.append("% Table 6.1: metric families (corpus, reference set)\n" + "\n".join(f"{t[0]} & {t[1]} & {t[2]} \\\\" for t in tab_m))
L.append("% Table 6.2: fidelity / utility / both / classical benchmark\n" + "\n".join(" & ".join(map(str, t)) + " \\\\" for t in tab_c))
open(P("5_tables", "main_tables.tex"), "w", encoding="utf-8").write("\n\n".join(L) + "\n")

DOMAIN_LABEL = {"synthetic_data_augmentation": "SD", "forecasting": "FC", "microstructure_lob": "MS",
                "risk_pricing": "RP", "portfolio_strategy": "PS", "evaluation_study": "EV"}
FAM_LABEL = {"adversarial": "GAN", "diffusion": "Diffusion", "vae_autoencoder": "VAE/AE", "hybrid": "Hybrid",
             "energy_based": "EBM/PC", "autoregressive": "AR", "multiple_families": "Several", "distance_minimising": "Distance",
             "normalizing_flow": "Flow", "evaluation_method": "Metric"}
SHORT_VENUE = {"yoon2019timegan": "NeurIPS 2019", "pei2021rtsgan": "IEEE ICDM 2021", "srinivasan2022tstgan": "arXiv preprint 2022",
               "liu2023timetransformer": "arXiv preprint 2023", "zhou2023ls4": "ICML 2023", "naiman2024kovae": "ICLR 2024",
               "desai2021timevae": "arXiv preprint 2021", "chen2024sdformer": "NeurIPS 2024", "alaa2021fourier": "ICLR 2021",
               "yuan2024diffusionts": "ICLR 2024", "lim2023tsgm": "arXiv preprint 2023", "naiman2024imagentime": "NeurIPS 2024",
               "galib2024fide": "NeurIPS 2024", "li2025padts": "AAAI 2025", "remlinger2022cegen": "AAAI 2022"}
NAMES = {"yoon2019timegan": "TimeGAN", "pei2021rtsgan": "RTSGAN", "srinivasan2022tstgan": "TsT-GAN", "liu2023timetransformer": "Time-Transformer",
         "zhou2023ls4": "LS4", "naiman2024kovae": "KoVAE", "desai2021timevae": "TimeVAE", "chen2024sdformer": "SDformer",
         "alaa2021fourier": "Fourier Flows", "yuan2024diffusionts": "Diffusion-TS", "lim2023tsgm": "TSGM", "naiman2024imagentime": "ImagenTime",
         "galib2024fide": "FIDE", "li2025padts": "PaD-TS", "remlinger2022cegen": "CEGEN"}
def families_used(r): return "; ".join(lab for lab, f in METRIC_FAMILIES if f(r))
# Table S1
s1 = []
for r in inc:
    s1.append({"ID": r["study_id"], "Reference": f'{r["authors"]} ({r["year"]})', "Title": r["title"], "Venue": r["venue"],
               "Peer reviewed": r["peer_reviewed"], "Family": r["family_detail"], "Domain": DOMAIN_LABEL.get(r["domain"], ""),
               "Data": r["data"], "Frequency": r["frequency"], "Metric families": families_used(r),
               "Fidelity": "yes" if fidelity(r) else "no", "Utility": "yes" if utility(r) else "no",
               "Stylized facts": r["stylized_facts_tested"], "Classical benchmark": r["classical_baselines"],
               "Q1-Q7": " ".join(r[q] for q in Q), "Q score": r["Q_score"], "citekey": r["citekey"]})
with open(P("5_tables", "table_S1_included_studies.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(s1[0].keys())); w.writeheader(); w.writerows(s1)
T = ["% Generated by 4_analysis/analysis.py -- do not edit by hand.",
     "\\begin{longtable}{p{1.0cm}p{3.2cm}p{1.6cm}p{0.7cm}p{3.6cm}p{1.1cm}p{1.6cm}p{0.7cm}}",
     "\\caption{Included studies (137). Domain: SD synthetic data, FC forecasting, MS microstructure, RP risk and pricing, PS portfolio and strategies, EV evaluation study. F/U: fidelity and utility measured (Section~\\ref{subsec:gap}). Q: quality score out of 7.}\\label{tab:S1}\\\\",
     "\\toprule ID & Study & Family & Dom. & Data & SF & F/U & Q \\\\ \\midrule \\endfirsthead",
     "\\toprule ID & Study & Family & Dom. & Data & SF & F/U & Q \\\\ \\midrule \\endhead",
     "\\bottomrule \\endfoot"]
for r in inc:
    fu = ("F" if fidelity(r) else "-") + "/" + ("U" if utility(r) else "-")
    sfx = r["stylized_facts_tested"].replace("SF", "").replace(";", ",") if r["stylized_facts_tested"] != "none" else "--"
    T.append(f'{r["study_id"]} & \\citet{{{r["citekey"]}}} & {FAM_LABEL[r["family"]]} & {DOMAIN_LABEL.get(r["domain"],"")} & '
             f'{tex_escape(r["data"][:60])} & {sfx} & {fu} & {r["Q_score"]} \\\\')
T.append("\\end{longtable}")
open(P("5_tables", "table_S1_included_studies.tex"), "w", encoding="utf-8").write("\n".join(T) + "\n")
# Table S2
s2 = [{"ID": r["study_id"], "Model": NAMES.get(r["citekey"], ""),
       "Reference": f'{r["authors"]} ({r["year"]})', "Venue": r["venue"], "Venue (short)": SHORT_VENUE.get(r["citekey"], ""), "Family": r["family"], "Datasets": r["data"],
       "Financial data": "yes" if re.search(r"Google|Stocks|stockv|Stock\b|FRED|Exchange", r["data"]) else "no",
       "Metric families": families_used(r), "Stylized facts": r["stylized_facts_tested"], "citekey": r["citekey"]} for r in ref]
with open(P("5_tables", "table_S2_reference_set.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(s2[0].keys())); w.writeheader(); w.writerows(s2)
T = ["% Generated by 4_analysis/analysis.py -- do not edit by hand.",
     "\\begin{longtable}{p{0.7cm}p{2.0cm}p{2.6cm}p{2.1cm}p{1.5cm}p{5.0cm}}",
     "\\caption{Methodological reference set: general-purpose time-series generators excluded under EC7 (Section~\\ref{subsec:eligibility}), with the datasets on which they are evaluated.}\\label{tab:S2}\\\\",
     "\\toprule ID & Model & Study & Venue & Family & Datasets \\\\ \\midrule \\endfirsthead",
     "\\toprule ID & Model & Study & Venue & Family & Datasets \\\\ \\midrule \\endhead", "\\bottomrule \\endfoot"]
for r in ref:
    T.append(f'{r["study_id"]} & {NAMES.get(r["citekey"], "")} & \\citet{{{r["citekey"]}}} & {SHORT_VENUE.get(r["citekey"], "")} & {FAM_LABEL[r["family"]]} & {tex_escape(r["data"])} \\\\')
T.append("\\end{longtable}")
open(P("5_tables", "table_S2_reference_set.tex"), "w", encoding="utf-8").write("\n".join(T) + "\n")
# Table S3
ex = [x for x in ft if x["outcome"] in ("excluded", "not retrieved") and x["code"] != "EC7"]
with open(P("5_tables", "table_S3_fulltext_exclusions.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["report_title", "channel", "outcome", "code", "reason"]); w.writeheader()
    for x in ex: w.writerow({k: x[k] for k in ["report_title", "channel", "outcome", "code", "reason"]})
T = ["% Generated by 4_analysis/analysis.py -- do not edit by hand.",
     "\\begin{longtable}{p{8.2cm}p{1.2cm}p{4.0cm}}",
     "\\caption{Reports excluded after full-text assessment or not retrieved, with reasons (PRISMA item 16b). The fifteen general-purpose generators excluded under EC7 are listed in Table~\\ref{tab:S2}. Theses are not identified, in line with the decision not to cite grey literature.}\\label{tab:S3}\\\\",
     "\\toprule Report & Code & Reason \\\\ \\midrule \\endfirsthead", "\\toprule Report & Code & Reason \\\\ \\midrule \\endhead",
     "\\bottomrule \\endfoot"]
for x in sorted(ex, key=lambda x: (x["outcome"] != "excluded", x["code"], x["report_title"])):
    t = "Thesis (title withheld)" if x["code"] == "EC5" else tex_escape(x["report_title"])
    T.append(f'{t} & {x["code"] or "NR"} & {tex_escape(x["reason"])} \\\\')
T.append("\\end{longtable}")
open(P("5_tables", "table_S3_fulltext_exclusions.tex"), "w", encoding="utf-8").write("\n".join(T) + "\n")
print("\n".join(out))
