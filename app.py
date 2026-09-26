from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Compound Activity Screener", page_icon="💊", layout="centered")

# Data files are read from the folder this script lives in, not from wherever
# the app happens to be launched.
HERE = Path(__file__).resolve().parent

SEED = 42
FEATURES = ["molecular_weight", "binding_affinity", "solubility_score", "toxicity_score"]
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
# nilotinib was developed from imatinib's structure and is not in the library,
# so it makes a good first example
NILOTINIB = "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1"


@st.cache_resource(show_spinner="Training the model...")
def build_model():
    """Rebuild the notebook's final model.

    Same generated data, same split, same Random Forest. It trains in about a
    second, and unlike a saved model file it cannot break when the server
    installs a different scikit-learn version.
    """
    X, y = make_classification(n_samples=500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, random_state=SEED)
    X = pd.DataFrame(X, columns=FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = Pipeline([("scaler", StandardScaler()),
                      ("model", RandomForestClassifier(n_estimators=400, random_state=SEED))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
    }

    perm = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                  n_repeats=20, random_state=SEED)
    importance = pd.Series(perm.importances_mean, index=FEATURES)

    ranges = {f: (float(X_train[f].min()), float(X_train[f].max()), float(X_train[f].median()))
              for f in FEATURES}
    return model, metrics, importance, ranges, len(X_train), len(X_test)


@st.cache_data
def load_library():
    path = HERE / "candidate_library.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def pretty(name):
    return name.replace("_", " ").title()


def ranked_bars(frame, value, label, x_title, colour=None):
    """Horizontal bars sorted by value, largest at the top.

    st.bar_chart orders categories alphabetically, which is wrong for a ranking.
    """
    enc = {
        "x": alt.X(f"{value}:Q", title=x_title),
        "y": alt.Y(f"{label}:N", sort="-x", title=None),
        "tooltip": [alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=".3f")],
    }
    if colour:
        enc["color"] = alt.Color(
            f"{colour}:N", title=None,
            scale=alt.Scale(domain=["kinase inhibitor", "other drug"], range=["#2a7ab0", "#a0a4aa"]),
            legend=alt.Legend(orient="bottom"),
        )
    else:
        enc["color"] = alt.value("#2a7ab0")
    return alt.Chart(frame).mark_bar().encode(**enc).properties(height=alt.Step(24))


model, metrics, importance, RANGES, n_train, n_test = build_model()

st.title("Compound Activity Screener")
st.write(
    "Two ways to prioritise compounds for testing. The **classifier** predicts whether a "
    "compound is active from four descriptors. The **similarity screen** ranks real drugs by "
    "how closely they resemble imatinib, using RDKit Morgan fingerprints. "
    "Capstone project for the BiotechTrek AI in Healthcare & Drug Discovery bootcamp."
)
st.caption(
    f"Model: Random Forest  |  test accuracy {metrics['accuracy']:.2f}  |  "
    f"test ROC-AUC {metrics['roc_auc']:.3f}"
)

tab_single, tab_batch, tab_similar, tab_about = st.tabs(
    ["Single compound", "Batch screening", "Similarity screen", "About the model"]
)

# ---------------- single compound ----------------
with tab_single:
    st.subheader("Enter descriptor values")
    st.caption("Slider ranges come from the training data.")

    values = {}
    col1, col2 = st.columns(2)
    for i, f in enumerate(FEATURES):
        lo, hi, med = RANGES[f]
        target = col1 if i < 2 else col2
        values[f] = target.slider(pretty(f), round(lo, 2), round(hi, 2), round(med, 2), step=0.01)

    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.50, 0.05,
        help="Lower it to keep more candidates: fewer missed actives, more false hits.",
    )

    if st.button("Predict", type="primary"):
        p = float(model.predict_proba(pd.DataFrame([values]))[0, 1])

        if p >= threshold:
            st.success(f"Likely ACTIVE  ({p:.1%} probability)")
        else:
            st.error(f"Likely INACTIVE  ({p:.1%} probability of being active)")
        st.progress(p)

        if abs(p - threshold) < 0.15:
            st.warning("This compound is close to the decision threshold. Treat the call with caution.")

# ---------------- batch ----------------
with tab_batch:
    st.subheader("Screen a list of compounds")
    st.write("Upload a CSV with these four columns:")
    st.code(", ".join(FEATURES), language="text")

    sample_path = HERE / "sample_compounds.csv"
    if sample_path.exists():
        st.download_button("Download an example file", sample_path.read_bytes(),
                           "sample_compounds.csv", "text/csv")

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                numeric = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = numeric.isnull().any(axis=1)
                if bad.any():
                    st.warning(f"Skipped {int(bad.sum())} row(s) with missing or non-numeric values.")

                out = batch.loc[~bad].copy()
                if out.empty:
                    st.error("No usable rows in this file.")
                else:
                    out["prob_active"] = model.predict_proba(numeric.loc[~bad])[:, 1].round(3)
                    out["prediction"] = np.where(out["prob_active"] >= 0.5, "Active", "Inactive")
                    out = out.sort_values("prob_active", ascending=False).reset_index(drop=True)

                    n_active = int((out["prediction"] == "Active").sum())
                    st.write(f"**{n_active} of {len(out)}** compounds predicted active "
                             "(threshold 0.5), ranked by probability:")
                    st.dataframe(out)
                    st.download_button("Download results", out.to_csv(index=False),
                                       "screening_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

# ---------------- similarity screen ----------------
with tab_similar:
    st.subheader("Rank compounds by similarity to imatinib")
    st.write(
        "A Morgan fingerprint records the small substructures in a molecule. Tanimoto "
        "similarity compares two fingerprints: bits shared divided by bits set in either, "
        "from 0 to 1."
    )

    library = load_library()
    if library is None:
        st.info("candidate_library.csv was not found. Run the notebook to create it.")
    else:
        ranked = library.sort_values("tanimoto_to_imatinib", ascending=False)
        others = ranked.drop(index="Imatinib", errors="ignore")

        st.dataframe(ranked[["tanimoto_to_imatinib", "class", "MolWt", "LogP", "TPSA",
                             "Lipinski_violations"]].round(3))
        chart_data = others.rename_axis("compound").reset_index()
        st.altair_chart(ranked_bars(chart_data, "tanimoto_to_imatinib", "compound",
                                    "Tanimoto similarity to imatinib", colour="class"))

        kin = others[others["class"] == "kinase inhibitor"].tanimoto_to_imatinib.mean()
        oth = others[others["class"] != "kinase inhibitor"].tanimoto_to_imatinib.mean()
        st.caption(
            f"Kinase inhibitors average {kin:.2f} similarity to imatinib, against {oth:.2f} for "
            "the unrelated drugs. The values are low because these are separate chemical "
            "series, not analogues. The ordering is what matters."
        )

        st.divider()
        st.subheader("Score your own compound")
        query = st.text_input(
            "SMILES string", value=NILOTINIB,
            help="The default is nilotinib, a later drug developed from imatinib's structure. "
                 "It is not in the library. Paste any SMILES to replace it.",
        )

        if st.button("Compare to imatinib", type="primary"):
            try:
                # Scoring only needs the RDKit core. Drawing needs extra system
                # libraries, so it is imported separately further down.
                from rdkit import Chem, DataStructs, RDLogger
                from rdkit.Chem import rdFingerprintGenerator
                RDLogger.DisableLog("rdApp.*")
            except ImportError as exc:
                st.error(f"RDKit could not be loaded on this server: {exc}")
            else:
                mol = Chem.MolFromSmiles(query.strip()) if query.strip() else None
                if mol is None:
                    st.error("That is not a valid SMILES string. Check it and try again.")
                else:
                    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                    score = DataStructs.TanimotoSimilarity(
                        gen.GetFingerprint(mol), gen.GetFingerprint(Chem.MolFromSmiles(IMATINIB))
                    )
                    st.metric("Tanimoto similarity to imatinib", f"{score:.3f}")

                    # is this molecule already in the library? compare canonical SMILES
                    canon = Chem.MolToSmiles(mol)
                    match = None
                    for name, smi in library.get("smiles", pd.Series(dtype=str)).items():
                        ref = Chem.MolFromSmiles(str(smi))
                        if ref is not None and Chem.MolToSmiles(ref) == canon:
                            match = name
                            break

                    if match == "Imatinib":
                        st.write("This is **imatinib** itself, the reference compound.")
                    else:
                        pool = others.drop(index=[match]) if match else others
                        position = int((pool.tanimoto_to_imatinib > score).sum()) + 1
                        if match:
                            st.write(f"This is **{match}**, already in the reference library. "
                                     f"It sits at position **{position} of {len(others)}**.")
                        else:
                            st.write(f"Added to the reference library, it would sit at position "
                                     f"**{position} of {len(others) + 1}**.")

                    try:
                        from rdkit.Chem import Draw
                        st.image(Draw.MolToImage(mol, size=(360, 270)), caption="Your compound")
                    except Exception:
                        st.caption("Structure drawing is not available on this server.")

# ---------------- about ----------------
with tab_about:
    st.subheader("Model summary")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c1.metric("Recall (actives)", f"{metrics['recall']:.2f}")
    c2.metric("Precision", f"{metrics['precision']:.2f}")

    st.write("**Which descriptors matter** (permutation importance on the test set)")
    imp_data = pd.DataFrame({"descriptor": [pretty(f) for f in importance.index],
                             "importance": importance.values})
    st.altair_chart(ranked_bars(imp_data, "importance", "descriptor",
                                "Drop in ROC-AUC when shuffled"))

    st.markdown(
        f"""
**Classifier data.** 500 simulated compounds from scikit-learn's `make_classification`, as
the project brief specifies. All four features are informative. The names describe what each
feature stands for in a real screen, but the values are not computed from molecules.

**Training.** Stratified 80/20 split made before any preprocessing, scaling inside the
pipeline, five models compared by 5-fold cross-validation on the {n_train} training compounds,
the best two tuned, and the winner tested once on {n_test} held-out compounds. The model is
rebuilt from the same seed each time the app starts, so it is identical to the notebook's.

**Similarity screen data.** 17 real approved drugs. Descriptors are calculated from their
actual structures with RDKit, and fingerprints are Morgan, radius 2, 2048 bits.

**Limitations.** The classifier metrics come from simulated data, so they show that the
workflow is sound, not that the model works on real chemistry. For that reason the classifier
is not applied to the real molecules in the similarity screen. Training on real molecules
needs measured activity data such as IC50 values from ChEMBL.

This is a learning project, not a tool for real drug development decisions.
"""
    )
