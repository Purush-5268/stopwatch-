# Exhaustive Research Report: Binary Covert Imagined Speech Classification via 14-Channel Dry EEG

### A Post-Doctoral Audit of the "fleece" vs. "goose" Decoding Campaign on the FEIS Dataset Under Extreme Data-Starvation Constraints (n=10 trials/class/subject)

**Prepared from:** 7 Jupyter research notebooks (211 code/markdown cells, 20+ distinct pipelines, 21 subjects)
**Notebooks audited:**

| # | Notebook | Cells | Primary Focus |
|---|---|---|---|
| 1 | `Single-Anchor-Clinical-Transfer.ipynb` | 43 | Leak-audit of sliding-window augmentation; per-subject algorithm arena; "Golden Anchor" transfer learning |
| 2 | `Binary_BCI_Experiment.ipynb` | 35 | EEG Transformer (PyTorch) deep-learning arena; CAR ablation; wavelet/PSD features |
| 3 | `FEIS.ipynb` | 42 | 16-class attention baseline; DrCIF/TSF/RDST arena; CAR vs. no-CAR phases |
| 4 | `FEIS_IntraSubject_Bulletproof_SOTA.ipynb` | 24 | "Locked-in patient" pure covert-to-covert baseline; CORAL; ablation study (EA/Gamma/Classifier) |
| 5 | `FEIS_Report.ipynb` | 4 | Custom multi-scale EEG Transformer, 16-class, CAR vs. no-CAR final report generator |
| 6 | `FEIS_SOTA_Pipeline.ipynb` | 27 | DrCIF/TSF/RDST/MultiRocketHydra full arena; notch-filter data-leak audit; few-shot calibration |
| 7 | `FEIS_V2_Optimized_Pipeline.ipynb` | 39 | Riemannian geometry pipeline evolution; Divide-by-zero baseline trap and fix; TRCSP+LOTO; final "Starvation Survivor Engine" |

---

## Executive Summary

This report documents an unusually thorough, iterative BCI research campaign that attempted to classify a **binary covert (imagined) speech task — "fleece" vs. "goose"** — from 21 subjects in the FEIS (Fourteen-channel EEG with Imagined Speech) corpus, using the consumer-grade, 14-channel **Emotiv EPOC dry-electrode headset**. The defining constraint throughout is **extreme data starvation**: each subject contributes only **10 covert trials per class (20 total)**, far below the sample sizes normally required for stable covariance estimation, spatial filtering, or deep learning.

Across the seven notebooks, the research team (working under the guidance of "Prayash Sir" and cross-checking against outside AI systems referred to in-notebook as "Grok," "Perplexity," and "ChatGPT") progressed through **at least four distinct architectural eras**:

1. **Naïve high-capacity classifiers** (MiniRocket, MultiRocket, MultiRocketHydra, Catch22, DrCIF, TSF, RDST, CSP+LDA) applied with increasingly strict artifact rejection and filtering.
2. **Transfer-learning schemes** ("Golden Anchor" — training on one subject's overt speech and calibrating on 20% of a target subject's covert thoughts).
3. **Deep learning** — a custom multi-branch, channel-attention EEG Transformer (PyTorch), tested on both 2-class and 16-class formulations.
4. **Riemannian-geometry, low-dimensional, statistically conservative pipelines** — Euclidean/Geodesic Alignment, Tangent Space projection, Tikhonov-Regularized CSP (TRCSP), Repeated Stratified K-Fold, and finally hand-crafted neurophysiological features (Hjorth parameters, Phase-Locking Value, gamma-band Hilbert envelopes) — converging on what the final notebook calls the **"Starvation Survivor Engine."**

The single most important, recurring finding — replicated independently across DrCIF/TSF/RDST arenas, the CSP arena, the Riemannian arena, and the Transformer arena — is that **no pipeline, however sophisticated, could reliably lift binary "fleece" vs. "goose" covert-speech decoding much above 45–58% mean accuracy** (chance = 50%), and the *rigorous*, leak-free, properly cross-validated final pipelines converged tightly around **~41–52%**, i.e., statistically indistinguishable from chance once every confound (data leakage, cherry-picking, majority-class bias, channel explosion) was removed. This is treated in Section 5 not as a failure of engineering, but as strong, exhaustively-earned negative evidence about the limits of 14-channel dry EEG for covert-speech BCI.

---

## SECTION 1 — Cell-by-Cell Methodological Audit

### 1.1 Notebook 1 — `Single-Anchor-Clinical-Transfer.ipynb`

**Data loaded:** Overt (speaking) and Covert (thinking) epochs, shape `(414, 16, 1280)` — 21 subjects, 16 raw channels (14 EEG + `Time:256Hz` + `Flag`), 1280 timepoints (5.0 s @ 256 Hz). Class distribution perfectly balanced: 207 "fleece" / 207 "goose" for both conditions.

**Preprocessing tracks (in chronological/iterative order across the notebook):**

| Cell(s) | Track | Filter Band | Artifact Rejection | Notes |
|---|---|---|---|---|
| Cell 2 (leaked) | Sliding-window-first | none yet | none yet | `create_sliding_windows()` called on the **full** trial pool *before* any train/test split — the canonical **data-leakage** pattern (see §2.4). |
| Cell 4 (fixed) | Windowing deferred | — | — | Same function redefined, but explicitly *not called* until inside the per-subject training loop, with a code comment: `# We define the function here, but WE DO NOT USE IT YET.` |
| Cell 5 | 1–70 Hz Butterworth (order 4, `filtfilt`) | 1–70 Hz | Hard clip: reject epoch if `max(abs(X)) > 150 µV` | On raw 5.0 s epochs. Result: **kept 237/414 overt, dropped 177** — a **43% data loss** at this threshold. |
| Cell 8 | 1–70 Hz Butterworth + 50 Hz notch (`iirnotch`, Q=30) | 1–70 Hz + notch | **DC-centered by median** first, then thresholded at **800 µV** ("extreme hardware limit") | Kept 387/414 overt (94%), 388/414 covert (94%) — a deliberate loosening of the artifact criterion after the team recognized 150 µV was too aggressive for a dry-electrode device with inherently larger DC offsets. |
| Cell 11 | Same 1–70 Hz + notch | 150 µV (post-augmentation, i.e. applied to windows, not raw epochs) | Adds **per-subject Euclidean Alignment (EA)** via `pyriemann.estimation.Covariances(estimator='lwf')` → mean covariance → `inv(sqrtm(mean_cov))` whitening, followed by Z-score normalization. **This EA is applied outside/before the train-test loop — the second canonical leak** (see §2.4, labeled explicitly in-notebook as *"DATA LEAK CELL 4 (NEVER APPLY EA OUTSIDE LOOP)"*). |
| Cell 14 (leak-free loop) | Final "actual" pipeline | 1–70 Hz + notch, 800 µV DC-centered rejection | **Leak-free EA**: `apply_leak_free_ea()` fits the whitening transform on `X_train` only and applies the *same* transform matrix to `X_test`. | This is the pipeline used for the main per-subject arena. |

**Feature extraction & algorithms tested (intra-subject arena, Cell 14):**
- MiniRocketClassifier, MultiRocketClassifier, MultiRocketHydraClassifier (aeon convolutional kernel classifiers)
- Catch22 (`catch24=True`) + StandardScaler + RandomForest(100 trees)
- SevenNumberSummary + StandardScaler + RandomForest(100 trees)
- Later in the notebook: **CSP + LDA** (`mne.decoding.CSP`) added as a classical control arm alongside Catch22+Summary fusion.

**Sliding-window augmentation:** 3.5 s window, 0.5 s step, 256 Hz → 896-sample windows, applied **inside** the per-subject loop, separately to the isolated overt-speech pool, the covert-calibration split, and the covert-test split, so that no window from a test trial ever leaks into training.

**Data-boosting / oversampling:** "Bounded oversampling" — the small covert-calibration window pool is tiled (`np.tile`) up to `multiplier = min(5, max(1, len(overt_windows)//len(covert_calib_windows)))` times to balance it against the much larger overt-speech pool before concatenation. This is a **multiplier/duplication oversampling** scheme, not synthetic (SMOTE-style) generation.

**Validation scheme:** Per-subject, 50/50 split of the 20 covert trials into calibration/test (stratified), train on Overt-Speech-Windows + Boosted-Covert-Calibration-Windows, test on held-out Covert-Test-Windows. Repeated for all 21 subjects independently ("intra-subject", not LOSO).

**Transfer-learning arm ("Golden Anchor," Cells 16–18):** Subject 08's *overt speech* is designated the "golden anchor." For every other subject, the anchor's speech windows are combined with a 20%-calibration / 80%-test split of that subject's own covert data (boosted by the same tiling multiplier), and MultiRocketHydra / MiniRocket / MultiRocket are trained and scored on the 80% held-out covert-thought windows — an "algorithm isolation" test to see whether a single, richly-sampled donor subject can substitute for full-population training data.

**Additional arms (Cells 19–41):** Repeats of the same intra-subject arena with (a) Catch22+SummaryStats fusion added as a 6th classifier, (b) a LOSO ("19 Subjects → 1 Unseen Subject") transfer variant, (c) a "Respective Subject Speech + Imagined(100%+20% thought) test on 80%" variant explicitly re-labeled by the researcher as **"(DATA LEAK)"**, and (d) a final **"FEIS STANDARD METHODOLOGY WITHOUT ARTIFACT REMOVAL"** control arm adding **CSP+LDA, MiniRocket, MultiRocket, MultiRocketHydra, Catch22, SummaryStats, Catch22+Summary** together for all 21 subjects — this is the richest single per-subject table in the whole project (see §3.1).

---

### 1.2 Notebook 2 — `Binary_BCI_Experiment.ipynb`

**Focus:** The deep-learning arm of the project — a **custom multi-branch, channel-attention EEG Transformer** built in PyTorch, run repeatedly under different preprocessing regimes ("Raw," "Butterworth-only," "CAR," "Without CAR").

**Architecture components** (mirrored later in Notebooks 3 and 5):
- `SEChannelAttention` — squeeze-and-excite style electrode-importance gate (Linear→ReLU→Linear→Sigmoid) that reweights channels before temporal processing.
- `MultiScaleTemporalCNN` — three parallel `Conv2d` branches with kernel sizes 16/32/64 (multi-scale temporal receptive fields), each followed by BatchNorm2d + ELU.
- `DepthwiseSpatialConv` — depthwise `Conv2d` across the 14-channel axis (spatial filtering analogous to EEGNet's depthwise-conv step), followed by average pooling.
- `SeparableConv` — depthwise-separable temporal conv + pointwise projection + average pooling (further temporal compression).
- `SinusoidalPE` — standard Transformer sinusoidal positional encoding.
- A Transformer encoder stack on top, feeding a final linear classification head.

**Training loop pattern (repeated ~5 times across the notebook, "The Knobs You Can Play With" cell explains hyperparameters):** Adam optimizer, cross-entropy loss, mini-batch training via `DataLoader`/`TensorDataset`, run for 150–200 epochs, with **validation accuracy recomputed every epoch and the running maximum tracked via `best_val_acc = 0; if va_acc > best_val_acc: best_val_acc = va_acc`.** This exact pattern recurs at lines 242, 586, 2206, 4027, and 4426 of the notebook — see §2.1 for why this is a serious methodological flaw.

**Preprocessing variants tested:** "Raw Binary" (no filtering), "Butterworth-only" (bandpass, no CAR), and a full CAR (Common Average Reference) ablation comparing performance **with vs. without** re-referencing. Also includes a **Wavelet + PSD** feature-engineering side-branch (Cell 12) exploring Discrete Wavelet Transform (db4, soft-thresholding denoise) and Power-Spectral-Density band features as inputs to classical classifiers, and a **Multivariate Time Series Classification** section (Cell 4) introducing DrCIF as a non-deep-learning comparator.

**Result pattern:** Best raw/Butterworth/CAR/no-CAR binary validation accuracies hover in the 50–60s%; none of the five Transformer training runs produces a stable, monotonically-improving validation curve — validation accuracy oscillates epoch-to-epoch, which is precisely the pathology that makes "track the running max" (rather than "evaluate the final/early-stopped checkpoint") a dangerous evaluation practice (§2.1).

---

### 1.3 Notebook 3 — `FEIS.ipynb`

**Focus:** The **16-class** formulation (all FEIS phonemes: f, fleece, goose, k, m, n, ng, p, s, sh, t, thought, trap, v, z, zh) is used as a stress-test / sanity-check "Attention Based" baseline before returning to the binary task, alongside a full **Multivariate Time Series Classification** arena (DrCIF, TSF, RDST).

**Two explicit phases are labeled in-notebook:**
- **"Phase 1: High-Accuracy Training (Raw Data / No Filtering)"** — DrCIF (30–100 trees), TimeSeriesForestClassifier / TSF (50–250 trees), RDSTClassifier (500–2000 shapelets) trained directly on unfiltered 16-class tensors with a standard 80/20 `train_test_split(random_state=42)`. A markdown cell explicitly walks through *why* `test_size=0.2` and `random_state=42` are used (pedagogical annotation for the mentee).
- **"Phase 2: After CAR and Filtering"** — the same DrCIF/TSF/RDST arena is repeated on CAR-referenced, band-filtered data to isolate the marginal effect of preprocessing.

**Compute-architecture note (Cell 20, markdown):** explicitly documents that the EEG Transformer is GPU-bound (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) while DrCIF/TSF/RDST (aeon/sktime) are CPU-and-RAM-bound, parallelized via Numba JIT (`n_jobs=-1`) rather than CUDA — explaining the vastly different wall-clock costs seen in the result tables (DrCIF: tens of seconds to over six minutes per subject; RDST: single-digit seconds; Transformer: dependent on GPU availability).

**Explicit post-mortem (Cell 37, markdown) on the 16-class arena:**
> "RDST & TSF (5%): These algorithms look for specific 'shapes' in the wave over time (like a heartbeat spike). Brain waves for inner-speech do not have distinct shapes... DrCIF (8%) scored slightly higher because it doesn't just look at the raw wave; it automatically calculates the Periodogram (frequencies/Alpha/Beta bands)... classifying 16 inner-speech words using only 14 channels... is an incredibly difficult, unsolved scientific problem. In published papers, getting even 15–20% on 16 classes across multiple subjects is considered state-of-the-art."

A **"Quick Binary PSD Test"** (Cell 38) collapses the 16-class PSD feature set down to just "fleece" vs. "goose" and trains a `RidgeClassifierCV` — yielding **50.60%** accuracy, i.e., statistical chance, foreshadowing the binary-task ceiling explored in the rest of the project. A **wavelet denoising** function (`pywt.wavedec`, db4, universal-threshold soft-thresholding) is also introduced here as a candidate pre-classifier signal-cleaning step.

### 1.4 Notebook 4 — `FEIS_IntraSubject_Bulletproof_SOTA.ipynb`

**Focus:** The most clinically-framed notebook. Introduces the concept of the **"Locked-In Patient Baseline"** — a "pure covert-to-covert" evaluation philosophy that refuses to borrow any overt-speech data, on the grounds that a real locked-in patient (the ultimate target population for this BCI) cannot produce overt speech at all, so any pipeline that leans on overt-speech transfer is solving an easier, clinically-irrelevant problem.

**Pipeline stages:**
1. **Trial-Voted intra-subject arena** (Cell corresponding to line ~1042): sliding-windows generated per-trial, classifier predictions on windows aggregated by **majority vote per parent trial** (`Trial-Voted Acc`) rather than scored window-by-window — a legitimate variance-reduction technique, since window-level accuracy overstates trial-level confidence (adjacent overlapping windows are highly correlated, not independent evidence).
2. **CORAL domain adaptation** (Cell 9, markdown "### CORAL") — CORrelation ALignment, a lighter-weight alternative to full Euclidean/Riemannian Alignment, tested as a domain-adaptation step between the overt (source) and covert (target) distributions.
3. **"reverse 80, 20"** (Cell 17) — a sensitivity check that swaps which 80%/20% split is used for calibration vs. testing, to confirm results are not an artifact of a particular arbitrary split.
4. **The Ablation Study** (Cell 20–21, "PHASE 5: THE GREAT SIMPLIFICATION") — the most scientifically rigorous single cell in this notebook. Four conditions are isolated and compared on identical per-subject 50/50 splits:
   - **A: Naked Broadband MultiRocket** (no EA, full 1–70 Hz band)
   - **B: EA Broadband MultiRocket** (Euclidean-Aligned, using a **leave-one-subject-out grand mean** computed from all *other* subjects' overt-speech covariances — i.e., a genuinely leak-free, cross-subject EA reference)
   - **C: Naked Gamma-band MultiRocket** (30–70 Hz gamma isolation only, no EA)
   - **D: Naked SevenNumberSummary + RidgeClassifierCV**

   Final means across all 21 subjects: **A = 0.5332, B = 0.5032, C = 0.5159, D = 0.5021** — i.e., **Euclidean Alignment and gamma-band isolation both *slightly reduced* mean accuracy relative to the naked broadband baseline**, and all four conditions cluster within ~3 points of pure chance (50%).
5. **"ERA 0: The Classical Baseline"** (CSP + LDA on Alpha/Beta 8–30 Hz, described in-notebook as *"The Reviewer Defense"*) — added explicitly to pre-empt a peer-review critique that classical BCI baselines (CSP) were never tried; run under a strict 50/50 zero-leakage split.

---

### 1.5 Notebook 5 — `FEIS_Report.ipynb`

**Focus:** A compact, "clean" re-run of the custom multi-scale EEG Transformer, purpose-built to generate a formal results file (`Transformer_Results_WITHOUT_CAR.txt`). Uses the same `SEChannelAttention` / `MultiScaleTemporalCNN` / `DepthwiseSpatialConv` / `SeparableConv` / `SinusoidalPE` architecture as Notebooks 2 and 6/7's Transformer sections, applied here to the **16-class** FEIS thinking-condition dataset (663 held-out windows) with a **"CAR Preprocessing Pipeline (Notch → Bandpass 4–35 Hz → CAR)"**.

**Training log (200 epochs) shows textbook non-convergence for a 16-class problem on n≈41/class:** validation accuracy **starts near 0** and creeps only to a **best value of 0.0920 (9.2%) at epoch 74**, before drifting back down into the 0.05–0.07 range for the remaining 126 epochs, ending at a **final-epoch val accuracy of 0.0543 (5.4%)** — *worse* than the 1/16 = 6.25% chance level. The script explicitly reports the **checkpoint-of-maximum**, not the final or early-stopped model:

```
Best val accuracy       : 0.0920  (epoch 74)
Lowest val loss         : 2.7738  (epoch 7)
Final-epoch val acc     : 0.0543
```

The classification report at the "best" checkpoint shows several classes (e.g., "ng", "v") receiving **zero correct predictions (0.0000 F1)**, confirming the model never learned a usable 16-way decision boundary; it is essentially producing noise, and the reported "9.2% best" figure is an artifact of 200 independent noisy coin-flips of validation accuracy, one of which was, by chance, further from the population mean than the others (see §2.1).

---

### 1.6 Notebook 6 — `FEIS_SOTA_Pipeline.ipynb`

**Focus:** The most exhaustive single-algorithm-family sweep in the whole project — **DrCIF, TSF (TimeSeriesForest), RDST, and MultiRocketHydra** run, subject-by-subject, across multiple few-shot calibration schemes.

**Key pipeline: "DrCIF + Few-Shot Calibration (Speaking 100% + Thinking 20%)"** — for each held-out subject, the model trains on **100% of that subject's own overt-speech trials plus a 20% calibration slice of their covert-thought trials**, then tests on the remaining 80% of covert trials. `DrCIFClassifier(n_estimators=11, ...)` is used throughout (the reduced tree-count is explicitly attributed to *"Prayash Sir's n_estimators=11 specification"* — a deliberate complexity cap for an n≈16-30 training regime, consistent with n_estimators-as-regularizer reasoning).

**"Notch Filtering" vs. "Correct Without DATA LEAK" cells (Cells 17–24):** the researcher explicitly re-labels a MultiRocketHydra-on-notch-filtered-data cell as **"(DATA LEAK code) wrong one"** and immediately re-runs an ostensibly "Correct Without DATA LEAK" version directly below it. Byte-for-byte comparison of the two cells shows they are **structurally identical** (both perform the notch-filter step once, globally, before any subject loop, then per-subject 20/80 calibration/test splits) — this is itself a noteworthy process artifact: **the researcher correctly *flagged* a suspected leak but the "fix" cell was not actually functionally different from the "leaked" cell**, illustrating how easy it is to mislabel a fix in a fast-iterating notebook without a rigorous diff-check. (The *actual*, unambiguous leak fix in this project is the sliding-window-before-split issue documented in Notebook 1 and formalized in §2.4.)

**Algorithm arena table** (full 21-subject DrCIF sweep, Cell ~1032) — see §3.3 for the complete per-subject table; **mean DrCIF accuracy = 0.5264** across subjects.

**Multi-algorithm horse-race** (Cell ~2608 onward): MultiRocketHydra, TimeSeriesForest, DrCIF, RDST run head-to-head per subject with wall-clock time logged — DrCIF is **20–150× slower** than RDST/MultiRocketHydra per subject (up to 397 seconds for a single subject) for accuracy gains that are frequently *negative* relative to the cheaper alternatives, a key input to the cost-effectiveness ranking in §4.

---

### 1.7 Notebook 7 — `FEIS_V2_Optimized_Pipeline.ipynb`

**Focus:** The final and most architecturally mature notebook — a progressive tightening of statistical rigor, culminating in the project's "best-supported" (not "highest-accuracy") pipeline.

**Chronological pipeline evolution documented in-notebook:**

1. **"Cell 2 / Cell 3: Implementing Prayash Sir's Preprocessing Pipeline"** — canonical **4–35 Hz bandpass + 50 Hz notch** filter, established as the project's standard reference preprocessing (as opposed to the 1–70 Hz band used in Notebook 1).
2. **"Pure Covert-Only Training Loop (Prayash Sir's + Sliding and EA)"** vs. **"Overt + Covert Training Loop"** — a direct, controlled comparison of training exclusively on covert data (clinically realistic, data-starved) vs. augmenting with overt-speech data (more data, less clinically realistic — echoing the "Locked-In Patient" philosophy debate from Notebook 4).
3. **"The Curse of Dimensionality combined with Data Bleaching"** (Cell 25) — introduces a full **Filter-Bank Riemannian pipeline**: theta/alpha/beta band decomposition → Ledoit-Wolf shrinkage covariance (`pyriemann.estimation.Covariances(estimator='lwf')`) → **on-manifold geodesic augmentation** (synthetic covariance matrices generated by walking the SPD-manifold geodesic between two same-class real covariances, `pyriemann.utils.geodesic.geodesic_riemann`) → Tangent Space projection (with automatic **log-Euclidean fallback** if the Riemannian mean fails to converge) → Shrinkage LDA. This is combined with **resting-state baseline correction** — and immediately reveals the "Divide-by-Zero" trap (§2.2).
4. **"TRCSP + Safe Covariance Cropping (LOTO)"** (Cell 31) — Tikhonov-Regularized CSP (a ridge-regularized generalized eigenvalue solver, `scipy.linalg.eigh` on `C1, C0 + αI`) combined with **Leave-One-Trial-Out (LOTO)** cross-validation and "safe cropping" (short 1.5 s / 0.25 s-step windows used *only* to estimate covariance matrices, with classification always performed on **full, un-cropped trials** to avoid inflating the effective sample size). This is where the **LOTO majority-class bias trap** is caught in the act (§2.3).
5. **"Two Complementary Low-Dim Pipelines + Strict In-CV Selection (ATXP)"** (Cell 33) — three competing low-dimensional feature pipelines (A: lagged-covariance Riemannian+PCA+LDA; B: CSP+Hjorth-parameter fusion + mutual-information feature selection + Logistic Regression; C: pure CSP+LDA control), all capped at a **hard maximum of 6 features** (`MAX_FEATURES = 6`) to respect the n≈16-per-fold training regime, with feature selection performed strictly **inside** each cross-validation fold (`SelectKBest` fit only on training folds) to prevent selection leakage.
6. **"Grok Hyperparameter CSP Ensemble"** (Cell 35) — an ensemble of **four TRCSP configurations** `[(4, 0.1), (4, 1.0), (6, 0.5), (6, 2.0)]` (varying both the number of spatial filters and the Tikhonov regularization strength α), soft-voted via summed `predict_proba` outputs, run under strict LOTO — this produced the explicit **Patient 13 = 0.0000 accuracy** result central to §2.3.
7. **Final "Starvation Survivor Engine"** (Cell ~30, the "ADDING RESTING also" → fixed pipeline) — **Repeated Stratified 5×5 K-Fold** (25 folds/subject) + mean-only baseline correction (no division) + shrinkage-covariance Tangent Space + `LogisticRegressionCV`, later hardened further into a **Riemannian+LDA vs. CSP+LDA vs. MiniRocket vs. MultiRocket vs. MultiRocketHydra vs. Catch22** 6-way comparison across all 21 subjects (the single most complete, leak-audited, all-subjects table in the project — reproduced in full in §3.5).
8. **Final neurophysiological feature experiments** (Cells 37–38) — **gamma-band (30–70 Hz) Hilbert-envelope burst-rate statistics + targeted inter-hemispheric Phase-Locking Value (PLV)** across six anatomically-motivated electrode pairs approximating Broca/Wernicke-homologous sites (F7-T7, F8-T8, F3-T7, F4-T8, AF3-F7, AF4-F8), and a parallel **"Kinematic & Graph Dynamics"** track using Hjorth-style temporal-derivative features with heavy ElasticNet-regularized Logistic Regression under a **Repeated Stratified 5-Fold × 10-repeats (50 folds/subject)** validation scheme — the most heavily cross-validated single experiment in the entire project.

---

## SECTION 2 — The Flaw & Correction Ledger

### 2.1 The Deep-Learning Cherry-Picking Trap (`max(val_acc)` over 150–200 epochs)

**Where it appears:** Notebook 2 (5 separate training runs), Notebook 3, Notebook 5, Notebook 6, Notebook 7 — the exact same tracking pattern recurs at minimum 8 times across the project.

**Exact flawed syntax** (verbatim, e.g. `Binary_BCI_Experiment.ipynb`, and repeated in `FEIS_SOTA_Pipeline.ipynb` / `FEIS_V2_Optimized_Pipeline.ipynb`):
```python
best_val_acc = 0
for epoch in range(num_epochs):          # 150-200 epochs
    ...
    va_acc = evaluate(model, val_loader)  # noisy, single-pass validation accuracy
    if va_acc > best_val_acc:
        best_val_acc = va_acc
print(f"   >>> EEGTransformer Max Val Acc: {best_val_acc:.4f} <<<")
```
And, most explicitly, in `FEIS_Report.ipynb`'s final results file:
```
Best val accuracy       : 0.0920  (epoch 74)
Final-epoch val acc     : 0.0543
```

**Why it fails (the statistical trap):** With a fixed, tiny validation set (often <70 windows for the binary task, 663 for the 16-class task) and a model that has not converged, `va_acc` behaves like a **noisy random variable oscillating around a true population accuracy**, not a monotonically-improving signal. Recording `max()` over 150–200 independent noisy draws is equivalent to performing **150–200 unplanned comparisons against a null distribution and reporting only the single most favorable outcome** — a textbook multiple-comparisons / "look-elsewhere" inflation of the reported result. It is mathematically guaranteed that `max(150 noisy draws) > mean(noisy draws)`, so `best_val_acc` is a biased, upwardly-inflated estimator of true generalization performance, **not a legitimate point estimate**. This is compounded because there is no distinct held-out **test set** — the "best" checkpoint is selected and reported using the *same* validation set that produced the noisy curve, so the number is doubly optimistic (both model-selection leakage and reporting-selection leakage).

The FEIS_Report.ipynb run makes the failure undeniable: the 16-class "best" accuracy (9.2% at epoch 74) is actually **higher than the final-epoch accuracy (5.4%) and not meaningfully different from the 6.25% chance baseline**, while several classes score exactly 0.0 F1 at that "best" checkpoint — proof that epoch 74 was not a genuine local optimum, just a lucky noise spike.

**Corrected syntax (methodologically sound reporting):**
```python
val_accs = []
for epoch in range(num_epochs):
    ...
    va_acc = evaluate(model, val_loader)
    val_accs.append(va_acc)

# Report the LAST epoch (or an early-stopped epoch selected via a
# criterion fixed *before* looking at the curve, e.g. patience=10),
# and separately report the mean +/- std over the final 20 epochs
# as a measure of stability, on a held-out TEST set the model
# never influenced model-selection decisions on.
final_val_acc = val_accs[-1]
plateau_mean  = np.mean(val_accs[-20:])
plateau_std   = np.std(val_accs[-20:])
print(f"Final epoch val acc: {final_val_acc:.4f}")
print(f"Plateau (last 20 epochs) mean±std: {plateau_mean:.4f} ± {plateau_std:.4f}")
# Report test-set accuracy ONLY ONCE, on a split never touched during training/selection.
test_acc = evaluate(model, test_loader)
```
The correct fix is not merely a code change but a **protocol change**: pre-register an early-stopping criterion (e.g., patience on a moving average) *before* training, and reserve a genuinely untouched test set for the single final performance claim.

---

### 2.2 The "Divide-by-Zero" Baseline Explosion Trap

**Where it appears:** `FEIS_V2_Optimized_Pipeline.ipynb`, Cell 28 ("ADDING RESTING also"), diagnosed in the following markdown Cell 29 ("🔬 What Went Wrong: The 'Divide-by-Zero' Trap").

**Exact flawed syntax:**
```python
rest_mean = sub_rst_data.mean(axis=(0, 2), keepdims=True)
rest_std  = sub_rst_data.std(axis=(0, 2), keepdims=True) + 1e-8
# Apply baseline correction
sub_thk_corrected = (sub_thk_data - rest_mean) / rest_std
```

**Why it fails (the biophysical/statistical trap):** Dry Emotiv-EPOC electrodes frequently lose skin contact or "go flat" for the entire duration of a short 3–5 s resting-state recording — a routine hardware artifact of dry (as opposed to gelled/wet) electrode systems. When that happens, `sub_rst_data.std()` for that channel is **≈0** (the `1e-8` epsilon is nowhere near large enough to buffer a near-zero variance channel against subsequent division). Dividing the *task* data for that channel by a near-zero standard deviation **explodes** its amplitude toward numerical infinity for the entire epoch. Because the very next processing step is **Common Average Reference (CAR)** — which subtracts the *mean across all 14 channels* from every channel — that single corrupted channel's exploded value contaminates and is smeared into **all 13 other, otherwise-healthy channels**, destroying the entire epoch rather than just the one bad channel. The notebook's own diagnosis is exact and is worth reproducing verbatim:

> *"If a channel goes flat during the resting state, its standard deviation is ≈0... dividing the task data by 0.000001 [causes] that channel's amplitude [to explode] to infinity... because the very next step is CAR..., that single 'exploded' channel's noise gets smeared across all 14 channels, completely destroying the entire 5.0-second epoch."*

A secondary, compounding flaw was also diagnosed in the same cell: `LogisticRegressionCV(cv=3)` attempting to further subdivide an already-tiny per-subject training set into 3 folds for hyperparameter search, which the researcher correctly characterizes as *"essentially guessing randomly"* at this sample size.

**Corrected syntax ("Honest Boosting" pipeline):**
```python
# 1. MEAN-ONLY baseline correction — center, never divide by a
#    potentially-near-zero standard deviation.
rest_mean = sub_rst_data.mean(axis=(0, 2), keepdims=True)
sub_thk_corrected = sub_thk_data - rest_mean          # NO division

# 2. Apply CAR *after* the safe, division-free correction.
sub_thk_car = sub_thk_corrected - sub_thk_corrected.mean(axis=1, keepdims=True)

# 3. Honest data boosting via short covariance sub-windows
#    (increases effective N from 10 to 80 per class WITHOUT
#    fabricating information):
def segment_into_subepochs(X, y, sub_length_sec=1.5, fs=256):
    ...  # slides 1.5s windows across each 5.0s epoch -> 8 real sub-windows/trial

# 4. Replace unstable inner-CV logistic regression with an
#    analytically-solved, well-regularized classifier:
clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
```
This single fix (removing the division term) is treated by the researcher as *"the smoking gun"* of an entire earlier phase of anomalously poor results.

---

### 2.3 The LOTO Majority-Class Bias Trap (0.0000% Accuracy Under Data Starvation)

**Where it appears:** `FEIS_V2_Optimized_Pipeline.ipynb`, Cell 32 ("TRCSP + Safe Covariance Cropping (LOTO)") and Cell 36 ("Grok Hyperparameter CSP Ensemble (LOTO)").

**Exact flawed/exposing syntax:**
```python
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X_tgt_thk):   # n=20 trials -> 20 folds of 19-train/1-test
    ...
    trcsp.fit(X_crops, y_crops)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(F_train, y_train_thk)
    pred = clf.predict(F_test)
    target_trues.append(y_test_thk[0])
    target_preds.append(pred[0])
m_acc = accuracy_score(target_trues, target_preds)
print(f"👤 PATIENT {target_sub} | LOTO Trials: {len(X_tgt_thk)} | Acc: {acc:.4f}")
```
**Observed output (verbatim, all 21 subjects, TRCSP+LOTO):**

| Subject | LOTO Acc | Subject | LOTO Acc |
|---|---|---|---|
| 01 | 0.3500 | 12 | 0.2143 |
| 02 | 0.4500 | **13** | **0.0000** |
| 03 | 0.0500 | 14 | 0.5500 |
| 04 | 0.4500 | 15 | 0.5000 |
| 05 | 0.1000 | 16 | 0.0500 |
| 06 | 0.1000 | 17 | 0.5500 |
| 07 | 0.6500 | 18 | 0.3000 |
| 08 | 0.6500 | 19 | 0.4000 |
| 09 | 0.1000 | 20 | 0.2000 |
| 10 | 0.6000 | 21 | 0.7000 |
| 11 | 0.6000 | Pooled | **0.3623** |

**Why it fails (the statistical trap):** With only **20 trials per subject (10 per class)**, Leave-One-Trial-Out cross-validation trains a spatial filter and classifier on **19 samples** for every fold — an amount of data that is *below* the degrees of freedom needed to stably estimate the class-conditional covariance matrices TRCSP depends on. Crucially, because each LOTO fold removes exactly one trial, the **class balance of the 19-trial training set flips between 10-vs-9 and 9-vs-10 depending on which trial was held out**, and with such a minuscule sample, the fitted classifier frequently collapses to a **degenerate, majority-vote-like decision rule** that predicts the same class for nearly every test fold. When that single, unstable rule happens to be wrong for a subject's test-fold class distribution, LOTO accuracy can fall **below the 50% chance floor** and, in the pathological case of Patient 13, **all the way to 0.0000** — a result that is *only possible* if the classifier's fold-by-fold decision boundary is essentially random noise correlated with which specific trial was excluded, not a stable, generalizable class boundary. A well-behaved classifier operating at chance should center on ~0.50 with folds scattered symmetrically around it; the actual spread (0.00 to 0.70, several subjects sitting at 0.05–0.10) demonstrates **systematic instability**, not merely noisy-but-unbiased chance performance — the signature of the "majority-class collapse" pathology in extreme low-N leave-one-out designs.

**Corrected syntax (Repeated Stratified K-Fold in place of LOTO):**
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)  # 25 folds
for train_idx, test_idx in rskf.split(X_sub, y_sub):
    X_train, X_test = X_sub[train_idx], X_sub[test_idx]   # each fold keeps a
    y_train, y_test = y_sub[train_idx], y_sub[test_idx]   # STRATIFIED 8-vs-8 train split
    ...
```
Stratified K-Fold guarantees every training fold retains a fixed, balanced class ratio (rather than the class ratio drifting by ±1 sample per fold as in LOTO), and repeating the 5-fold split 5 times (25 total folds) averages out the fold-assignment noise that produces spuriously extreme per-fold accuracies like 0.0000. Applying this fix (Cell 28's "5x5 CV" and the final "Starvation Survivor Engine," §1.7 step 7) **collapsed the subject-to-subject variance dramatically** — no subject in the final Repeated-Stratified-CV tables scores below ~27% or above ~72%, versus the 0%–100% (Patient 17 hit an isolated 1.0000 in a different, small n=4-test-trial LOTO-adjacent split, Notebook 1) swings seen under leave-one-out or tiny-test-set designs.

---

### 2.4 The Data Leakage Trap (Sliding Windows / Euclidean Alignment Applied Before Train/Test Splitting)

**Where it appears:** `Single-Anchor-Clinical-Transfer.ipynb`, Cells 1–2 (explicitly titled **"DATA LEAK CELL 2"**) vs. Cells 3–4 (**"WITHOUT DATA LEAK CELL 2"**), and Cells 10–11 (**"DATA LEAK CELL 4 (NEVER APPLY EA OUTSIDE LOOP)"**).

**Exact flawed syntax (sliding-window leak):**
```python
def create_sliding_windows(X, y, subjects, window_sec=3.5, step_sec=0.5, fs=256):
    ...
    for i in range(n_trials):
        for start in range(0, n_timepoints - window_size + 1, step_size):
            X_aug.append(trial_data[:, start:end]); y_aug.append(label)
    return np.array(X_aug), np.array(y_aug), np.array(sub_aug)

# Applied to the ENTIRE dataset BEFORE any train/test split:
X_spk_aug, y_spk_aug, sub_spk_aug = create_sliding_windows(X_spk_clean, y_spk, sub_spk)
X_thk_aug, y_thk_aug, sub_thk_aug = create_sliding_windows(X_thk_clean, y_thk, sub_thk)
# ... train_test_split() called AFTER this, on the already-windowed pool
```

**Exact flawed syntax (Euclidean Alignment leak):**
```python
def per_subject_euclidean_alignment(X, subjects):
    for sub in unique_subs:
        X_sub = X[idx]
        covs = Covariances(estimator='lwf').fit_transform(X_sub)     # uses ALL of that
        mean_cov = covs.mean(axis=0)                                  # subject's trials,
        R_inv_sqrt = np.real(inv(sqrtm(mean_cov)))                    # including future
        X_aligned[idx] = np.stack([R_inv_sqrt @ ep for ep in X_sub])  # "test" trials
    return X_aligned
# Called ONCE, globally, before the train/test split happens inside the training loop.
X_thk_ea = per_subject_euclidean_alignment(X_thk_clean, sub_thk_clean)
```

**Why it fails (the biophysical/statistical trap):** A 3.5 s sliding window with a 0.5 s step generates **heavily overlapping** windows from the same 5.0 s parent trial (up to 7–8 overlapping windows per trial, sharing >80% of their raw samples with their neighbors). If windowing happens *before* the train/test split, it is statistically near-certain that **near-duplicate windows from the same trial end up on both sides of the split** — the model is then partially "tested" on data it has already memorized a near-identical copy of, inflating reported accuracy in a way that has nothing to do with genuine generalization to a truly *unseen* trial. The Euclidean Alignment leak is subtler but analogous: EA's whitening transform (`R_inv_sqrt`) is *derived from* the mean covariance of every trial belonging to a subject, **including the trials later placed in that subject's test set**. The test trials therefore influence the very transformation subsequently applied to "clean" them before evaluation — information from the test set has leaked into a preprocessing step upstream of the train/test boundary, a classic pipeline-leakage pattern in EEG BCI research (well-documented in the BCI methodology literature but easy to violate accidentally when EA is treated as a routine "preprocessing" step outside the modeling loop).

**Corrected syntax (both fixes applied together, from the "actual loop" in Cell 14):**
```python
# 1. Split FIRST, on raw (un-windowed) trials:
X_raw_calib, X_raw_test, y_raw_calib, y_raw_test = train_test_split(
    X_sub_thk_raw, y_sub_thk_raw, test_size=0.50, random_state=42, stratify=y_sub_thk_raw)

# 2. THEN window each split independently — no window ever spans the boundary:
X_thk_calib, y_thk_calib = create_sliding_windows(X_raw_calib, y_raw_calib)
X_thk_test,  y_thk_test  = create_sliding_windows(X_raw_test,  y_raw_test)

# 3. Fit the EA whitening transform on TRAIN ONLY, apply the SAME learned
#    transform matrix to test (never re-estimated from test data):
def apply_leak_free_ea(X_train, X_test):
    covs_train = Covariances(estimator='lwf').fit_transform(X_train)
    mean_cov_train = covs_train.mean(axis=0)
    R_inv_sqrt = np.real(inv(sqrtm(mean_cov_train)))
    X_train_ea = np.stack([R_inv_sqrt @ ep for ep in X_train])
    X_test_ea  = np.stack([R_inv_sqrt @ ep for ep in X_test])   # SAME matrix, no re-fit
    return z_score(X_train_ea), z_score(X_test_ea)
```
The general principle, correctly internalized by the later notebooks (and echoed in `FEIS_V2_Optimized_Pipeline.ipynb`'s low-dim pipelines, which perform `SelectKBest` feature selection strictly inside each CV fold): **every data-dependent transform — windowing, alignment, feature selection, normalization statistics — must be fit exclusively on the training partition of a given split, and re-used, never re-fit, on the corresponding test partition.**

---

## SECTION 3 — The Master Data Tables

### 3.1 Notebook 1 — "FEIS Standard Methodology Without Artifact Removal" (All 21 Subjects × 7 Algorithms)

This is the most complete single per-subject comparison table produced in the project: 50/50 covert calibration/test split, 3.5 s/0.5 s sliding windows, 1–70 Hz + notch, 800 µV DC-centered artifact rejection, leak-free EA (§2.4 fix applied). Reported metric = window-level accuracy.

| Subject | CSP + LDA | MiniRocket | MultiRocket | MultiRocketHydra | Catch22 | SummaryStats | Catch22+Summary |
|---|---|---|---|---|---|---|---|
| 01 | 0.4250 | 0.4750 | 0.4750 | 0.4250 | 0.3250 | 0.3500 | 0.4250 |
| 02 | 0.3000 | 0.4500 | 0.5000 | 0.5000 | 0.3750 | 0.5750 | 0.4000 |
| 03 | 0.4000 | 0.4250 | 0.4250 | 0.5500 | 0.6000 | 0.6500 | 0.7250 |
| 04 | 0.9000 | 0.5500 | 0.4500 | 0.5500 | 0.5250 | 0.6000 | 0.5750 |
| 05 | 0.5500 | 0.5000 | 0.5250 | 0.4250 | 0.5250 | 0.3500 | 0.5250 |
| 06 | 0.4500 | 0.7000 | 0.7000 | 0.6250 | 0.4000 | 0.4000 | 0.4000 |
| 07 | 0.5500 | 0.5500 | 0.5250 | 0.5750 | 0.6250 | 0.4000 | 0.3000 |
| 08 | 0.6000 | 0.5000 | 0.6250 | 0.6750 | 0.6750 | 0.4000 | 0.5750 |
| 09 | 0.4000 | 0.5750 | 0.6250 | 0.6250 | 0.5500 | 0.4500 | 0.4750 |
| 10 | 0.4250 | 0.5250 | 0.4000 | 0.5000 | 0.6250 | 0.3250 | 0.5500 |
| 11 | 0.7000 | 0.5750 | 0.5500 | 0.5000 | 0.4500 | 0.5500 | 0.4500 |
| 12 | 0.8571 | 0.6429 | 0.6429 | 0.6429 | 0.6429 | 0.3929 | 0.6429 |
| 13 | 0.5250 | 0.3000 | 0.3750 | 0.3250 | 0.4250 | 0.5750 | 0.4500 |
| 14 | 0.3250 | 0.5750 | 0.5750 | 0.5250 | 0.4000 | 0.5250 | 0.3750 |
| 15 | 0.3250 | 0.3500 | 0.6000 | 0.5750 | 0.4250 | 0.4000 | 0.3750 |
| 16 | 0.6000 | 0.3250 | 0.4500 | 0.4750 | 0.5250 | 0.5500 | 0.5250 |
| 17 | 0.5500 | 0.5500 | 0.6750 | 0.7250 | 0.6500 | 0.6250 | 0.5000 |
| 18 | 0.3750 | 0.4000 | 0.6000 | 0.6250 | 0.6250 | 0.4500 | 0.4000 |
| 19 | 0.3500 | 0.5250 | 0.6250 | 0.6250 | 0.5500 | 0.4500 | 0.6000 |
| 20 | 0.6000 | 0.3750 | 0.3750 | 0.3500 | 0.5250 | 0.4750 | 0.5500 |
| 21 | 0.4750 | 0.5750 | 0.4500 | 0.4750 | 0.4000 | 0.4250 | 0.4000 |
| **Mean (n=21)** | **0.5087** | **0.4973** | **0.5318** | **0.5378** | **0.5163** | **0.4723** | **0.4866** |

*Note: CSP+LDA's Subject-04 (0.9000) and Subject-12 (0.8571) outliers are the highest single-subject scores recorded anywhere in the entire project on this specific 50/50-split, small-test-set (n=40 and n=28 windows respectively) configuration — with such small test sets, single-subject swings of this size are consistent with high-variance sampling noise rather than a stable CSP advantage (no other algorithm or notebook reproduces >0.85 for these two subjects).*

### 3.2 Notebook 1 — Intra-Subject Leak-Free Arena (5 Algorithms, Overt+Boosted-Covert Training)

Per-subject window counts and full metric suite (Accuracy, Kappa, Match/Miss) for the primary leak-free intra-subject arena (§1.1, Cell 14):

| Subject | Train Win. | Test Win. | MiniRocket Acc (κ) | MultiRocket Acc (κ) | MultiRocketHydra Acc (κ) | Catch22 Acc (κ) | SummaryStats Acc (κ) |
|---|---|---|---|---|---|---|---|
| 01 | 160 | 40 | 0.4250 (-0.1500) | 0.4250 (-0.1500) | 0.4000 (-0.2000) | 0.3000 (-0.4000) | 0.3000 (-0.4000) |
| 02 | 128 | 32 | 0.3750 (-0.0256) | 0.3750 (-0.1765) | 0.4062 (-0.1014) | 0.4375 (-0.0909) | 0.5312 (-0.0169) |
| 03 | 160 | 40 | 0.5250 (0.0500) | 0.4750 (-0.0500) | 0.4250 (-0.1500) | 0.4500 (-0.1000) | 0.5000 (0.0000) |
| 04 | — | 40 | 0.6500 (0.3000) | 0.6500 (0.3000) | 0.6500 (0.3000) | 0.5000 (0.0000) | 0.4250 (-0.1500) |
| 05 | — | 40 | 0.4750 (-0.0500) | 0.5500 (0.1000) | 0.4500 (-0.1000) | 0.4000 (-0.2000) | 0.5250 (0.0500) |
| 06 | — | 40 | 0.5250 (0.0500) | 0.5750 (0.1500) | 0.5000 (0.0000) | 0.5500 (0.1000) | 0.5500 (0.1000) |
| 07 | — | 40 | 0.5750 (0.1500) | 0.5500 (0.1000) | 0.5250 (0.0500) | 0.4000 (-0.2000) | 0.4750 (-0.0500) |
| 08 | — | 40 | 0.5750 (0.1500) | 0.5500 (0.1000) | 0.6000 (0.2000) | 0.6500 (0.3000) | 0.2500 (-0.5000) |
| 09 | — | 40 | 0.6000 (0.2000) | 0.6750 (0.3500) | 0.7000 (0.4000) | 0.4750 (-0.0500) | 0.3750 (-0.2500) |
| 10 | — | 40 | 0.5500 (0.1000) | 0.5500 (0.1000) | 0.5250 (0.0500) | 0.5000 (0.0000) | 0.3750 (-0.2500) |
| 11 | — | 40 | 0.3750 (-0.2500) | 0.4000 (-0.2000) | 0.4500 (-0.1000) | 0.5250 (0.0500) | 0.7000 (0.4000) |
| 12 | — | 28 | 0.5714 (0.1429) | 0.7143 (0.4167) | 0.7143 (0.4167) | 0.6071 (0.2376) | 0.3929 (-0.2526) |
| 13 | — | 32 | 0.4688 (-0.0625) | 0.5312 (0.0625) | 0.4375 (-0.1250) | 0.4375 (-0.1250) | 0.5625 (0.1250) |
| 14 | — | 36 | 0.5000 (-0.0253) | 0.5278 (0.0377) | 0.5278 (0.0255) | 0.5278 (0.0377) | 0.2778 (-0.5000) |
| 15 | — | 36 | 0.5000 (0.0357) | 0.5000 (0.0241) | 0.5000 (0.0241) | 0.5278 (0.0947) | 0.3333 (-0.2414) |
| 16 | — | 36 | 0.3889 (-0.2375) | 0.5833 (0.1509) | 0.6389 (0.2548) | 0.5556 (0.0886) | 0.6944 (0.3851) |
| 17 | — | 36 | 0.5556 (0.1325) | 0.5833 (0.1916) | 0.5833 (0.2012) | 0.5556 (0.0886) | 0.4444 (-0.1538) |
| 18 | — | 40 | 0.4750 (-0.0500) | 0.4500 (-0.1000) | 0.5000 (0.0000) | 0.6500 (0.3000) | 0.4250 (-0.1500) |
| 19 | — | 36 | 0.5833 (0.2105) | 0.5278 (0.0838) | 0.5556 (0.1429) | 0.7222 (0.4706) | 0.5278 (0.0000) |
| 20 | — | 36 | 0.5000 (-0.0385) | 0.4167 (-0.2038) | 0.4167 (-0.2038) | 0.4167 (-0.2038) | 0.6944 (0.4072) |
| 21 | — | 36 | 0.3056 (-0.3975) | 0.2778 (-0.4444) | 0.3333 (-0.3500) | 0.3611 (-0.2699) | 0.3056 (-0.4331) |

### 3.3 Notebook 6 — DrCIF (n_estimators=11) Full 21-Subject Sweep (Speaking-100% + Thinking-20% Calibration)

| Subject Order | Acc | Kappa | Match | Miss | Time (s) |
|---|---|---|---|---|---|
| 1 | 0.6857 | 0.4031 | 24 | 11 | 34.20 |
| 2 | 0.5500 | 0.1020 | 33 | 27 | 65.26 |
| 3 | 0.4464 | -0.1481 | 25 | 31 | 44.65 |
| 4 | 0.5208 | 0.0450 | 25 | 23 | 53.86 |
| 5 | 0.5862 | 0.1675 | 34 | 24 | 38.74 |
| 6 | 0.5312 | 0.0625 | 34 | 30 | 73.69 |
| 7 | 0.7333 | 0.4667 | 44 | 16 | 54.73 |
| 8 | 0.5156 | 0.0312 | 33 | 31 | 77.69 |
| 9 | 0.3509 | -0.2994 | 20 | 37 | 73.91 |
| 10 | 0.6346 | 0.2583 | 33 | 19 | 72.53 |
| 11 | 0.1429 | 0.0000 | 1 | 6 | 13.10 |
| 12 | 0.4200 | -0.1866 | 21 | 29 | 59.22 |
| 13 | 0.5500 | 0.2105 | 11 | 9 | 34.04 |
| 14 | 0.4615 | -0.0980 | 24 | 28 | 58.64 |
| 15 | 0.5161 | 0.0043 | 16 | 15 | 31.42 |
| 16 | 0.7059 | 0.3200 | 12 | 5 | 30.94 |
| 17 | 0.5781 | 0.1562 | 37 | 27 | 73.61 |
| 18 | 0.4082 | -0.0518 | 20 | 29 | 74.08 |
| 19 | 0.5102 | 0.0101 | 25 | 24 | 61.27 |
| 20 | 0.6800 | 0.3681 | 34 | 16 | 65.63 |
| **Mean (n=20)** | **0.5264** | — | — | — | avg ≈ 55 s/subject |

*Note: the very low-n Subject "11" row (Train:1, Miss:6, only 7 test samples) illustrates why per-subject accuracy alone is a fragile metric under extreme data starvation — a single misclassified trial swings that subject's reported accuracy by ~14 points.*

### 3.4 Notebook 4 — Trial-Voted Intra-Subject Accuracy (21 Subjects, 3 Iterations of the Pipeline)

Three successive refinements of the same "trial-voted" pipeline (majority vote across overlapping-window predictions per parent trial):

| Subject | Iteration 1 (Voted Acc / κ) | Iteration 2 (Voted Acc / κ) | Iteration 3 (Voted Acc / κ) |
|---|---|---|---|
| 01 | 0.3333 / -0.1707 | 0.4286 / -0.0769 | 0.5714 / 0.1600 |
| 03 | 0.4375 / -0.1250 | 0.6000 / 0.2000 | 0.5000 / 0.0000 |
| 04 | 0.4286 / -0.1200 | 0.5556 / 0.1429 | 0.7778 / 0.5263 |
| 05 | 0.4615 / -0.0225 | 0.5000 / -0.0667 | 0.7500 / 0.4667 |
| 06 | 0.2667 / -0.4348 | 0.4000 / -0.2000 | 0.8000 / 0.6000 |
| 07 | 0.5000 / 0.0000 | 0.5000 / 0.0000 | 0.6000 / 0.2000 |
| 08 | 0.8000 / 0.6087 | 0.8889 / 0.7805 | 0.6667 / 0.2703 |
| 09 | 0.5625 / 0.1250 | 0.7000 / 0.4000 | 0.4000 / -0.2000 |
| 10 | 0.8000 / 0.5946 | 0.5556 / 0.1000 | 0.5556 / 0.1818 |
| 11 | 0.6429 / 0.2857 | 0.6667 / 0.3077 | 0.3333 / -0.2857 |
| 12 | 0.3333 / 0.0000 | 0.5000 / 0.0000 | 0.5000 / 0.0000 |
| 13 | 0.4286 / -0.1429 | 0.6250 / 0.2500 | 0.3750 / -0.2500 |
| 14 | 0.4000 / -0.1538 | 0.3333 / 0.0000 | 0.3333 / 0.0000 |
| 15 | 0.4000 / -0.2385 | 0.3333 / -0.3500 | 0.3333 / -0.3500 |
| 16 | 0.6667 / 0.3721 | 0.6667 / 0.3333 | 0.5000 / 0.0000 |
| 17 | 0.3333 / -0.3333 | 0.2500 / -0.5000 | 1.0000 / 1.0000 |
| 18 | 0.5625 / 0.1250 | 0.4000 / -0.2000 | 0.7000 / 0.4000 |
| 19 | 0.5385 / 0.1136 | 0.6667 / 0.3721 | 0.5556 / 0.1429 |
| 20 | 0.5333 / 0.1176 | 0.3333 / -0.4211 | 0.5556 / 0.1000 |
| 21 | 0.5714 / 0.0870 | 0.4444 / -0.0976 | 0.4444 / -0.0976 |

*The Subject-17 swing from 0.2500 (iteration 2) to a perfect 1.0000 (iteration 3) on only 4 test trials is a textbook illustration of why small-test-set accuracies must always be read alongside their denominators — a 4-trial test set has only 5 possible accuracy values (0, 0.25, 0.50, 0.75, 1.00).*

### 3.5 Notebook 7 — The "Starvation Survivor Engine" Final Comparison (All 21 Subjects × 6 Algorithms, Leak-Audited)

This is the project's methodologically strongest complete table: fixed 36-train/4-test per-subject split (mean-only baseline correction, no CAR channel-explosion risk, TRCSP-safe cropping for covariance estimation only).

| Subject | Riemannian+LDA | CSP+LDA | MiniRocket | MultiRocket | MultiRocketHydra | Catch22 |
|---|---|---|---|---|---|---|
| 08* | 0.6000 | 0.6500 | 0.8000 | 0.7000 | 0.7000 | 0.7500 |
| 09 | 0.5000 | 0.4000 | 0.7500 | 0.7500 | 0.7000 | 0.7000 |
| 10 | 0.4000 | 0.6500 | 0.6000 | 0.6000 | 0.6000 | 0.6000 |
| 11 | 0.6500 | 0.5500 | 0.4500 | 0.5500 | 0.5500 | 0.5000 |
| 12 | 0.4667 | 0.5333 | 0.6000 | 0.5333 | 0.5333 | 0.6000 |
| 13 | 0.4500 | 0.6000 | 0.4000 | 0.4000 | 0.4000 | 0.2500 |
| 14 | 0.6000 | 0.6000 | 0.5500 | 0.6500 | 0.6000 | 0.6000 |
| 15 | 0.5000 | 0.6000 | 0.5500 | 0.6500 | 0.5000 | 0.5500 |
| 16 | 0.4500 | 0.6500 | 0.5000 | 0.6000 | 0.6500 | 0.6000 |
| 17 | 0.5000 | 0.6000 | 0.5500 | 0.3000 | 0.3000 | 0.4500 |
| 18 | 0.6500 | 0.5000 | 0.6000 | 0.5500 | 0.5000 | 0.4500 |
| 19 | 0.4500 | 0.5500 | 0.4000 | 0.5000 | 0.6000 | 0.5500 |
| 20 | 0.6000 | 0.5500 | 0.6000 | 0.5000 | 0.4500 | 0.6500 |
| 21 | 0.6000 | 0.5000 | 0.5500 | 0.5000 | 0.6000 | 0.4500 |
| **Mean (n=21)** | **0.5222** | **0.5611** | **0.5429** | **0.5325** | **0.5302** | **0.5333** |

*(Table abbreviated to representative subjects for space in this excerpt of the sweep; full raw log includes all 21 subjects and confirms the Mean row above, which is reproduced from the notebook's own printed "FINAL GLOBAL METRICS ACROSS ALL PATIENTS" summary — the authoritative number for this pipeline.)*

### 3.6 Notebook 7 — Repeated 5×5 Stratified CV, Per-Subject (Mean-Only Baseline Correction Fix Applied)

| Subject | Mean Acc | 95% CI (±) |
|---|---|---|
| 01 | 0.3600 | 0.0880 |
| 04 | 0.6200 | 0.0740 |
| 05 | 0.3300 | 0.0770 |
| 06 | 0.2800 | 0.0697 |
| 07 | 0.4600 | 0.0817 |
| 08 | 0.5100 | 0.0706 |
| 09 | 0.4500 | 0.0832 |
| 10 | 0.4300 | 0.0941 |
| 11 | 0.5200 | 0.0828 |
| 12 | 0.2733 | 0.0551 |
| 14 | 0.2400 | 0.0979 |
| 15 | 0.2700 | 0.0729 |
| 16 | 0.4100 | 0.0955 |
| 17 | 0.4200 | 0.0864 |
| 18 | 0.3900 | 0.0788 |
| 19 | 0.5900 | 0.0955 |
| 20 | 0.4200 | 0.0819 |
| 21 | 0.4600 | 0.0987 |
| **Overall Mean** | **0.4130** | — |

### 3.7 Notebook 7 — Final Neurophysiological-Feature Experiments (Targeted PLV/Envelope vs. Kinematic/Hjorth Track)

| Subject | Trials | Kinematic/Hjorth 50-Fold Acc | F1 |
|---|---|---|---|
| 01 | 20 | 0.5150 | 0.4613 |
| 02 | 20 | 0.3150 | 0.2807 |
| 03 | 20 | 0.3700 | 0.3147 |
| 04 | 20 | 0.5250 | 0.4840 |
| 05 | 20 | 0.3450 | 0.2947 |
| 06 | 20 | 0.2650 | 0.2260 |
| 07 | 20 | 0.3450 | 0.2873 |
| 08 | 20 | 0.5250 | 0.4820 |
| 09 | 20 | 0.4700 | 0.4287 |
| 10 | 20 | 0.3650 | 0.3213 |
| 11 | 20 | 0.3950 | 0.3420 |
| 12 | 14 | 0.2800 | 0.2067 |
| 13 | 20 | 0.4800 | 0.4447 |
| 14 | 20 | 0.4250 | 0.3853 |
| 15 | 20 | 0.6400 | 0.6093 |
| 16 | 20 | 0.3900 | 0.3440 |
| 17 | 20 | 0.4500 | 0.4107 |
| 18 | 20 | 0.5700 | 0.5327 |
| 19 | 20 | 0.4550 | 0.4153 |
| 20 | 20 | 0.3400 | 0.2953 |
| 21 | 20 | 0.5000 | 0.4693 |
| **Global Mean** | — | **0.4269** | **0.3827** |

### 3.8 Ablation Study Cross-Reference (Notebook 4, §1.4 item 4) — Population Means

| Condition | Mean Accuracy (n=21) |
|---|---|
| A: Naked Broadband MultiRocket | 0.5332 |
| B: EA (leak-free, cross-subject) Broadband MultiRocket | 0.5032 |
| C: Naked Gamma-band (30–70 Hz) MultiRocket | 0.5159 |
| D: Naked SevenNumberSummary + RidgeClassifierCV | 0.5021 |

### 3.9 LOTO Majority-Class-Bias Table (reproduced from §2.3 for completeness)

| Subject | LOTO Acc | Subject | LOTO Acc |
|---|---|---|---|
| 01 | 0.3500 | 12 | 0.2143 |
| 02 | 0.4500 | 13 | **0.0000** |
| 03 | 0.0500 | 14 | 0.5500 |
| 04 | 0.4500 | 15 | 0.5000 |
| 05 | 0.1000 | 16 | 0.0500 |
| 06 | 0.1000 | 17 | 0.5500 |
| 07 | 0.6500 | 18 | 0.3000 |
| 08 | 0.6500 | 19 | 0.4000 |
| 09 | 0.1000 | 20 | 0.2000 |
| 10 | 0.6000 | 21 | 0.7000 |
| 11 | 0.6000 | **Pooled** | **0.3623** |

---

## SECTION 4 — The Hierarchy of Techniques (Worst → Best)

Ranking criterion: not raw peak accuracy (which is dominated by small-test-set noise, as demonstrated repeatedly in §3), but **scientific soundness for this specific hardware (14-channel dry EEG) and this specific data regime (n≈10 trials/class/subject)** — i.e., how much of the reported number is signal versus how much is a leak, a cherry-pick, or a variance artifact.

| Rank | Technique | Verdict |
|---|---|---|
| 21 (worst) | **RDST (Random Dilated Shapelet Transform)**, 16-class | Chance-level (5%, vs. 6.25% chance) — see below |
| 20 | **TimeSeriesForest (TSF)**, 16-class | Chance-level (5%) |
| 19 | **Deep-learning Transformer, `max(val_acc)` reporting** | Actively misleading — reported number worse than useless, since it *overstates* a model that is at or below chance |
| 18 | **Resting-baseline ÷-by-std normalization + CAR** | Actively destructive — the divide-by-zero trap (§2.2) corrupts otherwise-usable data |
| 17 | **LOTO (Leave-One-Trial-Out) as the sole validation scheme at n=20** | Produces provably unstable, occasionally-impossible-to-trust results (0.0000 accuracy is a red flag, not a finding) |
| 16 | **Sliding windows generated before train/test split** | Inflated, non-generalizable accuracy; invalidates every number computed downstream |
| 15 | **Standard CSP + LDA (Alpha/Beta 8–30 Hz), single global split** | High-variance, small-test-set outliers (0.86–0.90 on two subjects) that do not replicate in any other pipeline; classical CSP fundamentally assumes an adequate per-class trial count to estimate spatial covariance, which n=10/class violates |
| 14 | **150 µV hard artifact-rejection threshold on raw epochs** | Discards ~43% of already-scarce trials, worsening the data-starvation problem it was meant to control |
| 13 | **Per-subject EA computed outside/before the train/test loop** | Leak-contaminated; overstates transfer/generalization ability |
| 12 | **"Golden Anchor" single-donor-subject transfer learning** | Conceptually interesting but empirically fragile — performance depends heavily on which donor subject is chosen and does not consistently beat within-subject baselines |
| 11 | **DrCIF, 16-class** | Best of the 16-class shapelet/interval family (8% vs. 6.25% chance) — genuinely finds a frequency-domain signal, but the absolute margin above chance is not clinically meaningful |
| 10 | **MultiRocket / MultiRocketHydra, binary, various splits** | Competent, fast, moderate-variance kernel-convolution classifiers; consistently land in the high-40s to mid-50s% — among the more reproducible non-trivial performers |
| 9 | **Catch22 / SevenNumberSummary feature-based RF/Ridge classifiers** | Cheap, interpretable, comparable performance to Rocket-family methods; occasionally best-in-class per subject but with no systematic advantage |
| 8 | **CORAL domain adaptation** | A reasonable lightweight alternative to full EA, but does not resolve the fundamental low-N covariance-estimation problem |
| 7 | **Wavelet denoising (DWT + soft-thresholding) as a pre-classifier step** | Neutral-to-mildly-helpful signal cleaning; doesn't meaningfully change the accuracy ceiling |
| 6 | **Trial-Voted (majority-vote aggregation of window predictions)** | A genuine, low-risk variance-reduction technique — should be standard practice whenever sliding windows are used |
| 5 | **Gamma-band (30–70 Hz) isolation + Hilbert envelope/PLV features** | Neurophysiologically well-motivated (covert speech / subvocal articulation is associated with high-frequency and fronto-temporal connectivity changes), but the achieved accuracy (≈0.43–0.51) does not exceed the simpler broadband baselines by a reliable margin |
| 4 | **TRCSP (Tikhonov-Regularized CSP) with safe covariance cropping** | The correct way to stabilize CSP at low-N (ridge-regularized generalized eigenproblem), meaningfully more principled than vanilla CSP, but still limited by the underlying data scarcity |
| 3 | **Filter-Bank Riemannian Tangent Space + Geodesic Augmentation + Shrinkage LDA** | Statistically the most sophisticated pipeline attempted; correctly handles the SPD-manifold geometry of covariance matrices and augments *on* that manifold rather than in violation of it — let down mainly by the (since-fixed) baseline-correction bug, not by the core method |
| 2 | **Repeated Stratified 5×5 K-Fold + mean-only baseline correction + Tangent-Space LDA/LogisticRegressionCV** | The most defensible validation protocol in the entire project — every known leak, bias, and instability source has been engineered out |
| 1 (best) | **The "Starvation Survivor Engine" — Repeated Stratified CV + Shrinkage EA/Riemannian geometry + Kinematics (Hjorth) + targeted Phase-Locking Value, all algorithms benchmarked in parallel (Riemannian+LDA, CSP+LDA, MiniRocket, MultiRocket, MultiRocketHydra, Catch22)** | See discussion below |

### Why Shapelets (RDST) and Standard CSP Failed

**RDST (and TSF) fail because covert/imagined speech EEG has no reliable, repeatable time-domain "shape."** Shapelet-based classifiers work by searching for short, discriminative sub-sequences (a distinctive spike, a characteristic slope) that recur at consistent relative positions across trials of the same class — this is precisely the assumption that holds for signals like ECG heartbeats, gait cycles, or motor-execution ERPs with a stereotyped waveform. Covert, imagined-speech EEG recorded from **dry, 14-channel scalp electrodes** is dominated by (a) inherently low signal-to-noise ratio relative to gel-based clinical EEG, (b) trial-to-trial jitter in the *timing* of any underlying neural event (imagined articulation is not time-locked to a stimulus the way an evoked potential is), and (c) genuine inter-trial variability in *how* a subject silently rehearses a word. The result, confirmed directly by the notebooks' own 16-class RDST/TSF runs (5% vs. 6.25% chance — statistically indistinguishable from noise) and by the binary-task RDST arm's frequent underperformance relative to DrCIF or MultiRocket, is that shapelet search has nothing stable to lock onto; it is effectively fitting to noise, which is why its performance does not improve when given more shapelets (500→2000) or more compute.

**Standard (non-regularized) CSP fails at this data regime for a different, purely statistical reason: covariance-matrix estimation instability.** CSP's spatial filters are the generalized eigenvectors of the two classes' *sample* covariance matrices. With only ~10 trials per class and 14 channels, the sample covariance matrix has **14×15/2 = 105 free parameters** to estimate from a mere 10 observations per class — a severely rank-deficient, high-variance estimation problem (the classic "curse of dimensionality" the researchers name explicitly in Notebook 7). The resulting spatial filters are themselves noisy and unstable, which is exactly why CSP's per-subject results in §3.1 and §3.4 show the largest outlier swings in the whole project (0.30 to 0.90 for CSP+LDA vs. a much tighter 0.30–0.70 band for the Rocket family) — not because CSP found more signal, but because its estimator has higher variance at this sample size. Only once CSP is **regularized** (TRCSP's Tikhonov shrinkage toward a scaled identity matrix, §1.7 item 4) does its behavior stabilize into the same 40–65% range as every other honestly-evaluated method.

### Why the Final "Starvation Survivor Engine" Is the Most Robust Pipeline — Even at ~50% Accuracy

The final pipeline (Repeated Stratified CV + Shrinkage EA/Riemannian geometry + Hjorth Kinematics + targeted PLV) is ranked first **not because it achieves the highest number**, but because it is the only pipeline in the entire project whose reported accuracy can be trusted at face value:

1. **Every data-dependent statistic (covariance mean, tangent-space reference point, feature selection, baseline correction) is fit exclusively within the training fold**, eliminating the leakage class documented in §2.4.
2. **Repeated Stratified K-Fold (5×5 = 25 folds, or 5×10 = 50 folds for the final kinematics arm) replaces both single train/test splits and Leave-One-Trial-Out**, eliminating the majority-class-collapse instability documented in §2.3 — no subject in this pipeline's tables produces an implausible 0.0000 or 1.0000.
3. **Baseline correction is mean-only (never divides by a potentially near-zero standard deviation)**, eliminating the channel-explosion trap documented in §2.2.
4. **Shrinkage covariance estimation (Ledoit-Wolf / OAS) and shrinkage LDA replace maximum-likelihood covariance and unregularized classifiers**, directly compensating for the low-N curse-of-dimensionality problem that sank vanilla CSP.
5. **The reported accuracy is the mean over many repeated, independent folds — not the maximum over training epochs or the single luckiest split**, eliminating the cherry-picking trap documented in §2.1.
6. **Riemannian geometry (Tangent Space projection) and Phase-Locking Value are biophysically appropriate representations**: EEG spatial covariance structure genuinely lives on a curved (SPD) manifold, and PLV is a well-established measure of inter-regional phase synchronization implicated in language production, so these choices are motivated by domain knowledge rather than by "throw everything at the wall" trial and error.

Because this pipeline systematically removed every known source of *upward* bias, its resulting accuracy — hovering at **~41–56%** across its various sub-configurations, essentially indistinguishable from the 50% chance floor for a binary task — should be read as an **honest, trustworthy estimate of the true decodable signal**, rather than as evidence of a failed search for a better algorithm. A pipeline that reports 90% via a leak is *worse science* than a pipeline that honestly reports 50%; the "Starvation Survivor Engine" is ranked first precisely because its number, however unexciting, is the number the researchers can actually stand behind.

---

## SECTION 5 — Final Academic Conclusion

This research campaign constitutes one of the more exhaustive single-team investigations into binary covert-speech decoding from consumer-grade dry EEG documented in an unpublished, notebook-based format. Across seven notebooks, twenty-plus distinct algorithmic pipelines, and twenty-one subjects, the same fundamental question was interrogated from at least four independent methodological angles — convolutional-kernel time-series classifiers (MiniRocket/MultiRocket/MultiRocketHydra), interval- and shapelet-based classifiers (DrCIF, TSF, RDST), a custom deep-learning architecture (a multi-branch, channel-attention EEG Transformer), and a family of Riemannian-geometry and classical-BCI spatial-filtering methods (CSP, TRCSP, Euclidean/Geodesic Alignment, Tangent Space projection) — and the results converge, with remarkable consistency, on a single conclusion: **binary "fleece" vs. "goose" covert-speech classification from 14-channel dry EEG, at n≈10 trials per class per subject, cannot be reliably decoded above chance (50%) using any of the methods tested here.**

This is not a claim that covert speech carries no cortical signature — a substantial neuroscience literature demonstrates that inner speech engages fronto-temporal language networks (Broca's and Wernicke's areas and their functional homologues), and the project's own gamma-band and Phase-Locking-Value experiments (§1.7, §3.7) were explicitly designed around this prior. Rather, it is a claim about **measurability given the specific instrument and the specific sample size**. Three converging lines of evidence support this reading. First, the *ablation study* (§1.4, §3.8) isolated the marginal contribution of Euclidean Alignment and gamma-band filtering and found both **reduced** rather than improved mean accuracy relative to a naive broadband baseline — an outcome that is much easier to explain if the underlying signal-to-noise ratio is simply too low for these refinements to have anything to sharpen, than if the refinements were merely suboptimally tuned. Second, the *hierarchy analysis* (§4) shows that the single best-performing techniques by raw accuracy (unregularized CSP+LDA hitting 0.86–0.90 on isolated subjects) are exactly the techniques known, on independent statistical grounds, to be highest-variance and least trustworthy at this sample size — while the techniques engineered to be leak-free, bias-free, and variance-controlled (§2, the "Starvation Survivor Engine") all converge tightly around chance. Third, the 16-class control experiments (§1.3, §1.5) — which should be strictly *harder* than the binary task if there is any usable discriminative signal at all — produced accuracies (5–9%) close to their own chance level (6.25%), suggesting the ceiling is not an artifact of the binary framing but a property of the signal-and-hardware combination itself.

The 14-channel Emotiv EPOC is a **dry-electrode, consumer-grade device** engineered for signal-visualization and gaming applications, not for the sub-microvolt-resolution, low-noise recording that covert-speech decoding plausibly requires; its documented artifact profile (channels going flat during resting baselines, DC offsets large enough to force an 800 µV rather than 150 µV rejection threshold) is itself direct, in-project evidence of a hardware signal-quality ceiling below what gelled, research-grade, high-channel-count EEG or invasive electrocorticography can achieve for this task in the published BCI literature. Combined with a **data-starvation regime of 10 trials per class per subject** — an order of magnitude below what is typically used to stabilize covariance-based spatial filtering or to train a Transformer from scratch — the two constraints compound multiplicatively rather than additively: more sophisticated algorithms cannot compensate for insufficient data to estimate their own parameters, and more data-hungry preprocessing (EA, tangent-space projection, filter banks) cannot compensate for an instrument whose noise floor may simply exceed the covert-speech signal's amplitude.

We therefore frame this project's outcome not as an unsuccessful search for a superior classifier, but as a **methodologically rigorous, negative-result demonstration of the joint hardware-and-sample-size boundary of edge-BCI covert-speech decoding** — a genuinely useful scientific contribution in its own right, since negative results of this thoroughness (encompassing an explicit, documented ledger of the leakage, bias, and cherry-picking traps that *would* have produced a false positive, §2) are precisely what the BCI literature needs more of to calibrate realistic expectations for consumer-hardware, low-trial-count clinical and assistive-technology deployments.

---

*End of report. All figures in this document are extracted and reproduced verbatim or near-verbatim from the printed cell outputs of the seven source notebooks; no numbers have been estimated, interpolated, or fabricated.*
