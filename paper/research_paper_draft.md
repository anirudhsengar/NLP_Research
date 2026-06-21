# A Fairness-Constrained Extension of HC3/DAIGT TF-IDF Logistic Regression for AI-Generated Text Detection

Authors: Ali Eissa - 23000798; Ammar Okla - 23000240; Anirudh Sengar - 23001546; Mohammed Kohanfekr - 23003266; Suleiman Altaf - 23001666

Affiliation: Department of Computer Science, British University in Dubai, Dubai, United Arab Emirates.

## Abstract

AI-generated text detection is often reported as a binary classification problem, but educational use requires a stricter standard: detectors must avoid falsely accusing human writers, especially learner-English writers. This paper presents a fairness-constrained extension of the HC3 plus DAIGT v2 TF-IDF logistic-regression baseline used by Alikhanov et al. The study reproduces the sparse word TF-IDF baseline with the public codebase's fixed source split, then evaluates interpretable feature-family variants, leave-one-source-out robustness, ICNALE learner-English calibration, and a three-way selective policy. In the final DAIGT-inclusive evaluation, the selected word+character+statistics logistic model satisfies pooled-human and learner-human calibration constraints at 1 percent FPR. On the held-out ICNALE audit, high-confidence AI false positives fall from 8.32 percent at the default threshold to 0.81 percent, with learner-English FPR of 0.85 percent. On the full 300,000-row GPT-wiki-intro set, high-confidence FPR falls from a default-threshold FPR of 35.02 percent to 4.30 percent, with high-confidence AI recall of 29.81 percent. The contribution is therefore not state-of-the-art raw detection accuracy. It is an interpretable, artifact-backed framework for reducing accusation risk while exposing residual source-shift risk.

## Keywords

AI-generated text detection, HC3, DAIGT v2, logistic regression, TF-IDF, selective classification, learner English, fairness, ICNALE

## Introduction

AI writing tools create a difficult academic-integrity problem. A detector may identify some AI-generated text, but a false positive can place a real student under suspicion. This risk is particularly serious for non-native and learner-English writers, whose writing style can differ from the native-speaker or benchmark distributions that many detectors implicitly learn.

The methodological anchor is the recent HC3 and DAIGT v2 AI-text-detection study by Alikhanov et al. [1], which reports a TF-IDF logistic-regression baseline under topic-based splitting. Its public codebase provides a frozen merged HC3/DAIGT CSV and fixed manual source split [10]. This prior work is the closest same-dataset and same-model-family reference point for evaluating a sparse detector. The present study extends that baseline through four research questions:

1. How does an Alikhanov-style word TF-IDF logistic-regression baseline behave under topic/source holdout?
2. Which interpretable feature families improve in-domain accuracy, and which increase false positives under shifted human writing?
3. Can a fairness-constrained selective policy reduce learner-English false positives while preserving useful high-confidence AI recall?
4. How stable are sparse detector scores under source/topic shift and deterministic style-only text variants?

Accordingly, the evaluation emphasizes safety and robustness for sparse detectors rather than transformer leaderboard ranking.

## Related Work

HC3 introduced the Human ChatGPT Comparison Corpus and released detection systems for distinguishing human and ChatGPT answers [2]. Guo et al. also provide the dataset lineage used in the present study. Because HC3 is a question-answering corpus collected early in the ChatGPT period, high HC3 performance alone does not establish reliable educational deployment.

Alikhanov et al. provide the closest reproduction anchor for this study [1]. Their work uses HC3 and DAIGT v2, applies source/topic-based splitting to reduce information leakage, and reports TF-IDF logistic regression as a classical baseline alongside BiLSTM and DistilBERT models. DAIGT v2 is represented through the frozen merged CSV distributed by the public codebase [10], derived from the public DAIGT v2 release [9]. This extension keeps the classical sparse model family but changes the objective from raw binary accuracy to deployment-aware selective classification and fairness auditing.

Fairness concerns are motivated by Liang et al., who found that GPT detectors can misclassify non-native English writing as AI-generated [3]. Jiang et al. report a different result in a large-scale writing-assessment setting [7], which suggests that detector fairness depends strongly on data alignment, population representation, and threshold policy. ICNALE is therefore used here as a learner-English human audit, not as training data [8].

Robustness work further motivates the evaluation design. Ghostbuster shows that feature-based systems can remain competitive when features are chosen carefully [4], while M4 and RAID show that detectors often fail across unseen domains, generators, and attacks [5], [6]. These papers motivate source/topic holdout tests and the decision to report threshold transfer and review-zone behavior instead of only HC3 test accuracy.

## Data

The study uses three evaluation data sources. The reproduction dataset is the frozen HC3/DAIGT merged CSV from the public codebase, containing 124,195 rows: 80,453 human texts and 43,742 AI texts across five HC3 sources and fifteen DAIGT v2 prompts. The fixed source split contains 85,897 training rows, 24,987 validation rows, and 13,311 test rows. GPT-wiki-intro is used as a 300,000-row out-of-domain binary evaluation set. ICNALE contributes 8,140 human essays for learner-English fairness analysis; 5,698 are reserved as the final audit split.

The full reproduction path is intentionally strict. The expected local file is data/raw/anchor_merged/merged_dataset.csv, with required columns text, label, and source. Experiments that exclude the frozen HC3/DAIGT merge are treated as separate preliminary or ablation analyses, not as the full reproduction evaluation. The final reported evaluation uses the frozen merged CSV and its fixed source split.

ICNALE is human-only in this evaluation. It supports false-positive and subgroup analysis but cannot measure AI recall for learner-English prompts. The ICNALE calibration split is used only to choose the selective high-confidence threshold. The held-out ICNALE audit split is never used for training or model selection.

## Methodology

### Reproduction Baseline

The reproduction track trains an Alikhanov-style word TF-IDF logistic-regression model under the public codebase's final manual source split. For HC3, split groups are the source domains: Reddit ELI5, finance, open QA, medicine, and wiki CSAI. For DAIGT v2, split groups are prompt names such as Distance Learning, Car-free Cities, and Cell Phones at School. Entire sources/prompts are assigned to train, validation, or test partitions so the test split contains unseen topical groups.

### Model Variants

The comparison includes fixed sparse logistic-regression variants: Alikhanov-style word TF-IDF, word-only TF-IDF, word+character+statistics, no-character word+statistics, statistics-only, and a character-capped full model. These variants are intentionally interpretable and CPU-friendly. Elastic-net regularization is supported as an optional heavier ablation, but it is not required for the default final study.

The final modified model is selected by validation high-confidence AI recall, subject to human false-positive constraints. The constraints are pooled calibration human FPR <= 1 percent and ICNALE learner calibration FPR <= 1 percent.

### Selective Policy

The detector uses a three-way policy rather than a single accusation threshold. Scores below the low threshold are classified as human. Scores between the low and high thresholds enter a manual-review zone. Scores above the high threshold are high-confidence AI. The low threshold is the validation-calibrated detector threshold. The high threshold is calibrated from held-out human writing, including learner-English calibration examples when ICNALE is available.

This policy reports coverage, review-zone rate, selective risk, high-confidence AI recall, high-confidence human FPR, native and learner subgroup FPR, and bootstrap confidence intervals.

The selected model is chosen by validation high-confidence AI recall subject to both calibration constraints. The held-out ICNALE audit rows are never used for training, threshold selection, or model selection. Bootstrap confidence intervals use 1,000 resamples in the final evaluation; for the 300,000-row GPT-wiki-intro set, resampling is capped at 20,000 examples for feasible CPU execution.

### Style-Invariance Audit

The style audit uses deterministic transformations only: whitespace normalization, quote normalization, punctuation normalization, and a combined style-normalized variant. It also compares character-heavy and character-reduced model variants. The goal is to measure score sensitivity to surface style changes without introducing external LLM-generated counterfactuals.

## Results

TABLE I summarizes the selected model's behavior in the final experiment.

{{paper_study_summary_table}}

The selected model is the word+character+statistics logistic-regression variant. On the held-out source test set, the default-threshold FPR is 2.74 percent and the high-confidence FPR is 0.12 percent, while high-confidence AI recall is 36.14 percent. On the full GPT-wiki-intro out-of-domain set, the default-threshold FPR is 35.02 percent and high-confidence FPR is 4.30 percent, with bootstrap 95 percent CI from 3.84 to 4.64 percent. The high-confidence AI recall on GPT-wiki-intro is 29.81 percent, with bootstrap 95 percent CI from 28.67 to 30.51 percent. On the ICNALE fairness audit, high-confidence FPR is 0.81 percent, with bootstrap 95 percent CI from 0.60 to 1.04 percent.

Figure 1 shows the model-family trade-off between high-confidence AI recall and high-confidence human false positives.

![FIGURE 1. Fairness-constrained sparse model trade-off under the final evaluation protocol.](figures/paper_study_model_tradeoff.png)

TABLE II reports the learned selective thresholds and calibration outcomes for each sparse model family.

{{selective_policy_table}}

All reported sparse variants satisfy the pooled-human and learner-human calibration constraints. The selected word+character+statistics model has the highest validation high-confidence AI recall among constrained variants, 47.16 percent, with pooled calibration FPR of 0.19 percent and learner calibration FPR of 0.99 percent. This selection rule favors safer high-confidence decisions over raw recall.

TABLE III reports leave-one-HC3-source-out robustness for the word+character+statistics model. These results test whether the detector has learned general authorship signals or source-specific artifacts.

{{topic_holdout_table}}

The source-holdout results show that topic transfer remains the hardest failure mode. Medicine and Reddit ELI5 transfer reasonably well, but finance produces a 17.66 percent human FPR and wiki CSAI produces a 67.70 percent human FPR. This is why the paper treats source-shift robustness as an open deployment risk rather than as solved by the selective policy.

![FIGURE 2. Leave-one-HC3-source-out robustness across HC3 source domains.](figures/topic_holdout_results.png)

Figure 3 compares default-threshold false positives with high-confidence false positives for the selected model. The important change is conceptual: mid-range cases are no longer treated as automatic accusations.

![FIGURE 3. Three-way selective policy trade-off for the selected sparse logistic model.](figures/selective_policy_tradeoff.png)

TABLE IV reports deterministic style-invariance sensitivity. Lower score shifts and lower decision-flip rates are preferable, especially for human text.

{{style_invariance_table}}

For the selected model, deterministic style normalization produces small average score shifts in the 1,000-row style audit samples: 0.0050 on held-out source test, 0.0021 on GPT-wiki-intro, and 0.0056 on ICNALE. Decision flips remain at or below 0.3 percent in these samples. Character features still increase sensitivity to punctuation changes compared with no-character variants, so character evidence should be interpreted as useful but style-sensitive.

![FIGURE 4. Score sensitivity under deterministic style-only text variants.](figures/style_invariance_shift.png)

## Discussion

The results show that sparse detector evaluation changes substantially when the objective shifts from binary classification accuracy to calibrated deployment behavior. The Alikhanov-style baseline supplies the dataset family and fixed source-split protocol, while the added modification is fairness-constrained selective classification for sparse logistic detectors.

The key deployment insight is that a detector can have strong benchmark metrics while still being unsafe as an accusation tool. A selective policy changes the output semantics. It does not claim that every uncertain text is human or AI. Instead, it exposes uncertainty as a manual-review zone and reserves the AI label for high-confidence cases calibrated against human writing from the target population.

Feature-family comparisons are also important. Character n-grams improve validation high-confidence AI recall in the selected model, but they may encode style artifacts, punctuation habits, or corpus-specific formatting. The no-character and character-capped variants make this trade-off visible. The deterministic style audit gives a reproducible first test of whether surface edits move model scores.

The most important limitation is distribution shift. On GPT-wiki-intro, the selective policy substantially reduces false positives relative to the default threshold, but high-confidence FPR remains too high for automatic accusation. This means the proposed modification is not a complete detector. It is a safer decision policy for cases where the model has unusually strong evidence and the target population is represented in calibration. The model is useful only as decision support and should not be used as automatic proof of misconduct.

## Limitations

The frozen HC3/DAIGT merged CSV is a local external dataset and is not redistributed here. The full reproduction evaluation requires the configured CSV file to be available at the expected path.

ICNALE is human-only here, so it cannot measure recall on learner-English AI generations. It only audits false positives for human writing.

The transformer baselines are optional and may be sampled for feasibility. Any sampled baseline table must be labeled as sampled and should not be compared as if it were full-dataset inference.

The style-invariance audit is deterministic and lightweight. It does not replace a full counterfactual rewriting study or adversarial paraphrase benchmark.

Elastic-net logistic regression is implemented as an optional ablation but is excluded from the default final evaluation because it is slow on large sparse matrices and does not change the main paper claim.

Leave-one-source-out results show that some HC3 sources do not transfer cleanly. Future work should expand calibration data by assignment type and writing population, evaluate paraphrase and adversarial robustness, and test whether selective policies remain stable for newer generator families.

## Conclusion

Using the frozen HC3/DAIGT merged dataset, this study evaluates sparse TF-IDF logistic regression under fixed source splitting with feature-family, source-holdout, learner-English, and selective-policy analyses. The final evaluation shows that calibrated high-confidence thresholds can reduce false accusations for target-population human writing. It also shows the remaining cost: source shift can still produce unacceptable false-positive rates. The appropriate claim is harm reduction and careful deployment, not automatic authorship judgment.

## References

[1] A. Alikhanov, A. Amangeldi, D. Demeubay, D. Akhmetzhan, N. Moldakhmetov, O. Polat, and G. Zharas, "AI Generated Text Detection," arXiv:2601.03812, 2026.

[2] B. Guo, X. Zhang, Z. Wang, M. Jiang, J. Nie, Y. Ding, J. Yue, and Y. Wu, "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection," arXiv:2301.07597, 2023.

[3] W. Liang, M. Yuksekgonul, Y. Mao, E. Wu, and J. Zou, "GPT detectors are biased against non-native English writers," Patterns, vol. 4, no. 7, art. 100779, 2023, doi: 10.1016/j.patter.2023.100779.

[4] V. Verma, E. Fleisig, N. Tomlin, and D. Klein, "Ghostbuster: Detecting Text Ghostwritten by Large Language Models," in Proc. 2024 Conf. North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2024, pp. 1702-1717, doi: 10.18653/v1/2024.naacl-long.95.

[5] Y. Wang et al., "M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection," in Proc. 18th Conf. European Chapter of the Association for Computational Linguistics, 2024, pp. 1369-1407, doi: 10.18653/v1/2024.eacl-long.83.

[6] L. Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors," in Proc. 62nd Annual Meeting of the Association for Computational Linguistics, 2024, pp. 12463-12492, doi: 10.18653/v1/2024.acl-long.674.

[7] Y. Jiang, J. Hao, M. Fauss, and C. Li, "Detecting ChatGPT-generated essays in a large-scale writing assessment: Is there a bias against non-native English speakers?" Computers & Education, vol. 217, art. no. 105070, 2024, doi: 10.1016/j.compedu.2024.105070.

[8] S. Ishikawa, The ICNALE Guide: An Introduction to a Learner Corpus Study on Asian Learners' L2 English. London, U.K.: Routledge, 2023.

[9] D. Crofter, "DAIGT V2 Train Dataset," Kaggle dataset release, 2023.

[10] crusnix, "ai_text_detector_final," GitHub repository, 2026. Available: https://github.com/crusnix/ai_text_detector_final.
