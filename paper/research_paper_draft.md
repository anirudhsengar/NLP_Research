# Interpretable and Bias-Aware Detection of AI-Generated Text

Authors: Ali Eissa - 23000798; Ammar Okla - 23000240; Anirudh Sengar - 23001546; Mohammed Kohanfekr - 23003266; Suleiman Altaf - 23001666

Affiliation: Department of Computer Science, British University in Dubai, Dubai, United Arab Emirates.

## Abstract

AI-generated text detectors are increasingly discussed in educational settings, but detectors can produce harmful false positives when used for academic-integrity decisions. This project implements and evaluates an interpretable classical detector for AI-generated text, with special attention to false positives on learner-English writing. The detector is trained on HC3 using word TF-IDF, character TF-IDF, and lexical/readability statistics with logistic regression. It is evaluated on an HC3 held-out test split, GPT-wiki-intro as an out-of-domain benchmark, and ICNALE essays as a human-only fairness audit after the paid/restricted TOEFL data could not be accessed. On HC3 test data, the model obtains strong in-domain performance: F1-macro 0.9874, AUROC 0.9992, AUPR 0.9986, false-positive rate 0.0128, and true-positive rate 0.9928. However, the same threshold fails badly under domain and population shift, with GPT-wiki-intro human false-positive rate 0.7999 and ICNALE human false-positive rate 0.7773. A conservative education policy calibrated with held-out human writing reduces ICNALE audit false positives to 0.0086, but lowers HC3 AI recall to 0.4748 and GPT-wiki-intro AI recall to 0.1644. These results show that interpretable classical features can fit in-domain AI-text detection well, but that high in-domain accuracy is not enough for safe educational deployment. The recommended use is cautious manual review, not automated accusation.

## Keywords

AI-generated text detection, learner English, fairness, false positives, logistic regression, TF-IDF, HC3, ICNALE, GPT-wiki-intro

## Introduction

Universities face a practical problem: AI writing systems can generate fluent essays, answers, and summaries, but many available detectors are opaque and unreliable outside the data distribution on which they were tuned. The central risk in an academic-integrity context is not only missed AI text, but also the false accusation of human-written work. This risk is especially serious for non-native or learner-English writers. Liang et al. reported that several GPT detectors falsely classified non-native English essays as AI-generated at high rates, while native-speaker writing was much less affected [1].

This project studies whether an auditable, classical natural language processing (NLP) model can classify AI-generated text while measuring the false-positive risk for learner-English essays. The work follows three design principles. First, the model should be interpretable enough to inspect feature contributions rather than returning only a black-box score. Second, evaluation should include both in-domain and out-of-domain data, because detectors that work on one benchmark may fail on another. Third, fairness analysis should focus on false positives for human text, because in education a false positive can create a disciplinary burden for a student.

The original project proposal planned to reproduce the TOEFL-based fairness setting when possible. Direct TOEFL access was not available to the team because the relevant data are paid or restricted. Therefore, the project uses ICNALE as the learner-English fairness corpus, which was an allowed fallback in the proposal. This choice changes the scope: the study does not claim to reproduce the TOEFL experiment directly. Instead, it audits false positives on ICNALE learner and native-speaker human essays.

The main contribution is a complete, reproducible pipeline that prepares data, trains a logistic-regression AI-text detector, evaluates in-domain and out-of-domain performance, audits subgroup false-positive rates, calibrates a conservative education-use threshold, and provides a Gradio demo with feature explanations. The results are mixed but important: the model performs very well on HC3 test data, yet produces severe false positives on GPT-wiki-intro and ICNALE under the default threshold. This finding supports a cautious conclusion: detector performance should be reported with domain-shift and fairness audits before any educational use.

## Related Work

Liang et al. studied bias in GPT detectors against non-native English writing and showed that detectors could misclassify human learner writing as AI-generated [1]. Their result motivates the false-positive focus of this project. In academic settings, the harm of false positives is asymmetric: a missed AI-generated sample may reduce detector usefulness, but a false positive can wrongly implicate a student.

HC3, introduced by Guo et al., provides paired human and ChatGPT answers across domains and has become a useful benchmark for AI-text detection [2]. This project uses HC3 as the primary training and in-domain evaluation source. However, because HC3 is itself a benchmark distribution, good HC3 performance alone cannot establish safety on student writing or Wikipedia-style text.

GLTR is a statistical detection and visualization approach based on language-model token ranks [3]. The project includes a GLTR-style heuristic baseline using GPT-2 rank statistics. It also compares against two transformer detector baselines on a fixed sample: an OpenAI-community RoBERTa GPT-2 detector and the Hello-SimpleAI HC3 RoBERTa detector [6], [7]. These transformer baselines are used for comparison only; their full 300,000-row inference over GPT-wiki-intro was not run because of computational cost.

GPT-wiki-intro is used as an out-of-domain dataset. Its Hugging Face dataset card describes 150,000 rows containing Wikipedia introductions and GPT-generated introductions, including the fields `wiki_intro` and `generated_intro` [4]. In this project, each row is expanded into one human example and one generated example, resulting in 300,000 evaluation examples.

ICNALE is used for fairness auditing. The ICNALE readme describes Written Essays as 200-300 word essays, and the project data include ICNALE WE, WEP, and WEUAE modules. The WEP readme notes that participants declared they did not use online writing support tools, but also states that influence from such tools cannot be fully guaranteed for at-home writing. This limitation is important and is treated as part of the scope rather than hidden.

## Data

TABLE I summarizes the processed datasets used in the current experiments. All counts come from the saved `data_profile.json` artifact.

| Dataset | Role | Rows | Human | AI |
|---|---:|---:|---:|---:|
| HC3 | Train/validation/test | 85,431 | 58,546 | 26,885 |
| GPT-wiki-intro | Out-of-domain test | 300,000 | 150,000 | 150,000 |
| ICNALE | Human-only fairness audit | 8,140 | 8,140 | 0 |

The HC3 data contain five sources in the processed corpus: reddit_eli5, finance, open_qa, medicine, and wiki_csai. The split used in the saved run contains 59,768 training rows, 12,826 validation rows, and 12,837 test rows. The validation split is used to set a default decision threshold targeting approximately 1 percent false-positive rate on HC3 human validation examples.

GPT-wiki-intro is treated as out-of-domain because it differs from HC3 in source, style, and generation setup. The human side consists of Wikipedia introductions, while the generated side consists of model-written introductions generated from prompts. This dataset is not used for training the classical model.

ICNALE is used only as a human-writing audit. The processed ICNALE data contain 7,740 learner essays and 400 native-speaker essays. Sources include 5,600 WE essays, 2,340 WEP essays, and 200 WEUAE essays. Because ICNALE has no AI-generated labels in this project, AUROC, AUPR, F1, and recall are undefined for the ICNALE audit; the relevant metric is human false-positive rate.

## Methodology

### Preprocessing and Splitting

All datasets are converted to a canonical JSONL schema with sample ID, dataset name, split, text, label, source, domain, group ID, native status, country, first language, CEFR level, length bin, and license tag. Text is whitespace-normalized and empty texts are removed. Labels use 0 for human-written text and 1 for AI-generated text.

HC3 is split by group so that paired human and ChatGPT answers from the same source item do not cross train, validation, and test partitions. The split proportions are 70 percent training, 15 percent validation, and 15 percent test, with random seed 42.

### Features

The reported classical model uses three feature families: word TF-IDF features with unigram and bigram ranges; character TF-IDF features with 3-gram to 5-gram ranges; and basic lexical and readability statistics, including word count, sentence count, type-token ratio, hapax ratio, stopword ratio, punctuation and capitalization ratios, sentence-length variation, burstiness, repeated n-gram ratios, entropy, Flesch reading ease, and Flesch-Kincaid grade.

The implementation also contains optional spaCy POS, named-entity, and dependency features, plus GPT-2 language-model statistics such as perplexity and GLTR-style token-rank ratios. These optional feature families are not enabled in the reported final model. Therefore, the results in this paper should not be described as using POS tagging, NER, dependency syntax, or GPT-2 perplexity as classifier features. They are available only as optional future ablations unless rerun and reported separately.

### Model

The primary model is logistic regression with balanced class weights and maximum iteration count 1000. Logistic regression was selected because it provides transparent coefficients and can support feature-level explanations. The saved model uses the default HC3 validation threshold of 0.4563, chosen to target a false-positive rate near 1 percent on HC3 validation human examples.

Feature importance is reported from logistic-regression coefficients. The largest coefficient mass comes from character TF-IDF features, followed by word TF-IDF features and then basic statistics. This does not mean that all high-weight features are semantically meaningful; character n-gram coefficients can capture punctuation, spacing, and style artifacts. The demo therefore reports both raw feature contributions and grouped feature-family contributions.

### Baselines

The comparison includes three detector baselines on a fixed 500-row sample per evaluation set: a GLTR-style heuristic, an OpenAI-community RoBERTa GPT-2 detector, and the Hello-SimpleAI HC3 RoBERTa detector. The GLTR-style baseline uses GPT-2 token-rank statistics to construct a score. The transformer baselines use Hugging Face sequence-classification checkpoints. Baseline thresholds are calibrated on an HC3 validation sample to target the configured false-positive rate.

The baseline comparison is intentionally labeled as fixed-sample comparison. Full transformer inference over all 300,000 GPT-wiki-intro examples was not run, so the baseline results should not be interpreted as full-dataset transformer results.

### Conservative Education Policy

Because the default threshold produces severe false positives under domain shift, the project implements a conservative education policy. ICNALE is split by participant group into calibration and held-out audit partitions. A higher threshold is calibrated from pooled HC3 validation human text and ICNALE calibration human text, with special attention to learner-English human examples. The resulting education threshold is 0.9931.

The education policy should be interpreted as a manual-review policy, not as a high-recall detector. Scores below 0.4563 are treated as human-written by the default threshold. Scores from 0.4563 to 0.9931 fall into a review zone. Scores above 0.9931 are the only cases classified as AI-generated under the conservative policy.

## Evaluation Metrics

For binary datasets, the project reports F1-macro, AUROC, AUPR, confusion matrices, false-positive rate (FPR), true-positive rate (TPR), and TPR at FPR <= 1 percent. For human-only ICNALE fairness data, AUROC, AUPR, F1, and TPR are undefined; the paper reports human FPR, human recall, and subgroup FPR for groups of at least 30.

Bootstrap confidence intervals are supported by the implementation but disabled in the current saved run. Therefore, all reported results are point estimates.

## Results

### Full-Data Classical Results

TABLE II reports the main full-data evaluation for the classical logistic-regression model under the default HC3-calibrated threshold.

| Set | N | F1 | AUROC | FPR | TPR |
|---|---:|---:|---:|---:|---:|
| HC3 test | 12,837 | 0.9874 | 0.9992 | 0.0128 | 0.9928 |
| GPT-wiki OOD | 300,000 | 0.5043 | 0.7902 | 0.7999 | 0.9466 |
| ICNALE | 8,140 | N/A | N/A | 0.7773 | N/A |

The model performs strongly on the in-domain HC3 test split, but the default threshold does not transfer. GPT-wiki-intro human FPR is 0.7999, and ICNALE human-only FPR is 0.7773. These rates show that HC3 validation calibration is unsafe for Wikipedia-style text and learner-English essays.

### Native and Learner-English False Positives

The ICNALE full-data fairness summary shows 254 false positives among 400 native-speaker essays and 6,073 false positives among 7,740 learner essays. The native-speaker FPR is 0.6350, while the learner-English FPR is 0.7846. The learner-minus-native gap is 0.1496.

Because TOEFL data were unavailable, this is not a direct reproduction of Liang et al. It nevertheless supports the same concern: excellent benchmark performance can coexist with high false-positive rates for human learner writing.

### Calibrated Education Policy

TABLE III compares the default threshold with the conservative education threshold.

| Set | Policy | FPR | TPR |
|---|---|---:|---:|
| HC3 test | Default | 0.0128 | 0.9928 |
| HC3 test | Education | 0.0000 | 0.4748 |
| GPT-wiki-intro OOD | Default | 0.7999 | 0.9466 |
| GPT-wiki-intro OOD | Education | 0.0068 | 0.1644 |
| ICNALE audit | Default | 0.7769 | N/A |
| ICNALE audit | Education | 0.0086 | N/A |

The education threshold sharply reduces false positives. On the held-out ICNALE audit split, FPR falls from 0.7769 to 0.0086. Learner FPR falls to 0.0090, and native FPR is 0.0000. However, the cost is severe: HC3 test TPR drops from 0.9928 to 0.4748, and GPT-wiki-intro TPR drops from 0.9466 to 0.1644. This tradeoff means the education policy is safer for avoiding accusations, but weak as a binary detector.

### Fixed-Sample Baseline Comparison

TABLE IV reports the same-row 500-example comparison for the classical detector and baselines. These results are useful for approximate comparison, but they are not full-dataset transformer results.

| Model | Set | FPR | TPR | Learner FPR |
|---|---|---:|---:|---:|
| Classical default | HC3 | 0.000 | 0.996 | N/A |
| Classical default | GPT-wiki | 0.724 | 0.948 | N/A |
| Classical default | ICNALE | 0.738 | N/A | 0.812 |
| Classical education | ICNALE | 0.002 | N/A | 0.004 |
| GLTR heuristic | ICNALE | 0.208 | N/A | 0.096 |
| OpenAI RoBERTa | ICNALE | 0.118 | N/A | 0.052 |
| HC3 RoBERTa | ICNALE | 0.562 | N/A | 0.608 |

On the fixed ICNALE sample, the OpenAI RoBERTa GPT-2 detector has lower false-positive rates than the default classical detector, and the GLTR heuristic also has lower ICNALE FPR than the default classical detector. The HC3 RoBERTa detector remains high on ICNALE. The conservative classical policy has the lowest ICNALE FPR, but this comes from raising the decision threshold and sacrificing recall on binary datasets.

### Feature Ablation Summary

Feature ablations show that strong in-domain accuracy does not align cleanly with fairness safety. Word TF-IDF alone has HC3 test F1-macro 0.9724 and ICNALE FPR 0.1558. Basic lexical statistics alone have lower HC3 test F1-macro 0.7815 but ICNALE FPR 0.1061. The combined word, character, and statistics model has HC3 test F1-macro 0.9874 but ICNALE FPR 0.7773. This indicates that feature combinations that improve benchmark detection can also amplify false positives under distribution shift.

This result is not a reason to prefer a weak model. Instead, it shows why detector development must evaluate both predictive performance and human false-positive risk before recommending any threshold for educational use.

## Discussion

The most important finding is that in-domain success is misleading. On HC3, the classical logistic-regression detector appears excellent. It has high F1, AUROC, AUPR, and recall at a low false-positive target. Yet the same model fails on GPT-wiki-intro and ICNALE under the default threshold. The failure is not small; it is large enough to make the default detector inappropriate for direct academic-integrity decisions.

The likely reason is distribution shift. HC3 answers differ from Wikipedia introductions and from learner essays in length, topic, punctuation, discourse style, and lexical patterns. Logistic regression with high-dimensional n-grams can learn patterns that separate HC3 human and ChatGPT answers, but those patterns do not necessarily represent the general distinction between human and AI writing. The feature importance tables support this caution because character n-grams dominate coefficient mass and can capture dataset-specific style artifacts.

The fairness result is also important. The ICNALE learner false-positive rate is higher than the native false-positive rate under the default threshold. Even though this project did not use TOEFL data, the ICNALE audit supports the broader concern from Liang et al. that AI-text detectors can penalize non-native or learner-English writing [1]. The exact rates should not be compared directly across TOEFL and ICNALE because the corpora differ.

The conservative education policy demonstrates one practical mitigation: calibrate a higher threshold using human writing from the target population and treat mid-range scores as manual review. This reduces ICNALE false positives to near the target rate. However, the price is much lower recall for AI-generated text. Therefore, a conservative threshold can make the tool less harmful, but it cannot make it a definitive detector.

## Demo and Interpretability

The project includes a Gradio demo that accepts text, returns an AI score, shows the default and education thresholds, labels the review zone, and lists feature contributions. Explanations are computed from logistic-regression feature contributions. The demo also warns users when input text is short, because short text is less reliable for detection.

The interpretability layer is useful for transparency, but it has limits. A feature contribution is not a causal explanation of authorship. It shows which learned features pushed the score up or down for the trained model. In a real educational workflow, such explanations should support careful review, not replace human judgment.

## Limitations

The first limitation is data access. The project could not obtain the TOEFL dataset because the relevant access is paid or restricted. ICNALE is a reasonable fallback for learner-English false-positive auditing, but it is not a direct reproduction of the TOEFL experiment.

Second, ICNALE is human-only in this project. It supports false-positive analysis but cannot measure AI recall for learner-English prompts.

Third, transformer baselines were evaluated only on fixed samples for paper comparison. Full transformer inference on all 300,000 GPT-wiki-intro examples was not run.

Fourth, the reported final classical model does not include optional spaCy POS/NER/syntax features or GPT-2 perplexity features. These components exist in the implementation but are not part of the saved final model.

Fifth, the current run reports point estimates without bootstrap confidence intervals. Confidence intervals would strengthen the paper, especially for subgroup comparisons.

Finally, WEP includes a corpus-specific caveat: participants declared that they did not use writing-support tools, but the ICNALE team notes that influence from such tools cannot be fully guaranteed for at-home writing. This does not invalidate the audit, but it should be disclosed.

## Conclusion

This project implemented an interpretable and bias-aware AI-generated text detection pipeline using classical NLP features and logistic regression. The detector performs very well on the HC3 in-domain test split, but the full-data results show severe false-positive failures on GPT-wiki-intro and ICNALE under the default threshold. A conservative education threshold greatly reduces false positives on ICNALE, but also reduces AI recall. The main conclusion is therefore cautious: AI-text detectors should not be used as automatic evidence of misconduct, especially for learner-English writers. If used at all, they should be calibrated on target-domain human writing, framed as decision support, and paired with manual review.

## References

[1] W. Liang, M. Yuksekgonul, Y. Mao, E. Wu, and J. Zou, "GPT detectors are biased against non-native English writers," Patterns, vol. 4, no. 7, art. 100779, 2023, doi: 10.1016/j.patter.2023.100779.

[2] B. Guo, X. Zhang, Z. Wang, M. Jiang, J. Nie, Y. Ding, J. Yue, and Y. Wu, "How close is ChatGPT to human experts? Comparison corpus, evaluation, and detection," arXiv:2301.07597, 2023.

[3] S. Gehrmann, H. Strobelt, and A. M. Rush, "GLTR: Statistical detection and visualization of generated text," in Proc. 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, 2019, pp. 111-116.

[4] A. Bhat, "GPT-wiki-intro," Hugging Face dataset repository, 2023, doi: 10.57967/hf/0326. [Online]. Available: https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro

[5] S. Ishikawa, The ICNALE Guide: An Introduction to a Learner Corpus Study on Asian Learners' L2 English. London, U.K.: Routledge, 2023.

[6] OpenAI Community, "roberta-base-openai-detector," Hugging Face model repository. [Online]. Available: https://huggingface.co/openai-community/roberta-base-openai-detector

[7] Hello-SimpleAI, "chatgpt-detector-roberta," Hugging Face model repository. [Online]. Available: https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta
