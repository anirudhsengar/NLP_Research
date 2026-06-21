# Review Notes Before Submission

Generated after the anchor-compatible DAIGT-inclusive `paper-study` run completed on June 21, 2026.

## Items That May Need Human Confirmation

- Author emails were not provided, so the final paper uses names/student IDs and the confirmed department/university/city/country affiliation only.
- No instructor, course, sponsor, or funding acknowledgment was confirmed. The acknowledgment section is omitted rather than filled with unconfirmed text.
- The final reproduction run uses the frozen merged HC3/DAIGT CSV from the public anchor codebase and does not redistribute it. The local expected path is `data/raw/anchor_merged/merged_dataset.csv`.
- TOEFL remains out of scope. ICNALE is used as the learner-English fairness audit.
- ICNALE is human-only in this study, so it supports false-positive auditing but not AI recall for learner-English prompts.
- ICNALE audit rows are never used for training, threshold selection, or model selection.
- Transformer baselines are not part of the final paper-study claim. The final claim is about sparse logistic-regression reproduction plus selective-policy modification.
- Elastic-net logistic regression is implemented as an optional ablation but excluded from the default final study because it is too slow for the large sparse run and does not change the paper's main harm-reduction claim.

## Final Run Notes

- The selected model is `word_char_stats_lr`.
- Final result tables are generated from `reports/results/paper_study_summary.csv`, `reports/results/selective_policy_table.csv`, `reports/results/topic_holdout_results.csv`, and `reports/results/style_invariance_results.csv`.
- Bootstrap confidence intervals use 1,000 resamples. The GPT-wiki-intro bootstrap population is capped at 20,000 rows for CPU feasibility; this cap is recorded in the result CSV.
- The selected model is evaluated on the full 300,000-row GPT-wiki-intro set. Non-selected sparse variants use the configured comparison sample for large out-of-domain comparisons.
- The paper's claimed improvement is harm reduction through high-confidence selective deployment, not state-of-the-art raw detection accuracy.
