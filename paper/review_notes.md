# Review Notes Before Submission

Generated on June 10, 2026 after the final experiment and paper build pass.

## Items That May Need Human Confirmation

- Author emails were not provided, so the final paper uses names/student IDs and the confirmed department/university/city/country affiliation only.
- No instructor, course, sponsor, or funding acknowledgment was confirmed. The acknowledgment section was omitted instead of leaving unconfirmed draft text.
- TOEFL data were not used because access was paid/restricted. The paper frames ICNALE as a fallback learner-English fairness audit, not as a TOEFL reproduction.
- The reported transformer detector baselines are fixed-sample comparisons with 500 rows per evaluation set. Full 300,000-row transformer inference over GPT-wiki-intro was not run.
- The reported final classifier does not use optional spaCy POS/NER/dependency features or GPT-2 language-model statistics. Those features exist in code but are not part of the final reported model.
- Bootstrap confidence intervals are supported by the implementation but were not enabled in the saved final run, so the paper reports point estimates.
- ICNALE WEP has a corpus caveat: participants declared no online writing-support use, but the ICNALE documentation notes that influence from such tools cannot be fully guaranteed for at-home writing.

## Final Run Notes

- Full-data classical evaluation, calibration, fixed-sample baseline comparison, and full-data ablations were regenerated on June 10, 2026.
- `uv run pytest` passed earlier in the final run with 19 tests.
- `uv run ruff check` passed after the regenerated experiments.
- During baseline inference, Hugging Face emitted unauthenticated download warnings, but the configured models loaded and comparison outputs were written.
- The OpenAI RoBERTa detector load reported unexpected pooler weights, which is a standard Transformers warning for this checkpoint family and did not block inference.
