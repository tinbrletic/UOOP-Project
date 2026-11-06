# Operational checklists

Two procedures referenced from SKILL.md: auditing a dataset or model, and reading a scanned
source without misquoting it.

---

## Auditing a dataset, descriptor set or model

Work in this order. Each step can invalidate everything after it, so stopping early is fine;
skipping ahead is not.

**1. Label provenance.** What physical quantity, measured how, thresholded where. If you
cannot establish this, nothing below is interpretable and no number from the dataset is
comparable to a number from any other.

**2. Column integrity.** **Recompute two or three features from the raw sequences and compare
to the stored values.** Do not trust a column because of its name. Mislabelled or offset
descriptor columns are common, produce no error, and silently invalidate every feature
importance ranking downstream. Check specifically for:
- a function that returns more values per group than the caller writes (counts *and*
  fractions, where only the first N are stored)
- off-by-one column alignment after a rename or a reorder
- a column that is constant, or null, where the generating code says it should not be

**3. Reproducibility of the generating code.** Does the committed script reproduce the
committed data? Pin descriptor-library versions — physicochemical scale values differ between
releases.

**4. Redundancy and collinearity.** Counts stored alongside their own fractions, with length
also present, are near-deterministic functions of each other. This destabilises RFE,
SelectFromModel and sequential selection rankings without affecting headline metrics.

**5. Cohort structure.** Failure rate by length, by batch, by source library, and **by
laboratory or operator**. Look for one stratum holding a disproportionate share of both rows
and outcomes. Laboratory is not the same stratum as source library, and the interlaboratory
evidence identifies it as the dominant one — in one round-robin, racemization ran 2–8% across
submissions with all the high values from a single lab. If provenance is not recorded, say so:
an unrecorded dominant stratum cannot be stratified out.

**6. Leakage.** Resampling or feature selection fitted outside the cross-validation split. A
**persisted resampled file on disk is a strong smell** — resampling belongs inside the CV
pipeline so it only ever sees the training fold. Two further checks specific to synthetic
oversampling of peptide data:
- synthetic rows must keep their sequence and their descriptors **mutually consistent**; a
  randomly generated sequence attached to an interpolated feature vector describes two
  different molecules
- interpolating one-hot or count columns produces fractional values ("0.37 of an alanine at
  the N-terminus") that correspond to no peptide

**7. Near-duplicates across splits.** Deduplicate exactly, then cluster by sequence identity
and split at cluster level. Families of related sequences straddling the train/test boundary
inflate scores even under correct stratified CV.

**8. Metrics.** At 10–15% minority a constant classifier can score ~0.87 accuracy and ~0.93
F1 with MCC 0.000. Require **MCC, AUC-ROC and PR-AUC / average precision with the failure
class as the positive class**. The question a chemist is asking — "which of these will give me
trouble?" — is failure-class recall, and that is the number most often missing.

**9. Baseline.** What does the single best feature achieve alone? A model that does not beat a
one- or two-descriptor baseline has not earned its complexity. If the source reports no
baseline, that is itself a finding.

---

## Reading scanned and OCR'd sources

Much of the best material in this field — handbooks, older application notes, reference
volumes — arrives as a scan. OCR of a two-column page **silently reorders lines**, so a
quotation assembled from adjacent fragments can be wrong while reading perfectly fluently.
Table headers are the worst case: they can be emitted out of order, which inverts a table's
meaning with no visible corruption.

**1. Search single distinctive tokens, not phrases.** OCR joins adjacent words
("difficultsequences", "couplingsbetween"), so multi-word patterns silently fail — and a
failed search looks identical to an absent fact.

**2. Index before reading.** Extract to text, flatten to one `volume / page / line` record per
line, and build a heading index from numbered section lines. Then read located sections in
full. Do not read linearly.

**3. Reconstruct, then verify.** Before quoting, confirm the distinctive substring with a
literal fixed-string search (`grep -F`). Reconstruction without verification is how fluent
misquotations happen.

**4. Cross-check every table against an independent statement of the same fact** — the stated
mechanism, or numbers given elsewhere in the chapter. **Never trust a scanned table's
header-to-column mapping on its own.** If the mechanism text and the table disagree about
which column is which, the mechanism text wins.

**5. Paraphrase rather than quote where reconstruction is uncertain**, and mark it as a
paraphrase.

**6. Record the distinction.** Use the finding record's `reconstruction` field: `verbatim`, or
`reassembled from OCR fragments — verified by <method>`.

**7. Page numbers.** Say whether a citation is the PDF page or the printed page; in scanned
volumes they differ, and a reader cannot check your citation without knowing which.
