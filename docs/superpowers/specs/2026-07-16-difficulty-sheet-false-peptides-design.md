# Design: Add FALSE-synthesis peptides to Prediction.xlsm "Difficulty (Sheet)"

**Date:** 2026-07-16
**Status:** Approved design — pending spec review

## Goal

Extract every peptide sequence whose `synthesis_flag` is `FALSE` from
`data/raw/peptide_baza.csv` and place those sequences into the
**"Difficulty (Sheet)"** tab of `experiments/predictors/Prediction.xlsm`,
so the user can run the workbook's existing "Difficult couplings predictor"
macro on them.

Only the sequence text is written. Per-residue difficulty scores are **not**
computed here — the workbook's VBA macro produces them when the user clicks the
in-sheet button (its embedded help says: *"Just paste your list in the sheet …
and click the button in the sheet. List of difficulty coefficients will be
created …"*).

## Source data

- File: `data/raw/peptide_baza.csv`, **semicolon-delimited**, UTF-8.
- Relevant columns: `peptide_seq`, `synthesis_flag`.
- Filter: `synthesis_flag == "FALSE"` (case-insensitive, trimmed).
- Result: **232 sequences**, no duplicates, kept in original CSV row order.
- Validation (already confirmed): all 232 use only the standard 20 one-letter
  amino-acid codes (`ACDEFGHIKLMNPQRSTVWY`), uppercase; lengths 9–22 (202 are
  length 15). This matters because the macro substitutes a neutral 0.9 for any
  non-natural residue — none of ours trigger that.

## Target

- File: `experiments/predictors/Prediction.xlsm` (macro-enabled, contains VBA +
  13 ActiveX controls + charts + drawings).
- Sheet: **"Difficulty (Sheet)"**, which maps via `workbook.xml` rId4 to
  `xl/worksheets/sheet4.xml`.
- Layout of that sheet: `A1` = "Sequence" header; rows 2–5 = four demo peptides
  with pre-computed scores across columns B–O; an ActiveX button
  `cmdDifCoupSheet` anchored top-left (backed by `drawing4.xml` +
  `vmlDrawing4.vml` + `activeX12.xml`).

## Placement decision

**Append below the demos.** Keep existing rows 1–5 untouched. Write the 232
sequences into **column A, rows 6–237**, one sequence per row. Column A stays
contiguous from row 2, so the macro reads the demos and the new list together.
"Difficulty (Detail)" is **not** modified.

## Write method: surgical worksheet-XML edit

openpyxl is rejected: a load/save round-trip of this workbook drops
`drawing4.xml` (which anchors the Predict button), chart styling, and several
sheet-rels — visually breaking the very button the user needs. (VBA and the
ActiveX `.bin/.xml` parts do survive, but the drawing that places the button
does not.)

Instead, patch the zip in place:

1. **Back up** `Prediction.xlsm` → `Prediction.backup-YYYYMMDD-HHMMSS.xlsm`
   alongside it before any write.
2. Read the `.xlsm` as a zip. Copy **every** entry verbatim (same compression)
   **except** `xl/worksheets/sheet4.xml`.
3. For `sheet4.xml`, modify only two things:
   - Append 232 `<row>` elements (r="6"…r="237"), each containing a single
     inline-string cell in column A:
     `<c r="A6" s="2" t="inlineStr"><is><t>SEQ</t></is></c>`.
     Inline strings avoid touching `sharedStrings.xml` (and its count /
     `calcChain.xml`). Style `s="2"` matches column A's existing style.
   - Update `<dimension ref="A1:O5"/>` → `<dimension ref="A1:O237"/>` (columns
     B–O remain populated only in demo rows 2–5; column A now extends to 237).
   - Leave the root element, namespaces, `<cols>`, `<drawing>`,
     `<legacyDrawing>`, and `<controls>` blocks byte-identical.
4. XML-escape sequence text (defensive; our data is plain A–Z so no escaping is
   actually triggered).
5. Write the new zip to a temp file, then replace the original.

No new Python dependencies (uses stdlib `zipfile`, `csv`, `shutil`). No Excel or
COM required.

## Preconditions & handoff

- **Excel must be closed** on this workbook before the edit (a
  `~$Prediction.xlsm` owner-lock file indicates it is currently open). If it is
  open and the user later saves from that session, it will overwrite the edit.
- After the edit: user opens `Prediction.xlsm`, goes to "Difficulty (Sheet)",
  clicks the "Difficult couplings predictor" button. The macro fills the score
  columns and colors each sequence (red = difficult, green = easy).

## Verification

After writing, re-open the produced `.xlsm` as a zip and assert:

1. The set of zip entries is **identical** to the backup's set (no parts added
   or dropped) — confirms drawings/charts/VBA/ActiveX preserved.
2. `sheet4.xml` parses as XML and contains exactly `4 (demo) + 232 = 236`
   populated column-A cells below the header, with rows 6–237 present.
3. The 232 inline-string values equal the filtered FALSE sequences in order.
4. Cross-check the count: independently re-filter the CSV and confirm 232.

Optionally open in Excel (manual) to confirm the button still renders and the
macro runs — outside the scope of the automated script.

## Out of scope

- Computing difficulty coefficients in Python (left to the macro).
- Any change to "Difficulty (Detail)" or other sheets.
- Rebuilding the workbook via openpyxl or COM.

## Risks

- **Hand-written XML must be valid** or Excel shows a repair prompt. Mitigated
  by the verification step (parse + entry-set diff) and the backup.
- **Stale lock file**: if `~$Prediction.xlsm` is stale (Excel already closed),
  the edit is safe; the precondition is a no-op.
