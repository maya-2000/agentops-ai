# Project visuals

Every image here shows what exists in release 0.10.0. None is a mock-up.

| File | What it is | Source |
|---|---|---|
| `hero.png` | *Ask* mode: answer, KPI card and labelled findings for *Which region had the largest revenue decline last month?* | Screenshot of the running UI |
| `evidence.png` | The same answer's **Evidence & Provenance** table and **Analysis Trace** | Screenshot of the running UI |
| `investigation.png` | *Investigate* mode for *Why is revenue growth slowing?*: status, the 11-step analysis plan and the executive summary | Screenshot of the running UI |
| `investigation-drivers.png` | The same investigation's drivers and contributing factors (first four) | Screenshot of the running UI |
| `investigation-recommendations.png` | The same investigation's recommendations, with the findings each rests on | Screenshot of the running UI |
| `architecture.png` | System architecture: every box is a package under `app/` | [`source/architecture.html`](source/architecture.html) |
| `trust-model.png` | The four claim labels, their validation rules and real examples | [`source/trust-model.html`](source/trust-model.html) |

## How the screenshots were taken

- **Setup:** `python -m app.api` and `python -m app.ui`, on the default dataset (seed 42, data as of
  2026-08-31) with the default deterministic model.
- **Capture:** headless Chromium at twice the pixel density. The questions were typed into the UI,
  and some expanders were opened.
- **Editing:** crops and a thin border only. Nothing was edited, combined or retouched.
- **Request IDs** differ from run to run. Everything else is reproducible with the same data and model.

## How the diagrams were made

The diagrams are rendered from the HTML files in [`source/`](source/) with headless Chromium, at
twice the pixel density. They use the Source Sans 3 typeface, as the UI does.

Their content matches the code:

- the claim labels and notes come from `app/ui/view_models.py` (`CLAIM_STYLES`);
- the checks come from `app/evidence/validation.py`;
- the executor's steps come from `app/security/`.

## What is deliberately not here

- **A performance chart:** the per-step timings in the screenshots are single local runs, not
  benchmarks.
- **An evaluation chart:** the suites have different sizes and purposes, so they are shown as a table
  in the README and in [`../evaluation.md`](../evaluation.md), not plotted side by side.
- **A demo GIF:** the static screenshots show the same flow more legibly.
