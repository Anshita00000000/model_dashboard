# CLAUDE.md — Project Context

Save this at the repo root. Claude Code reads it automatically at the start of every
session, so the task prompts don't need to repeat any of it.

---

## What this system is

An internal ML platform for **CarePay**, a healthcare financing company in India.
CarePay partners with clinic chains ("merchants") and finances elective medical
treatments for their patients.

Clinics generate leads through paid ads, organic search, and phone enquiries. Most
leads never convert. The clinic's sales team has limited calling capacity, so the
question this platform answers is:

> **Given a fresh lead, how likely are they to actually book, attend, and pay for
> treatment — so the team knows who to call first?**

The platform automates the full path from raw clinic lead file to a trained,
versioned scoring model and batch predictions.

## Merchants currently in scope

| Merchant | Vertical | Typical volume | Conversion rate |
|---|---|---|---|
| Evoke | Hair treatment | ~24k leads/quarter | ~2.5% |
| Misya | Aesthetics / wellness | ~39k leads/quarter | ~3.2% |
| Nivaan | Pain management | ~21k leads/quarter | ~1.5% |

Each merchant has different fields, different funnel stage names, and different
data quality. **The platform must be merchant-agnostic** — nothing hardcoded to one
clinic's schema.

## The funnel

Merchants use different vocabulary for the same underlying progression:

```
Lead created → Appointment booked → Consultation attended → Converted (paid)
```

- Evoke:  No Appointment → Appointment Booked → Consulted → Converted
- Nivaan: (lead) → OPD Booked → OPD Done → Procedure Done
- Misya:  Not Converted → Consulted → Converted   (3 levels, no separate booking stage)

Two consequences the platform must handle:

1. **The number of funnel stages varies by merchant.** Never assume 4 levels.
2. **Stages can be degenerate.** Nivaan's "OPD Booked → OPD Done" had a 99.8%
   show-up rate — only 8 leads booked without attending. A stage with 8 examples
   cannot support its own model and must be detected and merged, not modelled.

## Why enrichment matters

Elective treatment is expensive and usually financed. A lead's **ability to pay** is
therefore a genuine predictor, not just their interest level. So lead data is
enriched from external sources:

| Source | What it provides |
|---|---|
| CRIF | Credit bureau — accounts, balances, credit score, enquiry history |
| Equifax | Credit bureau — alternative provider, similar fields |
| EPFO | Employment / provident fund status |
| Salary estimator | Modelled income |
| PayU | Payment behaviour |
| LeadCreditEngineOutput | Internal credit decisioning output |

**Critical operational reality: enrichment coverage is only 35–43%.** Most leads
have no bureau match at all — either they're genuinely credit-invisible ("no-hit")
or the pull failed. The platform must treat "no bureau data" as a legitimate,
informative state, never as zeros and never imputed away.

## Vocabulary

- **merchant** — a clinic chain (Evoke, Misya, Nivaan)
- **lead** — one prospective patient enquiry
- **disposition** — the outcome/funnel stage a lead reached
- **enriched** — this lead has real bureau data (has accounts on file)
- **no-hit** — bureau was queried successfully but found no credit history
- **chained scoring** — modelling each funnel stage conditionally on the previous
- **bundle** — an immutable trained-model artifact including fitted preprocessing
- **lift** — how much better the top-scored group converts vs a baseline

## Statistical character of the problem

- **Rare events.** Conversion is 1.5–3.2%. Accuracy is meaningless; use ROC-AUC,
  PR-AUC, and lift.
- **Realistic performance is ROC-AUC 0.65–0.80.** Anything above 0.90 on a rare
  event is almost certainly leakage, not skill. This has been verified repeatedly.
- **Enrichment adds roughly +0.06 to +0.10 AUC** when it's clean.
- **Small positive counts.** A test set may contain only ~250 converters, so
  ratio-based metrics computed on a thin denominator are volatile and need
  confidence intervals.

## Hard-won rules — every one of these came from a real bug

### Data quality
- **Sentinel values masquerade as numbers.** Equifax uses `-1` for "no score". CRIF
  uses `0` and `10`–`18` as "insufficient data" reason codes; real scores start at
  ~300. Left unmapped, a model reads a reason code as a credit score.
- **Bureau no-hits return zeros, not nulls,** in behavioural fields. "0 accounts"
  then looks like an observation rather than an absence. Blank these explicitly.
- **Phone numbers stored as floats** (`7889358744.0`). Strip the `.0` *before*
  extracting digits, or you slice off the leading digit. This once produced 1 match
  instead of 16,073.
- **Date formats are ambiguous.** Verify DD/MM vs MM/DD empirically by checking
  whether any component exceeds 12. Never assume.
- **Near-duplicate categories** — `Gurugram`/`Gurgaon`, `FB Ads`/`Fb Ads`, trailing
  whitespace. Consolidate before training or you fragment small categories further.
- **Text-encoded durations** — `"6yrs 0mon"` needs parsing to numeric months.

### Modelling
- **Chained conditional stages** beat predicting the final outcome directly, and
  guarantee ordinal consistency by construction.
- **K outcome levels require K−1 conditional stages**, not K.
- **`cat_smooth` from the start.** A clinic with 115 training rows once learned a
  ~2x-inflated conversion rate, then grew 9x in volume the following month and
  dragged predictions with it.
- **Never impute missing enrichment.** LightGBM's native NaN handling is
  informative — "no bureau data" is real signal.
- **Stratify splits jointly** on outcome × source × enrichment, not one at a time.

### Leakage — the dominant risk in this domain
- **Post-event fields leak.** Turnaround time, procedure performed, status
  restatements are only knowable after the outcome.
- **Targets accidentally left in features** produce a perfect 1.0000 AUC.
- **Data provenance leaks.** One merchant's converters had bureau data pulled at
  loan origination through a different process than non-converters. The model
  learned to identify *which file a row came from* — provenance detection scored
  AUC 0.99 and made P(converted) look like 0.97. No feature-dropping fixed it,
  because the difference was distributional, not confined to columns.
- **Inconsistent formulas across sources leak.** The same field computed two
  different ways in two batches becomes a batch-detector. Check arithmetic
  invariants per source.
- **Outcome immaturity.** Recently-created leads haven't had time to convert, so
  they look like negatives when they're really unknown. Check outcome rate against
  time-since-creation before trusting a temporal split.

## Design principles for this codebase

1. **All logic in `app/core/` — importable, testable, no Streamlit import.**
   Streamlit is a UI layer, not an application layer. A FastAPI service must be
   addable later without rewriting logic.
2. **Everything is user-configurable.** Split strategy, stratification columns,
   funnel stages, feature selection, architecture, hyperparameters. Sensible
   defaults pre-filled, nothing hardcoded.
3. **Append-only storage.** Every dataset, split, model bundle and prediction run is
   a new timestamped artifact. Nothing is updated or deleted — this is the audit
   trail, and it must answer "which model, trained on which data, produced this
   score, and when".
4. **Model bundles are immutable and self-contained.** A bundle pins fitted
   preprocessing state, not just the model object. `predict()` never refits.
5. **Leakage gates run automatically and block on failure.** They are not optional
   checks the user might remember to run.
6. **Fail loudly.** Silent coercion, silent zero-filling, and silently dropping rows
   have all caused real errors here. Raise, or surface a visible warning.

## Build phases

- **Phase 1 (current):** Tabs 5, 6, 7 — data prep, training/testing, prediction.
  Driven by an already-enriched CSV so the modelling layer can be built and
  validated before the ingestion layer exists.
- **Phase 2:** Tabs 1–4 — clinic file upload, EDA, field mapping to canonical CRM
  schema, enrichment orchestration across the six sources, feature extraction.

The Phase 1 modelling layer consumes the output contract of Phase 2. That contract
is defined by the enriched CSVs already produced for Evoke, Misya and Nivaan.
