# Q4 Manual Trace — Claude Code as Agent

**Question:** What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.

**Reference:** Operating Margin for 3M in FY2022 has decreased by 1.7% primarily due to:
- Decrease in gross Margin
- mostly one-off charges including Combat Arms Earplugs litigation, impairment related to exiting PFAS manufacturing, costs related to exiting Russia and divestiture-related restructuring charges

---

## Search 1: "3M 2022 operating margin change drivers"

Top 20 results — **no 3M_2022 pages in top 20:**

| Rank | Score | Doc ID | Notes |
|------|-------|--------|-------|
| 1 | 0.794 | AES_2022_10K_p100 | Wrong company |
| 2 | 0.793 | AES_2022_10K_p102 | Wrong company |
| 3 | 0.791 | AES_2022_10K_p101 | Wrong company |
| 4 | 0.790 | INTEL_2022Q4_EARNINGS_p1 | Wrong company |
| 5 | 0.786 | AES_2021_10K_p99 | Wrong company |
| 6 | 0.784 | 3M_2021_10K_p17 | Wrong year |
| 7 | 0.780 | INTEL_2022_10K_p24 | Wrong company |
| 8 | 0.780 | NIKE_2022_10K_p42 | Wrong company |
| 9 | 0.776 | 3M_2016_10K_p27 | Wrong year |
| 10-20 | ... | Various other companies | No 3M_2022 |

**Problem:** Generic "operating margin change" query returns other companies' margin discussions. AES dominates because they have detailed per-SBU margin tables.

## Search 2: "3M 2022 Combat Arms Earplugs litigation PFAS operating income impact"

| Rank | Score | Doc ID | Notes |
|------|-------|--------|-------|
| 1 | 0.762 | 3M_2022_10K_p111 | Litigation detail page |
| 2 | 0.754 | 3M_2022_10K_p121 | Litigation detail page |
| 3 | 0.742 | PFIZER_2022_10K_p4 | Wrong company |
| 7 | 0.737 | 3M_2022_10K_p29 | Segment performance |

**Better** — litigation-specific terms pull 3M 2022 pages. But these are deep litigation notes (p111, p121), not the operating margin summary.

## Search 3: "3M 2022 10K operating margin decrease gross margin one-off charges restructuring"

Top 10 — **still no 3M_2022 pages.** Intel and AES dominate.

## Key Finding: Where is the answer page?

The answer is on **3M_2022_10K_p19** — but this page never appeared in any top-20 search result.

**Why?** Page 19 starts with text about Belgium PFAS operations and Russia conflict (not "operating margin"), then has a financial table showing the margin bridge. The embedding model sees the Belgium/Russia text as the dominant semantic content and doesn't rank it for "operating margin" queries.

## Reading the Answer Page: 3M_2022_10K_p19

The page contains a table: "Operating income margin and earnings per share attributable to 3M common shareholders – diluted"

Key data:
- **2021 operating margin: 20.8%**
- **2022 operating margin: 19.1%**
- **Total decrease: 1.7 percentage points**

Breakdown of the -1.7% from special items:
| Item | Impact on Operating Margin |
|------|---------------------------|
| Net costs for significant litigation | -6.7% |
| Gain on business divestitures | +8.0% |
| PFAS manufacturing exit costs | -2.4% |
| Russia exit charges | -0.3% |
| Divestiture-related restructuring | -0.1% |
| Divestiture costs | -0.2% |
| **Total special items** | **-1.7%** |

Supporting detail from **3M_2022_10K_p18**: The litigation charges were "pre-tax charges associated with steps toward resolving Combat Arms Earplugs litigation and associated with additional commitments to address PFAS-related matters at its Zwijndrecht, Belgium site (approximately $1.3 billion and $355 million, respectively, in 2022)."

## Correct Answer

Operating Margin for 3M decreased by 1.7% in FY2022 (from 20.8% to 19.1%), driven primarily by one-off special items:
- Combat Arms Earplugs litigation costs (~$1.3B)
- PFAS manufacturing exit costs (-2.4% margin impact)
- Russia exit charges (-0.3% margin impact)
- Partially offset by gain on Food Safety business divestiture (+8.0%)

**Exact Answer:** Operating margin decreased by 1.7 percentage points, from 20.8% to 19.1%, primarily due to one-off litigation and restructuring charges.

---

## Why the Model Failed

1. **Retrieval failure:** The answer page (3M_2022_10K_p19) never appeared in top-20 results for any query. The page's text is mostly about Belgium operations and Russia conflict, with the financial table buried in the middle. The embedding model ranks it low for "operating margin" queries.

2. **The model found 3M_2022_10K_p20 (Step 0, result #4)** which discusses the organic growth/productivity side (+1.0% benefit), but this only tells half the story. The special items (-1.7%) are on p19.

3. **The model got stuck in a loop** — 51 searches repeating slight variations of the same query, never finding p19. This is the "Exhaustive Search, No Convergence" behavior described in the KARL paper (Section 8.2.3).

4. **Based on incomplete data, the model concluded margin *increased*** — it only saw the +1.0% organic benefit from p20, missed the -1.7% special items from p19, and gave the wrong direction.

## Implications for the Eval

This question would likely be hard for any vector-search-only system because the answer page doesn't have strong semantic overlap with "operating margin change" queries. The KARL paper's RL-trained model might handle this better through learned search strategies (e.g., searching for specific filing page numbers, or searching for litigation charges and then connecting them to margin impact).
