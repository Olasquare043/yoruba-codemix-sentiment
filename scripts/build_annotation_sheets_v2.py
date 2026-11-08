# Record of how the second-round annotation sheets were built (9 Oct 2026).
# It reads cells from notebooks/01 and a private working file (annotation_master.csv) that is not released;
# everything needed to reproduce the paper is in data/ and results/. Kept for transparency only.
"""Build v2 annotation sheets: full coverage of the AfriSenti Yoruba TEST set candidates.

Design
- Stratum S1: every test tweet with heuristic CMI > 0 (candidate mixed). Annotated in full.
- Stratum S0: test tweets with CMI == 0. A uniform random audit sample of 400 is annotated,
  so we can estimate how many mixed tweets the heuristic misses.
- Test rows already in the v1 sheets are excluded here (annotators finish them in v1).
- 250 rows are shared by both annotators (agreement), the rest are split evenly.
"""
import sys, hashlib, nbformat
import pandas as pd, numpy as np
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Alignment, Font, PatternFill

NB, V1_MASTER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEED_V2 = 2026

nb = nbformat.read(NB, 4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
g = {}
exec(compile("\n\n".join([cells[1], cells[2], cells[4]]), "nb", "exec"), g)
df = g["df"]

df["text_md5"] = df.tweet.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
test = df[df.split == "test"].copy()
test["stratum"] = np.where(test.cmi > 0, "S1_candidate", "S0_audit")

v1 = pd.read_csv(V1_MASTER)
v1_test_ids = set(v1.loc[v1.split == "test", "id"])

# sanity: v1 ids must point to identical text
chk = v1.merge(df[["id", "tweet"]], on="id", suffixes=("_v1", "_now"))
assert (chk.tweet_v1 == chk.tweet_now).all(), "ID mismatch between v1 master and current data"

s1 = test[(test.stratum == "S1_candidate") & ~test.id.isin(v1_test_ids)]
s0_all = test[test.stratum == "S0_audit"]
s0_in_v1 = s0_all[s0_all.id.isin(v1_test_ids)]
need = 400 - len(s0_in_v1)
s0_new = s0_all[~s0_all.id.isin(v1_test_ids)].sample(need, random_state=SEED_V2)

v2 = pd.concat([s1, s0_new]).sample(frac=1, random_state=SEED_V2).reset_index(drop=True)
rng = np.random.default_rng(SEED_V2)
shared_idx = rng.choice(len(v2), 250, replace=False)
v2["shared"] = 0
v2.loc[shared_idx, "shared"] = 1
rest = v2[v2.shared == 0].index.to_numpy().copy()
rng.shuffle(rest)
half = len(rest) // 2
v2["assigned"] = ""
v2.loc[rest[:half], "assigned"] = "A"
v2.loc[rest[half:], "assigned"] = "B"
v2.loc[v2.shared == 1, "assigned"] = "AB"

master_cols = ["id", "split", "text_md5", "tweet", "label", "cmi", "stratum", "shared", "assigned"]
v2[master_cols].to_csv(f"{OUT}/annotation_master_v2.csv", index=False)

def write_sheet(rows, path):
    out = rows[["id", "tweet"]].copy()
    out["is_codemixed"] = ""
    out["dominance"] = ""
    out["notes"] = ""
    out["shared"] = rows["shared"].values
    out = out.sample(frac=1, random_state=len(path)).reset_index(drop=True)
    out.to_excel(path, index=False)
    wb = load_workbook(path); ws = wb.active
    widths = {"A": 12, "B": 90, "C": 14, "D": 12, "E": 25, "F": 8}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, max_col=2, min_col=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
    ws.freeze_panes = "C2"
    n = ws.max_row
    dv1 = DataValidation(type="list", formula1='"1,0"', allow_blank=True,
                         error="Type 1 or 0 only", showErrorMessage=True)
    dv2 = DataValidation(type="list", formula1='"YO,EN,BAL"', allow_blank=True,
                         error="Type YO, EN or BAL", showErrorMessage=True)
    ws.add_data_validation(dv1); ws.add_data_validation(dv2)
    dv1.add(f"C2:C{n}"); dv2.add(f"D2:D{n}")
    wb.save(path)
    return len(out)

na = write_sheet(v2[v2.assigned.isin(["A", "AB"])], f"{OUT}/annotation_A_v2.xlsx")
nb_ = write_sheet(v2[v2.assigned.isin(["B", "AB"])], f"{OUT}/annotation_B_v2.xlsx")

v1_test = v1[v1.split == "test"]
print("Test set:", len(test), "| S1 candidates:", (test.stratum == "S1_candidate").sum(),
      "| S0:", (test.stratum == "S0_audit").sum())
print("Already in v1 (test):", len(v1_test), "of which S0:", len(s0_in_v1),
      "| v1 shared test rows:", int(v1_test.index.isin(v1.index[:150]).sum()))
print("v2 rows:", len(v2), "| S1 new:", len(s1), "| S0 new:", len(s0_new), "| shared:", 250)
print("Sheet A_v2:", na, "rows | Sheet B_v2:", nb_, "rows")
