from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, re
import pandas as pd

SYMPTOM_TO_INDICATION = {
    "fever": "fever",
    "high fever": "fever",
    "low fever": "fever",
    "cough": "cough",
    "sore throat": "cold",
    "cold": "cold",
    "runny nose": "cold",
    "congestion": "congestion",
    "blocked nose": "congestion",
    "dehydration": "dehydration",
    "fatigue": "fatigue",
    "tired": "fatigue",
    "weakness": "fatigue",
    "pain": "pain",
    "headache": "pain",
    "allergy": "allergy",
    "itching": "allergy",
    "acidity": "acidity",
    "heartburn": "acidity",
}

DEFAULT_SYNTH = pd.DataFrame([
    {"sku": "OTC001", "drug_name": "Paracetamol",        "indication": "fever",       "age_min": 0,  "contra_allergy_keywords": "acetaminophen"},
    {"sku": "OTC002", "drug_name": "Cough Syrup",        "indication": "cough",       "age_min": 12, "contra_allergy_keywords": "dextromethorphan"},
    {"sku": "OTC006", "drug_name": "Saline Nasal Spray", "indication": "congestion",  "age_min": 0,  "contra_allergy_keywords": ""},
])

def _lower_list(x: Optional[List[str]]) -> List[str]:
    if not x: return []
    return [str(i).strip().lower() for i in x if str(i).strip()]

def _parse_allergies_from_notes(notes: str) -> List[str]:
    m = re.search(r"allergies\s*:\s*([^\n\r|]+)", notes or "", flags=re.I)
    if not m: return []
    return _lower_list(re.split(r"[;,/|]", m.group(1)))

class therapyagent:
    def __init__(self, meds_csv: str = os.path.join("dataset", "meds.csv"), interactions_csv: str = os.path.join("dataset", "interactions.csv")):
        self.meds_csv = meds_csv
        self.interactions_csv = interactions_csv
        self._meds = self._load_meds()
        self._ints = self._load_interactions()

    def _load_meds(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.meds_csv):
                df = pd.DataFrame(columns=["sku","drug_name","indication","age_min","contra_allergy_keywords"])
            else:
                df = pd.read_csv(self.meds_csv)
        except Exception:
            df = pd.DataFrame(columns=["sku","drug_name","indication","age_min","contra_allergy_keywords"])
        if df is None or df.empty:
            return DEFAULT_SYNTH.copy()
        df.columns = [str(c).lstrip("\ufeff").strip().lower() for c in df.columns]
        for col in ["sku","drug_name","indication","age_min","contra_allergy_keywords"]:
            if col not in df.columns:
                df[col] = "" if col != "age_min" else 0
        df["sku"] = df["sku"].astype(str).str.strip()
        df["drug_name"] = df["drug_name"].astype(str).str.strip()
        df["indication"] = df["indication"].astype(str).str.strip().str.lower()
        df["age_min"] = pd.to_numeric(df["age_min"], errors="coerce").fillna(0).astype(int)
        df["contra_allergy_keywords"] = df["contra_allergy_keywords"].fillna("").astype(str).str.lower()
        return df

    def _load_interactions(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.interactions_csv):
                df = pd.DataFrame(columns=["drug_a","drug_b","level","note"])
            else:
                df = pd.read_csv(self.interactions_csv)
        except Exception:
            df = pd.DataFrame(columns=["drug_a","drug_b","level","note"])
        if df is None or df.empty:
            return pd.DataFrame(columns=["drug_a","drug_b","level","note"])
        df.columns = [str(c).lstrip("\ufeff").strip().lower() for c in df.columns]
        for col in ["drug_a","drug_b","level","note"]:
            if col not in df.columns: df[col] = ""
        df["drug_a"] = df["drug_a"].astype(str).str.strip()
        df["drug_b"] = df["drug_b"].astype(str).str.strip()
        df["level"]  = df["level"].astype(str).str.lower().str.strip()
        df["note"]   = df["note"].astype(str)
        return df

    def _derive_indications(self, notes: str, label: str) -> List[str]:
        notes_l = (notes or "").lower()
        hits = []
        for k, ind in SYMPTOM_TO_INDICATION.items():
            if k in notes_l:
                hits.append(ind)
        lab = (label or "").lower()
        if lab in ("covid_suspect", "pneumonia"):
            hits += ["fever","cough","congestion","fatigue"]
        if not hits:
            hits = ["fever","cough","cold"]
        out, seen = [], set()
        for h in hits:
            if h not in seen:
                seen.add(h); out.append(h)
        return out

    def _filter_by_indication(self, indications: List[str]) -> pd.DataFrame:
        meds = self._meds if isinstance(self._meds, pd.DataFrame) and not self._meds.empty else DEFAULT_SYNTH.copy()
        return meds[meds["indication"].isin(indications)].copy()

    def _filter_by_age(self, df: pd.DataFrame, age: Optional[int]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty: return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if age is None: return df
        return df[df["age_min"] <= int(age)].copy()

    def _filter_by_allergy(self, df: pd.DataFrame, allergies: List[str], notes: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty: return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        all_allergies = set(_lower_list(allergies)) | set(_parse_allergies_from_notes(notes))
        if not all_allergies: return df
        def ok(row):
            contras = [w.strip() for w in (row.get("contra_allergy_keywords") or "").split(",") if w.strip()]
            contras = [c.lower() for c in contras]
            if "nsaids" in all_allergies:
                contras = [c for c in contras if c not in ("acetaminophen","paracetamol")]
            return not any(a in contras for a in all_allergies)
        return df[df.apply(ok, axis=1)].copy()

    def _choose_recs(self, df: pd.DataFrame, indications: List[str]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty: return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        picks = []
        for ind in indications:
            sub = df[df["indication"] == ind]
            if not sub.empty:
                picks.append(sub.iloc[0])
        if not picks:
            return df.head(3)
        return pd.DataFrame(picks).reset_index(drop=True)

    def _safe_supportive(self, age: Optional[int]) -> pd.DataFrame:
        meds_ok = self._meds if isinstance(self._meds, pd.DataFrame) and not self._meds.empty else DEFAULT_SYNTH.copy()
        prefer = meds_ok[meds_ok["sku"].isin(["OTC001","OTC002","OTC006"])].copy()
        if not prefer.empty:
            prefer = self._filter_by_age(prefer, age)
            if not prefer.empty:
                return prefer.head(3)
        df_ind = meds_ok[meds_ok["indication"].isin(["fever","cough","congestion","dehydration","fatigue","cold"])].copy()
        df_ind = self._filter_by_age(df_ind, age)
        if not df_ind.empty:
            return df_ind.head(3)
        if age is None:
            return meds_ok.head(2)
        return meds_ok[meds_ok["age_min"] <= int(age)].head(2)

    def _interaction_checks(self, sku_list: List[str]) -> Dict[str, Any]:
        if not isinstance(self._ints, pd.DataFrame) or self._ints.empty or len(sku_list) < 2:
            return {"conflicts": [], "cautions": [], "tips": []}
        conflicts, cautions, tips = [], [], []
        seen = set()
        for i in range(len(sku_list)):
            for j in range(i+1, len(sku_list)):
                a, b = sku_list[i], sku_list[j]
                key = tuple(sorted([a,b]))
                if key in seen: 
                    continue
                seen.add(key)
                rows = self._ints[((self._ints["drug_a"] == a) & (self._ints["drug_b"] == b)) |
                                  ((self._ints["drug_a"] == b) & (self._ints["drug_b"] == a))]
                if not isinstance(rows, pd.DataFrame) or rows.empty:
                    continue
                for _, r in rows.iterrows():
                    lv = (r["level"] or "").lower()
                    rec = {"a": a, "b": b, "level": lv or "minor", "note": r["note"]}
                    if lv == "major": conflicts.append(rec)
                    elif lv == "moderate": cautions.append(rec)
                    else: tips.append(rec)
        return {"conflicts": conflicts, "cautions": cautions, "tips": tips}

    def suggest(self, patient: Dict[str,Any], imaging: Dict[str,Any], notes: str = "") -> Dict[str,Any]:
        if not isinstance(self._meds, pd.DataFrame) or self._meds is None:
            self._meds = DEFAULT_SYNTH.copy()
        if not isinstance(self._ints, pd.DataFrame) or self._ints is None:
            self._ints = pd.DataFrame(columns=["drug_a","drug_b","level","note"])
        age = patient.get("age")
        allergies = patient.get("allergies") or []
        top_label = (imaging.get("top_label") or "normal").lower()
        indications = self._derive_indications(notes, top_label)
        by_ind = self._filter_by_indication(indications)
        if not isinstance(by_ind, pd.DataFrame) or by_ind.empty:
            by_ind = self._meds[self._meds["indication"].isin(["fever","cough","cold","congestion","fatigue"])]
        age_ok = self._filter_by_age(by_ind, age)
        safe = self._filter_by_allergy(age_ok, allergies, notes)
        if not isinstance(safe, pd.DataFrame) or safe.empty:
            safe = self._safe_supportive(age)
        chosen = self._choose_recs(safe, indications)
        recs = []
        if isinstance(chosen, pd.DataFrame) and not chosen.empty:
            for _, r in chosen.iterrows():
                recs.append({
                    "sku": r["sku"],
                    "drug_name": r["drug_name"],
                    "indication": r["indication"],
                    "age_min": int(r["age_min"]),
                    "contra_allergy_keywords": r.get("contra_allergy_keywords",""),
                    "how_to_use": self._usage_hint(r["drug_name"], r["indication"]),
                })
        if not recs:
            synth = DEFAULT_SYNTH.copy()
            synth = self._filter_by_age(synth, age)
            recs = []
            for _, r in synth.iterrows():
                recs.append({
                    "sku": r["sku"],
                    "drug_name": r["drug_name"],
                    "indication": r["indication"],
                    "age_min": int(r["age_min"]),
                    "contra_allergy_keywords": r.get("contra_allergy_keywords",""),
                    "how_to_use": self._usage_hint(r["drug_name"], r["indication"]),
                })
        inter = self._interaction_checks([r["sku"] for r in recs])
        return {
            "condition": top_label,
            "derived_indications": indications,
            "patient_age": age,
            "patient_allergies": allergies,
            "recommendations": recs,
            "interactions": inter,
            "mitigations": [],
            "non_pharm": [
                "Rest and adequate hydration.",
                "Steam inhalation / saline gargles for cough/cold.",
                "Record temperature and symptoms twice daily.",
            ],
            "safety": [
                "Read the OTC label; follow dosing exactly.",
                "Do NOT combine multiple products with the same active ingredient (e.g., multiple acetaminophen sources).",
                "Avoid suggested products if you experience allergy symptoms; seek care if symptoms worsen.",
                "If fever > 72 hours, SpO₂ < 92%, chest pain, confusion, or breathlessness → escalate to a doctor.",
            ],
        }

    def _usage_hint(self, drug_name: str, indication: str) -> str:
        dn = (drug_name or "").lower()
        ind = (indication or "").lower()
        if "paracetamol" in dn or ind == "fever":
            return "500 mg every 6–8 hours for fever; max 3 g/day (adult)."
        if "cough" in dn or ind == "cough":
            return "Per label; may cause drowsiness. Avoid driving if sedated."
        if "saline" in dn or ind == "congestion":
            return "2–3 sprays per nostril as needed."
        if "rehydration" in dn or ind == "dehydration":
            return "Mix as directed; sip small frequent amounts."
        if "cetirizine" in dn or ind == "allergy":
            return "Once daily at night if sedating."
        return "Use as per OTC label."




    

    
