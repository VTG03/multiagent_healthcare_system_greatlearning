from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, ast, re, math
import pandas as pd
from datetime import datetime, timezone, timedelta

SPECIALTY_BY_LABEL = {
    "pneumonia": "Pulmonologist",
    "covid_suspect": "Pulmonologist",
    "normal": "General Physician",
}

class doctorescalationagent:
    def __init__(self, doctors_csv: str = os.path.join("datasets", "doctors.csv")):
        self.doctors_csv = doctors_csv

    
    def _load_doctors(self) -> pd.DataFrame:
        """Always return a DataFrame (possibly empty). Never None."""
        try:
            print(f"[DoctorEscalationAgent] Loading doctors from: {self.doctors_csv}")
            if not os.path.exists(self.doctors_csv):
                print("[DoctorEscalationAgent] File not found -> returning empty DataFrame")
                return pd.DataFrame(columns=["doctor_id", "name", "specialty", "tele_slots"])

            df = pd.read_csv(self.doctors_csv)
        except Exception as e:
            print(f"[DoctorEscalationAgent] read_csv error: {e} -> returning empty DataFrame")
            return pd.DataFrame(columns=["doctor_id", "name", "specialty", "tele_slots"])

        def parse_slots(val):
            
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return []
            if isinstance(val, list):
                return [str(s).strip() for s in val if s and str(s).strip().lower() != "nan"]

            s = str(val).strip()
            if not s or s.lower() == "nan":
                return []
            s = (s.replace("“", '"').replace("”", '"')
                   .replace("’", "'").replace("‘", "'")
                   .replace('""', '"'))
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if x and str(x).strip().lower() != "nan"]
                if isinstance(parsed, str) and parsed.strip():
                    return [parsed.strip()]
            except Exception:
                if s.startswith("[") and s.endswith("]"):
                    inner = s[1:-1]
                    items = [i.strip().strip('"').strip("'") for i in inner.split(",") if i.strip()]
                    return [x for x in items if x and x.lower() != "nan"]
            return [s] if s.lower() != "nan" else []

        if "tele_slots" not in df.columns:
            
            df["tele_slots"] = [[] for _ in range(len(df))]
        else:
            df["tele_slots"] = df["tele_slots"].apply(parse_slots)

        
        for col in ["doctor_id", "name", "specialty"]:
            if col not in df.columns:
                df[col] = None

        return df

    def _pick_specialty(self, top_label: str) -> str:
        top_label = (top_label or "").lower().strip()
        return SPECIALTY_BY_LABEL.get(top_label, "General Physician")

    
    def _choose_doctor(self, df: pd.DataFrame, specialty: str) -> Optional[Dict[str, Any]]:
        """df is guaranteed DataFrame. Return a booking dict or synthesize a slot."""
        try:
            if df is None or getattr(df, "empty", True):
                print("[DoctorEscalationAgent] Empty doctors DataFrame")
                return None

            cand = df[df["specialty"].astype(str).str.contains(specialty, case=False, na=False)].copy()
            if cand.empty:
                cand = df.copy()

            rows = []
            now = datetime.now(timezone.utc)

            for _, r in cand.iterrows():
                slots = r.get("tele_slots") or []
                cleaned = []
                for s in slots:
                    s = str(s).strip()
                    if not s or s.lower() == "nan":
                        continue
                    dt = None
                    try:
                        s_iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
                        dt = datetime.fromisoformat(s_iso)
                    except Exception:
                        pass
                    cleaned.append((s, dt))

                future = [x for x in cleaned if x[1] and x[1] >= now]
                past   = [x for x in cleaned if x[1] and x[1] <  now]
                unknown= [x for x in cleaned if x[1] is None]

                ordered = sorted(future, key=lambda x: x[1]) \
                        + sorted(past, key=lambda x: x[1]) \
                        + sorted(unknown, key=lambda x: x[0])

                if ordered:
                    rows.append((r.to_dict(), ordered[0][0]))  

            
            if not rows:
                first = cand.iloc[0].to_dict()
                synth_dt = (now + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
                synth = synth_dt.isoformat().replace("+00:00", "Z")
                print(f"[DoctorEscalationAgent] Synthesizing slot: {synth}")
                return {
                    "doctor_id": first.get("doctor_id"),
                    "name": first.get("name"),
                    "specialty": first.get("specialty"),
                    "booked_slot": synth,
                }

            rows.sort(key=lambda t: t[1])
            picked_row, booked = rows[0]
            return {
                "doctor_id": picked_row.get("doctor_id"),
                "name": picked_row.get("name"),
                "specialty": picked_row.get("specialty"),
                "booked_slot": booked,
            }
        except Exception as e:
            print(f"[DoctorEscalationAgent] _choose_doctor error: {e}")
            return None

    
    def assess_and_book(self, imaging: Dict[str, Any], notes: str = "") -> Dict[str, Any]:
        import re
        reasons: List[str] = []

        severity = (imaging.get("severity_hint") or "").lower()
        top_label = (imaging.get("top_label") or "normal").lower()
        if severity == "severe":
            reasons.append("Imaging severity: severe")

        text = (notes or "").lower()
        if "severe" in text:
            reasons.append("Report notes contain 'severe'")

        spo2 = None
        m = re.search(r"spo2\s*:\s*(\d{2,3})\s*%?", text, flags=re.I)
        if m:
            try:
                spo2 = int(m.group(1))
                if spo2 < 92:
                    reasons.append(f"Low SpO₂: {spo2}%")
            except ValueError:
                pass

        redflags = ["shortness of breath", "difficulty breathing", "chest pain", "confusion"]
        if any(flag in text for flag in redflags):
            reasons.append("Red-flag symptom detected")

        if not reasons:
            return {
                "escalate": False,
                "message": "No urgent escalation needed. Case is not severe by current rules.",
                "booking": None,
                "reasons": [],
            }

        df = self._load_doctors()  
        specialty = self._pick_specialty(top_label)
        booking = self._choose_doctor(df, specialty)

        if not booking or not booking.get("booked_slot"):
            return {
                "escalate": True,
                "message": "🚨 Severe indicators present. Please visit a qualified doctor ASAP (no slots found).",
                "booking": None,
                "reasons": reasons,
            }

        msg = (
            f"🚨 Severe indicators present. Appointment booked with **{booking['name']}** "
            f"({booking['specialty']}) at **{booking['booked_slot']}**."
        )
        return {"escalate": True, "message": msg, "booking": booking, "reasons": reasons}


