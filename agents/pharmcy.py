from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, math, time, uuid
import pandas as pd

def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lstrip("\ufeff").strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _coerce_float(x):
    try: return float(x)
    except Exception: return None

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

class PharmacyAgent:
    def __init__(self,
        inventory_csv: str = os.path.join("dataset","inventory.csv"),
        zipcodes_csv : str = os.path.join("dataset","zipcodes.csv"),
        city_avg_speed_kmh: float = 25.0,
    ):
        self.inventory_csv = inventory_csv
        self.zipcodes_csv  = zipcodes_csv
        self.city_speed    = max(5.0, float(city_avg_speed_kmh))
        self._inv = self._load_inventory()
        self._ph  = self._build_pharmacies_from_inventory_and_zip()

    def _load_inventory(self) -> pd.DataFrame:
        if not os.path.exists(self.inventory_csv):
            return pd.DataFrame(columns=["pharmacy_id","sku","drug_name","form","strength","price","qty"])
        df = pd.read_csv(self.inventory_csv)
        df = _norm_columns(df)
        # tolerate alternative names
        rename = {
            "pharmacyid":"pharmacy_id",
            "drug":"drug_name",
            "quantity":"qty",
        }
        for k,v in rename.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        for c in ["pharmacy_id","sku","drug_name","form","strength","price","qty"]:
            if c not in df.columns: df[c] = None
        df["pharmacy_id"] = df["pharmacy_id"].astype(str).str.strip()
        df["sku"]         = df["sku"].astype(str).str.strip()
        df["price"]       = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["qty"]         = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
        return df

    def _load_zip(self) -> pd.DataFrame:
        if not os.path.exists(self.zipcodes_csv):
            return pd.DataFrame(columns=["pincode","lat","lon"])
        df = pd.read_csv(self.zipcodes_csv)
        df = _norm_columns(df)
        # tolerate alternative headers
        alt_lat = next((c for c in df.columns if c in ("lat","latitude")), None)
        alt_lon = next((c for c in df.columns if c in ("lon","longitude")), None)
        if "lat" not in df.columns and alt_lat: df["lat"] = df[alt_lat]
        if "lon" not in df.columns and alt_lon: df["lon"] = df[alt_lon]
        if "pincode" not in df.columns: df["pincode"] = None
        df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
        df["lat"]     = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"]     = pd.to_numeric(df["lon"], errors="coerce")
        return df

    def _build_pharmacies_from_inventory_and_zip(self) -> pd.DataFrame:
        inv = self._inv.copy()
        if inv.empty:
            return pd.DataFrame(columns=["pharmacy_id","name","lat","lon","pincode"])
        unique_ph = inv[["pharmacy_id"]].drop_duplicates().reset_index(drop=True)

        zipdf = self._load_zip().reset_index(drop=True)
    # If zipcodes are missing or fewer, pad/cycle them so we never lose geo
        if zipdf.empty:
            zipdf = pd.DataFrame([{"pincode": None, "lat": None, "lon": None}])
        if len(zipdf) < len(unique_ph):
            reps = int((len(unique_ph) + len(zipdf) - 1) // len(zipdf))
            zipdf = pd.concat([zipdf] * reps, ignore_index=True)
        zipdf = zipdf.iloc[:len(unique_ph)].reset_index(drop=True)

        ph = pd.concat([unique_ph, zipdf[["lat","lon","pincode"]]], axis=1)
        ph["name"] = ph["pharmacy_id"]
        return ph


    def _resolve_patient_geo(self, patient: Dict[str,Any], ph_df: pd.DataFrame) -> Dict[str,float]:
        # 1) direct lat/lon
        lat = _coerce_float(patient.get("lat")); lon = _coerce_float(patient.get("lon"))
        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon, "source": "patient_latlon"}

        # 2) pincode centroid from zip file
        pin = patient.get("pincode")
        z = self._load_zip()
        z_ok = z[pd.notna(z["lat"]) & pd.notna(z["lon"])]
        if pin is not None and not z_ok.empty:
            m = z_ok[z_ok["pincode"].astype("Int64") == pd.Series([pin]).astype("Int64").iloc[0]]
            if not m.empty:
                return {"lat": float(m.iloc[0]["lat"]), "lon": float(m.iloc[0]["lon"]), "source": "pincode_centroid"}

        # 3) fallback: first zipcode row
        if not z_ok.empty:
            return {"lat": float(z_ok.iloc[0]["lat"]), "lon": float(z_ok.iloc[0]["lon"]), "source": "zip_fallback"}

        ph_ok = ph_df.dropna(subset=["lat","lon"])
        if not ph_ok.empty:
            return {"lat": float(ph_ok.iloc[0]["lat"]), "lon": float(ph_ok.iloc[0]["lon"]), "source": "pharmacy_fallback"}

        
        return {"lat": None, "lon": None, "source": "unresolved"}

    def _eta_minutes(self, km: float) -> int:
        return max(1, int(round((km / self.city_speed) * 60.0)))

    def match(self, patient: Dict[str,Any], therapy: Dict[str,Any], top_k: int = 5, make_dummy_reservation: bool = True) -> Dict[str,Any]:
        recs = therapy.get("recommendations") or []
        want_skus = [r.get("sku") for r in recs if r.get("sku")]
        want_skus = [s for s in want_skus if isinstance(s, str)]

        ph = self._ph.copy()
        pgeo = self._resolve_patient_geo(patient or {}, ph)
        p_loc = {"lat": pgeo.get("lat"), "lon": pgeo.get("lon"), "source": pgeo.get("source")}

        if not want_skus:
            return {"requested_skus": [], "patient_location": p_loc, "matches": [], "best_nearby": {}, "best_price": {}, "note": "No SKUs to match from therapy.", "debug": {"stage": "no_skus"}}
        if pgeo.get("lat") is None or pgeo.get("lon") is None:
            return {"requested_skus": want_skus, "patient_location": p_loc, "matches": [], "best_nearby": {}, "best_price": {}, "note": "Could not resolve patient location.", "debug": {"stage": "no_patient_geo"}}

        inv = self._inv
        if inv.empty or ph.empty:
            return {"requested_skus": want_skus, "patient_location": p_loc, "matches": [], "best_nearby": {}, "best_price": {}, "note": "Inventory or zipcode mapping missing/empty.", "debug": {"stage": "empty_inv_or_ph", "inv_rows": len(inv), "ph_rows": len(ph)}}

        inv = inv[(inv["sku"].isin(want_skus)) & (inv["qty"] > 0)].copy()
        if inv.empty:
            return {"requested_skus": want_skus, "patient_location": p_loc, "matches": [], "best_nearby": {}, "best_price": {}, "note": "No stock found for requested SKUs.", "debug": {"stage": "no_stock_for_skus", "skus": want_skus}}

        merged = inv.merge(ph, on="pharmacy_id", how="left")
        merged["lat"] = merged["lat"].apply(lambda x: float(x) if pd.notna(x) else None)
        merged["lon"] = merged["lon"].apply(lambda x: float(x) if pd.notna(x) else None)
        before_drop = len(merged)
        merged = merged.dropna(subset=["lat","lon"]).copy()
        if merged.empty:
            return {"requested_skus": want_skus, "patient_location": p_loc, "matches": [], "best_nearby": {}, "best_price": {}, "note": "No geocoded pharmacies (all rows lost after geo drop).", "debug": {"stage": "no_geocoded", "merged_before_drop": before_drop, "ph_rows": len(ph)}}

        merged = inv.merge(ph, on="pharmacy_id", how="left")
        merged["lat"] = merged["lat"].apply(_coerce_float)
        merged["lon"] = merged["lon"].apply(_coerce_float)
        merged = merged.dropna(subset=["lat","lon"]).copy()
        if merged.empty:
            return {"requested_skus": want_skus, "patient_location": p_loc, "matches": [], "note": "No geocoded pharmacies."}

        merged["distance_km"] = merged.apply(
            lambda r: _haversine_km(float(p_loc["lat"]), float(p_loc["lon"]), float(r["lat"]), float(r["lon"])),
            axis=1
        )
        merged["eta_min"] = merged["distance_km"].apply(self._eta_minutes)

        def _agg(g: pd.DataFrame) -> pd.Series:
            items = g[["sku","drug_name","form","strength","price","qty"]].to_dict("records")
            covers = set(g["sku"].tolist())
            total_price = float(g["price"].sum())
            dist_min = float(g["distance_km"].min())
            eta_min  = int(g.loc[g["distance_km"].idxmin(), "eta_min"])
            lat = float(g.iloc[0]["lat"]); lon = float(g.iloc[0]["lon"])
            name = g.iloc[0].get("name") or g.iloc[0]["pharmacy_id"]
            pincode = g.iloc[0].get("pincode")
            pincode = int(pincode) if pd.notna(pincode) else None
            return pd.Series({
                "distance_km": dist_min,
                "eta_min": eta_min,
                "items": items,
                "covers": covers,
                "covers_count": len(covers),
                "total_price_for_available": round(total_price, 2),
                "lat": lat,
                "lon": lon,
                "name": name,
                "pincode": pincode,
            })

        agg = merged.groupby("pharmacy_id").apply(_agg).reset_index()
        needed = set(want_skus)

        best_nearby_row = agg.sort_values(by=["covers_count","distance_km"], ascending=[False,True]).iloc[0]
        best_nearby = best_nearby_row.to_dict()
        best_nearby["coverage_ratio"] = round(len(best_nearby["covers"]) / max(1,len(needed)), 2)

        full_cover = agg[agg["covers"].apply(lambda s: set(s) >= needed)]
        if not full_cover.empty:
            best_price_row = full_cover.sort_values(by=["total_price_for_available","distance_km"]).iloc[0]
        else:
            best_price_row = agg.sort_values(by=["covers_count","total_price_for_available","distance_km"],
                                             ascending=[False,True,True]).iloc[0]
        best_price = best_price_row.to_dict()
        best_price["coverage_ratio"] = round(len(best_price["covers"]) / max(1,len(needed)), 2)

        ranked = agg.sort_values(by=["covers_count","distance_km","total_price_for_available"],
                                 ascending=[False,True,True]).reset_index(drop=True)
        all_matches: List[Dict[str,Any]] = []
        for _, r in ranked.iterrows():
            all_matches.append({
                "pharmacy_id": r["pharmacy_id"],
                "name": r["name"],
                "pincode": r["pincode"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "distance_km": round(float(r["distance_km"]), 2),
                "eta_min": int(r["eta_min"]),
                "items": r["items"],
                "covers": list(r["covers"]),
                "covers_count": int(r["covers_count"]),
                "coverage_ratio": round(len(r["covers"]) / max(1,len(needed)), 2),
                "total_price_for_available": float(r["total_price_for_available"]),
            })

        reservation = None
        reserve_items = []
        covers = set(best_nearby["covers"])
        for sku in want_skus:
            if sku in covers:
                item = next((i for i in best_nearby["items"] if i["sku"] == sku), None)
                if item:
                    reserve_items.append({"sku": sku, "qty": 1, "price": float(item.get("price", 0.0))})
        reservation = {
            "reservation_id": str(uuid.uuid4())[:8],
            "pharmacy_id": best_nearby_row["pharmacy_id"],
            "pharmacy_name": best_nearby_row["name"],
            "eta_min": int(best_nearby_row["eta_min"]),
            "distance_km": round(float(best_nearby_row["distance_km"]), 2),
            "items": reserve_items,
            "reserved_total_price": round(sum(x["price"] for x in reserve_items), 2),
            "expires_at_epoch": int(time.time()) + 15*60
        }

        return {
             "requested_skus": want_skus,
            "patient_location": p_loc,
            "best_nearby": best_nearby,
            "best_price": best_price,
            "all_matches": all_matches[:top_k],
            "note": f"Pharmacies geocoded by inventory→zip row order; ETA uses {self.city_speed} km/h.",
            "reservation": reservation,
            "debug": {
                "inv_rows": int(len(self._inv)),
                "ph_rows": int(len(ph)),
                "merged_rows": int(len(merged)),
                "unique_pharmacies": int(self._inv['pharmacy_id'].nunique()),
                "skus_requested": want_skus,
            }
        }


