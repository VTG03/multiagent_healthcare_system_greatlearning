from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import os, io, re, json, tempfile

try:
    import PyPDF2
except Exception:
    PyPDF2=None

def read_pdf_text(pdf_bytes:bytes)->str:
    """Extract text from PDF using PyPDF2; fallback to OCR (pytesseract+pdf2image) if installed."""
    text=""
    if PyPDF2:
        try:
            reader=PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            pages=[]
            for p in reader.pages:
                pages.append(p.extract_text()or"")
            text="\n".join(pages)
        except Exception:
            text=""
    if not text.strip():
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(pdf_bytes)
            ocr_texts = [pytesseract.image_to_string(img) for img in images]
            text = "\n".join(ocr_texts)
        except Exception:
            text=""
    return text

def _parse_fields_from_text(txt: str) -> Dict[str, Any]:
    """Regex-based parsing for Age, Allergies, Symptoms, Category, Pincode, Lat/Lon, SpO2.
    Returns normalized values (age int, allergies list, floats for lat/lon, etc.).
    """

    def find(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, txt, flags)
        if not m:
            return None
        
        last = m.lastindex or 1
        return (m.group(last) or "").strip()

    
    age_s = find(r"\bAge\s*[:\-]\s*(\d{1,3})\b")
    age = int(age_s) if age_s and age_s.isdigit() else None

    allergies_s = find(r"\bAllerg(?:y|ies)\s*[:\-]\s*([^\n\r]+)")
    allergies: List[str] = []
    if allergies_s:
        clean = allergies_s.strip().lower()
        if clean in ("none", "no", "nka", "na", "nil", "-"):
            allergies = []
        else:
            allergies = [a.strip().lower() for a in re.split(r"[,\|/;]+", allergies_s) if a.strip()]

    symptoms = find(r"\b(?:Primary\s+Symptoms|Symptoms)\s*[:\-]\s*([^\n\r]+)")
    
    if not symptoms:
        symptoms = find(r"\b(?:Presenting Complaint|Clinical History)\s*[:\-]\s*([^\n\r]+)")

    
    category = find(r"\bCategory\s*[:\-]\s*([^\n\r]+)")

    
    pin_s = find(r"\bP(?:in)?code\s*[:\-]\s*(\d{6})\b")
    if not pin_s:
        m = re.search(r"\b(\d{6})\b", txt)
        pin_s = m.group(1) if m else None
    pincode = int(pin_s) if pin_s and pin_s.isdigit() else None

   
    lat_s = find(r"\bLatitude\s*[:\-]\s*([-\d\.]+)")
    lon_s = find(r"\bLongitude\s*[:\-]\s*([-\d\.]+)")
    lat = float(lat_s) if lat_s and re.match(r"^-?\d+(\.\d+)?$", lat_s) else None
    lon = float(lon_s) if lon_s and re.match(r"^-?\d+(\.\d+)?$", lon_s) else None

    
    spo2_s = find(r"\bSp\s*O2\s*[:\-]?\s*(\d{2,3})\s*%?")
    if spo2_s is None:
        spo2_s = find(r"\bSpo2\s*[:\-]?\s*(\d{2,3})\s*%?")
    spo2 = int(spo2_s) if spo2_s and spo2_s.isdigit() else None

    return {
        "age": age,
        "allergies": allergies,
        "symptoms": symptoms,
        "category": category,
        "pincode": pincode,
        "lat": lat,
        "lon": lon,
        "spo2": spo2,
    }



    
  

def _mask_basic_pii(text: str) -> str:
    """Light de-identification (mask full names like 'John Doe')."""
    return re.sub(r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b", "[Patient]", text)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class ingestionagent:
    """Minimal ingestion agent:
      - validate & save X-ray (required)
      - save PDF (optional)
      - OCR PDF → parse Age/Allergies/Symptoms/Category/Pincode/Lat/Lon
      - return assignment-compatible JSON:
        { "patient": { "age": int, "allergies": [...] }, "xray_path": "...", "notes": "..." }
      - includes extra geo fields in 'patient' if present
    """
    def __init__(self,upload_dir:str= "uploads"):
        self.upload_dir = upload_dir
        _ensure_dir(upload_dir)

    def process(self,xray_file,pdf_file:Optional[Any]=None,)->dict[str,Any]:
        if xray_file is None:
            raise ValueError("image is required for ingestion")
        img_name = xray_file.name
        if not any(img_name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            raise ValueError("Unsupported image format. Please upload PNG/JPG.")
        xray_path=os.path.join(self.upload_dir,img_name)
        with open(xray_path, "wb") as f:
            f.write(xray_file.read())
        pdf_text = ""
        pdf_name = pdf_file.name
        if not pdf_name.lower().endswith(".pdf"):
                raise ValueError("Unsupported PDF format.")
        pdf_path = os.path.join(self.upload_dir, pdf_name)
        with open(pdf_path, "wb") as f:
                f.write(pdf_file.read())
            
        with open(pdf_path, "rb") as f:
            pdf_text = read_pdf_text(f.read())
        pdf_text = _mask_basic_pii(pdf_text)

        fields = _parse_fields_from_text(pdf_text) if pdf_text else {
            "age": None, "allergies": [], "symptoms": None, "category": None,
            "pincode": None, "lat": None, "lon": None, "spo2": None
        }

        notes_parts = []
        if fields.get("symptoms"): notes_parts.append(f"Symptoms: {fields['symptoms']}")
        if fields.get("category"): notes_parts.append(f"Category: {fields['category']}")
        if fields.get("spo2") is not None: notes_parts.append(f"SpO2: {fields['spo2']}%")
        notes = " | ".join(notes_parts) if notes_parts else ""

        result = {
            "patient": {
                "age": fields["age"],
                "allergies": fields["allergies"],
               
                "pincode": fields["pincode"],
                "lat": fields["lat"],
                "lon": fields["lon"],
            },
            "xray_path": xray_path,
            "notes": notes,
            
           
        }
        return result


        
        

        
    

