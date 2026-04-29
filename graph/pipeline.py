from __future__ import annotations 

from typing import Dict, Any, Optional

from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END


from agents.ingestion import ingestionagent
from agents.imaging import Imagingagent


from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, START, END
from agents.ingestion import ingestionagent
from agents.imaging import Imagingagent
from agents.doctor import doctorescalationagent
from agents.therapy import therapyagent
from agents.pharmcy import PharmacyAgent

class State(TypedDict, total=False):
    xray_file: Any
    pdf_file: Any
    ingestion: Dict[str, Any]
    imaging: Dict[str, Any]
    doctor: Dict[str, Any]
    route: str
    therapy: Dict[str, Any]
    pharmacy: Dict[str, Any]

def node_ingest(state: State) -> State:
    agent = ingestionagent(upload_dir="uploads")
    out = agent.process(
        xray_file=state["xray_file"],
        pdf_file=state.get("pdf_file")
    )
    state["ingestion"] = out
    return state

def node_imaging(state: State) -> State:
    xray_path = state["ingestion"]["xray_path"]
    
    img_agent = Imagingagent(mode="dummy")
    out = img_agent.predict(xray_path)
    state["imaging"] = out
    return state
def decide_route(state: State) -> str:
    """
    Route to 'doctor' if ANY of these:
      - imaging.severity_hint == 'severe'
      - notes contain 'severe' (case-insensitive)
      - notes contain SpO2 < 92
      - notes have red-flag terms
    Else → 'end'
    """
    imaging = state.get("imaging", {}) or {}
    ingestion = state.get("ingestion", {}) or {}
    notes = (ingestion.get("notes") or "").lower()

    
    severity = (imaging.get("severity_hint") or "").lower()
    if severity == "severe":
        return "doctor"

    
    if "severe" in notes:
        return "doctor"

    
    import re
    spo2 = None
    try:
        m = re.search(r"spo2\s*:\s*(\d{2,3})\s*%?", notes, flags=re.I)
        if m:
            spo2 = int(m.group(1))
    except Exception:
        pass
    if spo2 is not None and spo2 < 92:
        return "doctor"

    
    redflags = ["shortness of breath", "difficulty breathing", "chest pain", "confusion"]
    if any(flag in notes for flag in redflags):
        return "doctor"

   
    return "therapy"


def node_doctor(state: State) -> State:
    imaging = state.get("imaging", {}) or {}
    notes = (state.get("ingestion") or {}).get("notes", "") or ""
    from agents.doctor import doctorescalationagent
    doc_agent = doctorescalationagent(doctors_csv="datasets/doctors.csv")  
    state["doctor"] = doc_agent.assess_and_book(imaging, notes=notes)
    return state


def node_therapy(state: State) -> State:
    patient = (state.get("ingestion") or {}).get("patient", {}) or {}
    imaging = state.get("imaging", {}) or {}
    notes = (state.get("ingestion") or {}).get("notes", "") or ""
    th = therapyagent().suggest(patient=patient, imaging=imaging, notes=notes)
    state["therapy"] = th
    return state

def node_pharmacy(state: State) -> State:
    ph = PharmacyAgent(inventory_csv="datasets/inventory.csv",
                       zipcodes_csv="datasets/zipcodes.csv",
                       city_avg_speed_kmh=25.0)
    patient = (state.get("ingestion") or {}).get("patient", {}) or {}
    therapy = state.get("therapy", {}) or {}
    state["pharmacy"] = ph.match(patient=patient, therapy=therapy, top_k=5, make_dummy_reservation=True)
    return state



def node_done(state: State) -> State:
    
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("ingest", node_ingest)
    g.add_node("imaging", node_imaging)
    g.add_node("doctor", node_doctor)
    g.add_node("therapy", node_therapy)
    g.add_node("pharmacy",node_pharmacy)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "imaging")

    
    g.add_conditional_edges(
        "imaging",
        decide_route,
        {"doctor": "doctor", "therapy": "therapy"}
    )
    g.add_edge("therapy", "pharmacy")
    g.add_edge("doctor", END)
    
    g.add_edge("pharmacy",END)

    return g.compile()