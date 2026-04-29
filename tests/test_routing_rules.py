from graph.pipeline import decide_route

def test_routing_to_doctor_on_severe():
   
    state = {
        "imaging": {"severity_hint": "severe"},
        "ingestion": {"notes": "SpO2: 95%"}
    }
    assert decide_route(state) == "doctor"

def test_routing_to_therapy_on_mild():
    
    state = {
        "imaging": {"severity_hint": "mild"},
        "ingestion": {"notes": "Symptoms: fever"}
    }
    assert decide_route(state) == "therapy" or decide_route(state) == "end"
