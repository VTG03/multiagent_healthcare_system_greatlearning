from agents.therapy import therapyagent

def test_therapy_basic_recommendation():
    agent = therapyagent(
        meds_csv="datasets/meds.csv",
        interactions_csv="datasets/interactions.csv"
    )
    patient = {"age": 40, "allergies": []}
    imaging = {"top_label": "normal"}
    notes = "Symptoms: fever, cough"
    result = agent.suggest(patient, imaging, notes)

    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0
    assert all("drug_name" in r for r in result["recommendations"])
