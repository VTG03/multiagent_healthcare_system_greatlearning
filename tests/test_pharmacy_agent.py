from agents.pharmcy import PharmacyAgent

def test_pharmacy_find_nearest():
    agent = PharmacyAgent(
        inventory_csv="datasets/inventory.csv",
        zipcodes_csv="datasets/zipcodes.csv"
    )

   
    patient = {"pincode": 400703, "lat": 19.07, "lon": 73.0}
    
    therapy = {"recommendations": [
        {"sku": "OTC001"}, {"sku": "OTC002"}
    ]}

    result = agent.match(patient, therapy)

    assert "best_nearby" in result
    assert "reservation" in result
    assert isinstance(result["all_matches"], list)
    assert len(result["all_matches"]) > 0
