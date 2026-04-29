🩺 Multi-Agent Healthcare Assistant

An end-to-end AI healthcare pipeline built using Streamlit, LangGraph, and Python, where specialized agents collaborate to analyze patient data, assess severity, suggest safe treatments, and locate the nearest pharmacy with available stock.

🌟 Overview

This project demonstrates a multi-agent reasoning system that mimics a clinical workflow — from X-ray triage to doctor escalation, therapy recommendation, and pharmacy matching — all orchestrated by a conditional graph pipeline.

The assistant takes patient inputs (X-ray, PDF, notes, vitals) and passes them through a chain of intelligent agents that each perform one role in the decision-making process.

🧠 Core Workflow

Ingestion Agent – Extracts structured patient data (age, allergies, pincode, latitude/longitude, SpO₂, notes) from uploaded files.

Imaging Agent – Classifies chest X-rays as normal, pneumonia, or COVID-suspect using a dummy prediction model and returns severity.

Routing Logic – Dynamically decides whether the patient should:

Be escalated to a doctor (if severe / low SpO₂ / red-flag words), or

Proceed to therapy suggestions.

Doctor Escalation Agent – Finds the right specialist based on the diagnosis, matches availability from doctors.csv, and books a tele-consultation slot.

Therapy Agent – Suggests over-the-counter (OTC) medicines from meds.csv after checking:

Patient’s age restrictions

Allergy conflicts

Drug-to-drug interactions from interactions.csv
It also returns safety disclaimers and non-pharmacological tips.

Pharmacy Agent – Matches the therapy’s recommended medicines with the nearest pharmacy (using patient’s location and stock info from inventory.csv + zipcodes.csv) and computes:

Distance (haversine formula)

Estimated delivery time

Total cost and availability

All agents are coordinated through a LangGraph pipeline, allowing conditional execution — e.g., mild → Therapy Agent, severe → Doctor Escalation Agent.
⚙️ Installation
git clone https://github.com/VTG03/multiagent_healthcare_system_greatlearning.git
cd Agent_healthcare
python -m venv .venv
.venv\Scripts\activate        # on Windows
# or
source .venv/bin/activate     # on macOS/Linux

pip install -r requirements.txt
streamlit run app.py

🧾 Example Input & Output
Input
{
  "patient": {
    "age": 47,
    "allergies": ["nsaids"],
    "pincode": 400703,
    "lat": 19.07,
    "lon": 73.0
  },
  "notes": "SpO2: 96%, cough and mild fever"
}

Output (Simplified)
{
  "condition": "normal",
  "severity": "mild",
  "recommendations": [
    {"sku": "OTC001", "drug_name": "Paracetamol"},
    {"sku": "OTC002", "drug_name": "Cough Syrup"}
  ],
  "nearest_pharmacy": {
    "pharmacy_id": "ph001",
    "distance_km": 2.3,
    "eta_min": 6,
    "total_cost": 7.0
  }
}


Testing

All agents include independent tests. Run them with:

python -m pytest -q


Tests include:

TherapyAgent: Verifies medicine filtering and safe recommendation logic.

PharmacyAgent: Checks distance computation and stock matching.

Routing Logic: Ensures conditional branching (severe → doctor, mild → therapy).

🧭 Design Highlights

Agentic Modularity: Each task handled by a self-contained agent.

Conditional Graph Orchestration: LangGraph used to control flow based on real data.

Safety-Aware Therapy Suggestions: Considers age, allergy, and interactions before advice.

Dynamic Pharmacy Matching: Geo-based nearest store recommendation.

Streamlit Frontend: Interactive and user-friendly workflow UI.

Test Coverage: Independent pytest modules for agents.

🚀 Future Work

Integrate real CNN model for X-ray classification.

Connect with live doctor & pharmacy APIs.

Add FHIR-compliant patient record export.

Add multi-language symptom parsing (spaCy / Transformers).

Deploy on Streamlit Cloud or Hugging Face Spaces.
