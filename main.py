import os
import json
import re
import logging
from datetime import date
from typing import TypedDict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId, SON
from pymongo import MongoClient
import requests
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and validate environment variables
def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise EnvironmentError(f"Missing {var_name} in environment")
    logger.info(f"Loaded {var_name}")
    return value

# Load environment variables with validation
try:
    logger.info("Loading environment variables...")
    MONGODB_URI = get_required_env("MONGODB_URI")
    GOOGLE_API_KEY = get_required_env("GOOGLE_API_KEY")
    MALE_BN_API_URL = get_required_env("MALE_BN_API_URL")
    FEMALE_BN_API_URL = get_required_env("FEMALE_BN_API_URL")
except EnvironmentError as e:
    logger.error(f"Environment configuration error: {str(e)}")
    raise

# Initialize MongoDB client
try:
    logger.info("Initializing MongoDB connection...")
    tmp_client = MongoClient(MONGODB_URI)
    db = tmp_client.get_default_database()
    patients_col = db["patients"]
    metrics_col = db["healthmetrics"]
    medications_col = db["medications"]
    medicines_col = db["medicines"]
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Initialize LLM
try:
    logger.info("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Pydantic models
class Medication(BaseModel):
    medicationName: str
    dosage: str
    frequency: Optional[str] = None

class Medicine(BaseModel):
    name: str
    specialization: str
    description: Optional[str] = None

class Recommendations(BaseModel):
    patient_recommendations: Optional[List[str]] = None
    diet_plan: Optional[dict] = None
    exercise_plan: Optional[dict] = None
    nutrition_targets: Optional[dict] = None
    doctor_recommendations: Optional[List[str]] = None

class State(TypedDict):
    patient_data: dict
    sent_for: int
    risk_probabilities: dict
    recommendations: Recommendations
    selected_patient_recommendations: List[str]
    current_medications: List[Medication]
    available_medicines: List[Medicine]

# Helper functions
def parse_probability(prob_str: str) -> float:
    return float(prob_str.strip('%')) / 100

def get_risk_probabilities(patient_data: dict) -> dict:
    payload = patient_data.copy()
    payload.pop('gender', None)
    gender = patient_data.get('gender')
    
    if gender == 'M':
        api_url = MALE_BN_API_URL
    elif gender == 'F':
        api_url = FEMALE_BN_API_URL
    else:
        raise ValueError("Invalid gender in patient data; must be 'M' or 'F'")

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"BN API request failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"BN service error: {str(e)}")

def classify_recommendation(text: str) -> str:
    t = text.lower()
    if 'exercise' in t:
        return 'Physical Activity'
    if 'diet' in t or 'nutrition' in t:
        return 'Diet'
    if 'smoking' in t:
        return 'Smoking Cessation'
    return 'Other'

def adjust_metrics(data: dict, kind: str) -> dict:
    d = data.copy()
    if kind == 'Physical Activity':
        d['Exercise_Hours_Per_Week'] = d.get('Exercise_Hours_Per_Week', 0) + 2
    if kind == 'Diet':
        if 'BMI' in d:
            d['BMI'] = max(d['BMI'] - 1, 0)
        if 'glucose' in d:
            d['glucose'] = max(d['glucose'] - 10, 0)
    if kind == 'Smoking Cessation':
        d['is_smoking'] = False
    return d

def is_effective(orig: dict, new: dict) -> bool:
    o = orig['Health Risk Probabilities']
    n = new['Health Risk Probabilities']
    o_d = parse_probability(o['Diabetes'])
    o_c = parse_probability(o['Heart Disease'])
    n_d = parse_probability(n['Diabetes'])
    n_c = parse_probability(n['Heart Disease'])
    return ((n_d < o_d - 0.05 and n_c <= o_c + 0.01) or
            (n_c < o_c - 0.05 and n_d <= o_d + 0.01))

def get_patient_medications(patient_id: str) -> List[Medication]:
    try:
        medications = list(medications_col.find({"patientId": patient_id}))
        return [
            Medication(
                medicationName=med.get('medicationName'),
                dosage=med.get('dosage'),
                frequency=med.get('frequency')
            ) for med in medications
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medications: {str(e)}")
        return []

def get_available_medicines() -> List[Medicine]:
    try:
        medicines = list(medicines_col.find({}))
        return [
            Medicine(
                name=med.get('name'),
                specialization=med.get('specialization'),
                description=med.get('description', '')
            ) for med in medicines
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medicines database: {str(e)}")
        return []

# Graph nodes
def risk_assessment(state: State) -> dict:
    probs = get_risk_probabilities(state['patient_data'])
    return {'risk_probabilities': probs}

def generate_recommendations(state: State) -> dict:
    inp = state['patient_data']  # Use patient_data directly for flexibility
    probs = state['risk_probabilities']['Health Risk Probabilities']
    meds = state['current_medications']
    sent_for = state['sent_for']
    
    # Dynamically construct the input data list based on available inputs
    input_lines = []
    if inp.get('hypertension') is not None:
        input_lines.append(f"- Hypertension: {'yes' if inp['hypertension'] == 1 else 'no'}")
    if inp.get('Diet'):
        input_lines.append(f"- Diet quality: {inp['Diet']}")
    if inp.get('Stress_Level'):
        input_lines.append(f"- Stress level: {inp['Stress_Level']}")
    if inp.get('Exercise_Hours_Per_Week') is not None:
        input_lines.append(f"- Exercise hours per week: {inp['Exercise_Hours_Per_Week']}")
    if inp.get('glucose'):
        input_lines.append(f"- Glucose level: {inp['glucose']} mg/dL")
    if probs.get('Heart Disease'):
        input_lines.append(f"- CVD risk: {probs['Heart Disease']}")
    if probs.get('Diabetes'):
        input_lines.append(f"- Diabetes risk: {probs['Diabetes']}")
    
    input_data_str = "\n".join(input_lines) if input_lines else "No specific data provided."

    if sent_for == 0:
        # Dynamic prompt for patient recommendations
        prompt = (
            "You are a health advisor. Provide personalized exercise and diet recommendations in JSON format "
            "with 'patient_recommendations', 'diet_plan', 'exercise_plan', 'nutrition_targets'. "
            "Based on the following patient data:\n"
            f"{input_data_str}\n"
            "The patient is from Egypt, so ensure the advice is culturally appropriate, incorporating local foods "
            "and habits where possible. If some data is missing, provide general advice or make reasonable assumptions. "
            "In your response, include:\n"
            "1. Exercise recommendations, including a suggested target for exercise hours per week\n"
            "2. Diet recommendations, suggesting specific dietary changes and incorporating Egyptian cuisine\n"
            "3. Target values for key health metrics (e.g., glucose level, blood pressure) based on the patient's "
            "current data and health goals\n"
            "Current medications: "
            f"{', '.join([f'{m.medicationName} {m.dosage}' for m in meds]) or 'None'}\n"
            "Ensure the advice is practical, encouraging, and tailored to the patient's situation. Return only JSON."
        )
    elif sent_for == 1:  # Cardiology
        prompt = (
            "Provide JSON with 'doctor_recommendations' for cardiology. "
            "Key metrics: BP {bp}, BMI {BMI}, LDL {ldl}, diabetes risk {diab}%, CVD risk {cvd}%. "
            "Current meds: {meds}. Available cardiology meds: {available_meds}. "
            "Include: 1) Key risk factors, 2) Diagnostic tests needed, 3) Medication adjustments, "
            "4) Monitoring plan, 5) Red flags. Return only JSON."
        ).format(
            bp=inp.get('Blood_Pressure', 'unknown'),
            BMI=inp.get('BMI', 'unknown'),
            ldl=inp.get('ld_value', 'unknown'),
            diab=probs.get('Diabetes', 'unknown'),
            cvd=probs.get('Heart Disease', 'unknown'),
            meds="\n".join([f"- {m.medicationName} {m.dosage}" for m in meds]) or "None",
            available_meds="\n".join([f"- {m.name}" for m in state['available_medicines'] if 'cardiology' in m.specialization.lower()]) or "None"
        )
    elif sent_for == 2:  # Endocrinology
        prompt = (
            "Provide JSON with 'doctor_recommendations' for endocrinology. "
            "Key metrics: glucose {glucose}, HbA1c {hba1c}, BMI {BMI}, diabetes risk {diab}%. "
            "Current meds: {meds}. Available endocrine meds: {available_meds}. "
            "Include: 1) Metabolic risk factors, 2) Diagnostic tests, 3) Medication adjustments, "
            "4) Monitoring plan. Return only JSON."
        ).format(
            glucose=inp.get('glucose', 'unknown'),
            hba1c=inp.get('hemoglobin_a1c', 'unknown'),
            BMI=inp.get('BMI', 'unknown'),
            diab=probs.get('Diabetes', 'unknown'),
            meds="\n".join([f"- {m.medicationName} {m.dosage}" for m in meds]) or "None",
            available_meds="\n".join([f"- {m.name}" for m in state['available_medicines'] if 'endocrinology' in m.specialization.lower()]) or "None"
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid sent_for value')

    try:
        resp = llm.invoke(prompt)
        json_str = re.search(r'\{.*\}', resp.content, re.DOTALL).group(0)
        data = json.loads(json_str)
        
        # Process doctor recommendations if present
        if 'doctor_recommendations' in data and data['doctor_recommendations']:
            processed_recs = []
            for rec in data['doctor_recommendations']:
                if isinstance(rec, dict):
                    key = next(iter(rec))
                    processed_recs.append(f"{key}: {rec[key]}")
                else:
                    processed_recs.append(rec)
            data['doctor_recommendations'] = processed_recs
            
        return {'recommendations': Recommendations(**data)}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse recommendation response")
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

def evaluate_recommendations(state: State) -> dict:
    if state['sent_for'] != 0:
        return {'selected_patient_recommendations': []}
    
    original = state['risk_probabilities']
    selected = []
    for rec in state['recommendations'].patient_recommendations or []:
        kind = classify_recommendation(rec)
        if kind != 'Other':
            adj = adjust_metrics(state['patient_data'], kind)
            new_probs = get_risk_probabilities(adj)
            if is_effective(original, new_probs):
                selected.append(rec)
    return {'selected_patient_recommendations': selected}

def output_results(state: State) -> dict:
    probs = state['risk_probabilities']['Health Risk Probabilities']
    result = {
        'diabetes_probability': probs['Diabetes'],
        'cvd_probability': probs['Heart Disease'],
        'current_medications': [{
            'medicationName': m.medicationName,
            'dosage': m.dosage,
            'frequency': m.frequency
        } for m in state.get('current_medications', [])]
    }
    
    if state['sent_for'] == 0:
        result.update({
            'patient_recommendations': state['selected_patient_recommendations'][:3],
            'diet_plan': state['recommendations'].diet_plan,
            'exercise_plan': state['recommendations'].exercise_plan,
            'nutrition_targets': state['recommendations'].nutrition_targets
        })
    else:
        result['doctor_recommendations'] = state['recommendations'].doctor_recommendations[:6]
    
    return result

# Build and compile state graph
graph_builder = StateGraph(State)
for node in ['risk_assessment', 'generate_recommendations', 'evaluate_recommendations', 'output_results']:
    graph_builder.add_node(node, globals()[node])

graph_builder.add_edge(START, 'risk_assessment')
graph_builder.add_edge('risk_assessment', 'generate_recommendations')
graph_builder.add_edge('generate_recommendations', 'evaluate_recommendations')
graph_builder.add_edge('evaluate_recommendations', 'output_results')
graph_builder.add_edge('output_results', END)

graph = graph_builder.compile()

# FastAPI app
app = FastAPI()

@app.get("/recommendations/{patient_id}")
async def get_recommendations(patient_id: str, sent_for: Optional[int] = 0):
    try:
        oid = ObjectId(patient_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    patient = patients_col.find_one({"_id": oid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    metrics = list(metrics_col.find({"patientId": patient_id}).sort([('createdAt', -1)]).limit(1))
    if metrics:
        patient.update(metrics[0])

    # Get patient medications
    medications = get_patient_medications(patient_id)
    
    # Get available medicines from database
    available_medicines = get_available_medicines()

    patient_data = {
        "Blood_Pressure": patient.get('bloodPressure'),
        "Age": patient.get('anchorAge'),
        "Exercise_Hours_Per_Week": patient.get('exerciseHoursPerWeek'),
        "Diet":  patient.get('diet'),
        "Sleep_Hours_Per_Day": patient.get('sleepHoursPerDay'),
        "Stress_Level": patient.get('stressLevel'),
        "glucose": patient.get('glucose'),
        "BMI": patient.get('bmi'),
        "hypertension":  1 if patient.get("bloodPressure", 0) > 130 else 0,
        "is_smoking": patient.get('isSmoker'),
        "hemoglobin_a1c": patient.get('hemoglobinA1c'),
        "Diabetes_pedigree": patient.get('diabetesPedigree'),
        "CVD_Family_History": patient.get('ckdFamilyHistory'),
        "ld_value": patient.get('cholesterolLDL'),
        "admission_tsh": patient.get('admissionSOH'),
        "is_alcohol_user": patient.get('isAlcoholUser'),
        "creatine_kinase_ck": patient.get('creatineKinaseCK'),
        "gender": 'M' if patient['gender'].lower().startswith('m') else 'F',
    }

    initial_state = {
        'patient_data': patient_data,
        'sent_for': sent_for,
        'current_medications': medications,
        'available_medicines': available_medicines
    } 
    result = await graph.ainvoke(initial_state)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
