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

class ExercisePlan(BaseModel):
    type: str
    duration: int
    frequency: int
    description: str

class DietPlan(BaseModel):
    description: str
    calories: int
    meals: List[str]

class Recommendations(BaseModel):
    patient_recommendations: Optional[List[str]] = None
    diet_plan: Optional[DietPlan] = None
    exercise_plan: Optional[List[ExercisePlan]] = None
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
    if 'exercise' in t or 'activity' in t:
        return 'Physical Activity'
    if 'diet' in t or 'nutrition' in t or 'food' in t:
        return 'Diet'
    if 'smoking' in t or 'tobacco' in t:
        return 'Smoking Cessation'
    if 'sleep' in t:
        return 'Sleep Hygiene'
    if 'stress' in t or 'anxiety' in t:
        return 'Stress Management'
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
    if kind == 'Sleep Hygiene':
        d['Sleep_Hours_Per_Day'] = min(d.get('Sleep_Hours_Per_Day', 0) + 1, 9)
    if kind == 'Stress Management':
        d['Stress_Level'] = max(d.get('Stress_Level', 0) - 1, 0)
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
    pd = state['patient_data']
    probs = state['risk_probabilities']['Health Risk Probabilities']
    sent_for = state['sent_for']
    medications = state.get('current_medications', [])
    available_meds = state.get('available_medicines', [])

    # Filter medicines by specialization based on sent_for
    if sent_for == 1:  # Cardiology
        relevant_meds = [m for m in available_meds if 'cardiology' in m.specialization.lower()]
    elif sent_for == 2:  # Endocrinology
        relevant_meds = [m for m in available_meds if 'endocrinology' in m.specialization.lower()]
    else:
        relevant_meds = []

    meds_info = []
    for med in relevant_meds:
        info = f"- {med.name}"
        if med.description:
            info += f" (description: {med.description})"
        meds_info.append(info)

    if sent_for == 0:
        instruction = (
            "Provide up to five lifestyle and behavior change recommendations in 'patient_recommendations'.\n"
            "Additionally, you MUST provide a diet plan tailored for Egyptian patients in 'diet_plan', which must be a dictionary with 'description' (string describing the diet, including Egyptian foods), 'calories' (integer, daily calorie target), and 'meals' (list of strings, example meals).\n"
            "You MUST provide a comprehensive exercise plan in 'exercise_plan', which must be a LIST of dictionaries (each representing a different exercise type) with 'type' (string, e.g., 'aerobic', 'strength training'), 'duration' (integer, minutes per session), 'frequency' (integer, sessions per week), and 'description' (string, details about the exercise).\n"
            "You MUST provide nutrition targets in 'nutrition_targets', which must be a dictionary with target values for relevant metrics, e.g., 'target_BMI', 'target_glucose', etc.\n"
            "Set 'doctor_recommendations' to null.\n"
            "**Critical Instruction:** Consider the patient's current medications: {medications}. Ensure no conflicts with these medications.\n"
            "Here's an example of the expected JSON output:\n"
            "{{\n"
            "  \"patient_recommendations\": [\"Increase water intake\", \"Reduce sugar consumption\"],\n"
            "  \"diet_plan\": {{\"description\": \"A balanced diet with Egyptian staples like ful medames and koshari\", \"calories\": 2000, \"meals\": [\"Ful medames with bread\", \"Grilled chicken with rice\"]}},\n"
            " \"exercise_plan\": {\"type\": \"aerobic and strength mix\", \"duration\": 45, \"frequency\": 5, \"sessions\":[{"day":"Monday","activity":"Jogging","duration_minutes":30}, ...]},\n"
            "  \"nutrition_targets\": {{\"target_BMI\": 25.0, \"target_glucose\": 100}},\n"
            "  \"doctor_recommendations\": null\n"
            "}}"
        ).format(medications=", ".join([f"{m.medicationName} ({m.dosage})" for m in medications]))
    elif sent_for == 1:
        instruction = (
            "As a cardiology specialist, provide focused medication management and follow-up recommendations in 'doctor_recommendations'.\n"
            "Structure your response with these critical sections:\n\n"
            "1. **Medication Analysis & Recommendations**:\n"
            "   - Current medication review (considering patient's age {age}, gender {gender}, and risk scores)\n"
            "   - Potential drug interactions with current meds: {medications_list}\n"
            "   - Recommended adjustments with clear rationale based on:\n"
            "     * Blood pressure: {bp}\n"
            "     * LDL: {ldl}\n"
            "     * CVD risk: {cvd_risk}%\n"
            "     * Diabetes risk: {diabetes_risk}%\n"
            "   - New medication options from available: {available_meds}\n"
            "   - Specific dosing instructions and titration plans\n"
            "   - Monitoring requirements (e.g., renal function, liver enzymes)\n\n"
            "2. **Priority Follow-up Testing**:\n"
            "   - Immediate tests (within 1 week):\n"
            "     * Required based on current status\n"
            "     * Frequency and timing\n"
            "   - Short-term tests (1-3 months):\n"
            "     * Based on medication changes\n"
            "     * Risk factor monitoring\n"
            "   - Long-term tests (6-12 months):\n"
            "     * Annual screenings\n"
            "     * Condition progression\n\n"
            "3. **Personalized Follow-up Schedule**:\n"
            "   - Next visit timing based on:\n"
            "     * Medication changes\n"
            "     * Risk level\n"
            "     * Test results needed\n"
            "   - Red flags requiring immediate attention\n"
            "   - Monitoring frequency for specific parameters\n\n"
            "Format as a clear, numbered list with priority indicators (❗ for high priority, ⚠️ for medium).\n"
            "Example format:\n"
            "{{\n"
            "  \"doctor_recommendations\": [\n"
            "    \"1. **Medication Recommendations**:\\n❗ Discontinue X due to interaction with Y\\n⚠️ Start Atorvastatin 20mg nightly (LDL {ldl}, target <70)\\n- Monitor ALT/AST at 4 weeks\",\n"
            "    \"2. **Priority Testing**:\\n❗ Fasting lipid panel within 1 week\\n⚠️ ECG if chest pain develops\\n- Annual stress test recommended\",\n"
            "    \"3. **Follow-up Plan**:\\n❗ Recheck BP in 1 week\\n⚠️ Full reassessment in 3 months\\n- Red flags: Chest pain, SOB, palpitations\"\n"
            "  ]\n"
            "}}"
        ).format(
            age=pd.get('Age', 'N/A'),
            gender='Male' if pd.get('gender') == 'M' else 'Female',
            bp=pd.get('Blood_Pressure', 'N/A'),
            ldl=pd.get('ld_value', 'N/A'),
            cvd_risk=probs['Heart Disease'],
            diabetes_risk=probs['Diabetes'],
            medications_list="\n- ".join([f"{m.medicationName} {m.dosage}" + (f" ({m.frequency})" if m.frequency else "") for m in medications]),
            available_meds="\n".join(meds_info) if meds_info else "None"
        )
    elif sent_for == 2:
        instruction = (
            "As an endocrinology specialist, provide focused diabetes medication and monitoring recommendations in 'doctor_recommendations'.\n"
            "Structure your response with these critical sections:\n\n"
            "1. **Medication Management**:\n"
            "   - Current regimen analysis considering:\n"
            "     * Glucose: {glucose}\n"
            "     * HbA1c: {hba1c}\n"
            "     * Diabetes risk: {diabetes_risk}%\n"
            "     * BMI: {bmi}\n"
            "   - Potential interactions with: {medications_list}\n"
            "   - Recommended adjustments with clear glycemic targets\n"
            "   - New options from available: {available_meds}\n"
            "   - Renal dosing considerations\n"
            "   - Hypoglycemia prevention strategies\n\n"
            "2. **Essential Testing Schedule**:\n"
            "   - Immediate (within 1 week):\n"
            "     * Critical tests based on current status\n"
            "   - Short-term (1-3 months):\n"
            "     * Medication efficacy monitoring\n"
            "   - Long-term (6-12 months):\n"
            "     * Complication screening\n"
            "   - Frequency for each test\n\n"
            "3. **Personalized Follow-up**:\n"
            "   - Next visit timing based on:\n"
            "     * Medication changes\n"
            "     * Glycemic control\n"
            "   - Self-monitoring requirements\n"
            "   - Sick day management instructions\n"
            "   - Red flags for immediate care\n\n"
            "Format as a clear, numbered list with priority indicators (❗ for high priority, ⚠️ for medium).\n"
            "Example format:\n"
            "{{\n"
            "  \"doctor_recommendations\": [\n"
            "    \"1. **Medication Adjustments**:\\n❗ Increase Metformin to 1000mg BID (current HbA1c {hba1c})\\n⚠️ Add Empagliflozin for cardio protection\\n- Monitor for genital mycotic infections\",\n"
            "    \"2. **Testing Plan**:\\n❗ Fasting glucose weekly until stable\\n⚠️ Repeat HbA1c in 3 months\\n- Annual microalbuminuria screening\",\n"
            "    \"3. **Follow-up**:\\n❗ Visit in 2 weeks for medication tolerance\\n⚠️ Then monthly until HbA1c <7.0\\n- Red flags: Confusion, fruity breath, severe dehydration\"\n"
            "  ]\n"
            "}}"
        ).format(
            glucose=pd.get('glucose', 'N/A'),
            hba1c=pd.get('hemoglobin_a1c', 'N/A'),
            bmi=pd.get('BMI', 'N/A'),
            diabetes_risk=probs['Diabetes'],
            medications_list="\n- ".join([f"{m.medicationName} {m.dosage}" + (f" ({m.frequency})" if m.frequency else "") for m in medications]),
            available_meds="\n".join(meds_info) if meds_info else "None"
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid sent_for value')

    prompt = (
        f"Based on the following patient profile and risk probabilities, generate recommendations.\n"
        f"Patient Data: {pd}\n"
        f"Current Medications: {[f'{m.medicationName} {m.dosage}' + (f' ({m.frequency})' if m.frequency else '') for m in medications]}\n"
        f"Diabetes Risk: {probs['Diabetes']}\n"
        f"CVD Risk: {probs['Heart Disease']}\n\n"
        f"{instruction}\n"
        f"Return only the JSON object, without any additional text or explanations."
    )
    try:
        response = llm.invoke(prompt)
        json_str = re.search(r'\{.*\}', response.content, re.DOTALL).group(0)
        json_data = json.loads(json_str)
        
        # Convert any dictionary items in recommendations to strings
        if 'doctor_recommendations' in json_data and json_data['doctor_recommendations']:
            processed_recs = []
            for rec in json_data['doctor_recommendations']:
                if isinstance(rec, dict):
                    # Convert dict to string representation
                    key = next(iter(rec))
                    processed_recs.append(f"{key}: {', '.join(rec[key]) if isinstance(rec[key], list) else rec[key]}")
                else:
                    processed_recs.append(rec)
            json_data['doctor_recommendations'] = processed_recs
            
        recs = Recommendations(**json_data)
        
        if sent_for == 0 and (not recs.diet_plan or not recs.exercise_plan):
            raise ValueError("Missing required recommendation fields")
            
        return {'recommendations': recs}
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
            'exercise_plan': [{
                'type': plan.type,
                'duration': plan.duration,
                'frequency': plan.frequency,
                'description': plan.description
            } for plan in state['recommendations'].exercise_plan[:3]] if state['recommendations'].exercise_plan else None,
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
