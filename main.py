import os
import json
import re
import logging
from datetime import date
from typing import TypedDict, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from pymongo import MongoClient
import requests
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
from typing import Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    logger.info("Environment variables loaded successfully")
except EnvironmentError as e:
    logger.error(f"Environment configuration error: {str(e)}")
    raise

# Initialize MongoDB client with connection pooling
try:
    logger.info("Initializing MongoDB connection...")
    mongo_client = MongoClient(
        MONGODB_URI,
        maxPoolSize=100,
        minPoolSize=10,
        connectTimeoutMS=5000,
        socketTimeoutMS=30000,
        serverSelectionTimeoutMS=5000
    )
    db = mongo_client.get_default_database()
    patients_col = db["patients"]
    metrics_col = db["healthmetrics"]
    medications_col = db["medications"]
    medicines_col = db["medicines"]
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Initialize LLM with enhanced configuration
try:
    logger.info("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.3,
        max_output_tokens=2048,
        top_p=0.95,
        top_k=40
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Type definitions and models
class Medication(BaseModel):
    medicationName: str
    dosage: str
    frequency: Optional[str] = None
    indication: Optional[str] = None

class Medicine(BaseModel):
    name: str
    specialization: str
    description: Optional[str] = None
    contraindications: Optional[List[str]] = None
    monitoring_requirements: Optional[List[str]] = None

class DietPlan(BaseModel):
    description: str
    calories: int
    meals: List[str]
    cultural_preferences: Optional[List[str]] = None
    restrictions: Optional[List[str]] = None

class ExercisePlan(BaseModel):
    type: str
    duration: int
    frequency: int
    description: str
    intensity: Optional[str] = None
    precautions: Optional[List[str]] = None

class NutritionTargets(BaseModel):
    target_BMI: Optional[float] = None
    target_glucose: Optional[float] = None
    target_LDL: Optional[float] = None
    target_HDL: Optional[float] = None
    target_systolic_BP: Optional[float] = None

class ClinicalRecommendationSection(BaseModel):
    section: str
    content: List[str]
    priority: Optional[Literal["high", "medium", "low"]] = None

class Recommendations(BaseModel):
    patient_recommendations: Optional[List[str]] = None
    diet_plan: Optional[DietPlan] = None
    exercise_plan: Optional[List[ExercisePlan]] = None
    nutrition_targets: Optional[NutritionTargets] = None
    doctor_recommendations: Optional[List[ClinicalRecommendationSection]] = None

class State(TypedDict):
    patient_data: dict
    sent_for: int
    risk_probabilities: dict
    recommendations: Recommendations
    selected_patient_recommendations: List[str]
    current_medications: List[Medication]
    available_medicines: List[Medicine]

# Helper functions with enhanced error handling and logging
def classify_bp(bp: str) -> str:
    """Classify blood pressure according to ACC/AHA guidelines"""
    if not bp or bp == 'N/A':
        return 'Unknown'
    try:
        systolic, diastolic = map(int, bp.split('/'))
        if systolic >= 180 or diastolic >= 120:
            return 'Hypertensive Crisis'
        elif systolic >= 140 or diastolic >= 90:
            return 'Stage 2 Hypertension'
        elif systolic >= 130 or diastolic >= 80:
            return 'Stage 1 Hypertension'
        elif systolic >= 120:
            return 'Elevated'
        else:
            return 'Normal'
    except Exception:
        return 'Unknown'

def classify_bmi(bmi: float) -> str:
    """Classify BMI according to WHO standards"""
    if not bmi or bmi == 'N/A':
        return 'Unknown'
    try:
        bmi = float(bmi)
        if bmi >= 40:
            return 'Class 3 Obesity'
        elif bmi >= 35:
            return 'Class 2 Obesity'
        elif bmi >= 30:
            return 'Class 1 Obesity'
        elif bmi >= 25:
            return 'Overweight'
        elif bmi >= 18.5:
            return 'Normal'
        else:
            return 'Underweight'
    except Exception:
        return 'Unknown'

def classify_glucose(glucose: float) -> str:
    """Classify glucose levels"""
    if not glucose or glucose == 'N/A':
        return 'Unknown'
    try:
        glucose = float(glucose)
        if glucose >= 200:
            return 'Diabetic Range'
        elif glucose >= 140:
            return 'Prediabetic Range'
        elif glucose >= 70:
            return 'Normal'
        else:
            return 'Hypoglycemic'
    except Exception:
        return 'Unknown'

def classify_ascvd_risk(risk: float) -> str:
    """Classify ASCVD risk according to ACC/AHA guidelines"""
    if risk >= 20:
        return 'High'
    elif risk >= 7.5:
        return 'Borderline'
    elif risk >= 5:
        return 'Low'
    else:
        return 'Very Low'

def classify_diabetes_risk(risk: float) -> str:
    """Classify diabetes risk"""
    if risk >= 50:
        return 'Very High'
    elif risk >= 25:
        return 'High'
    elif risk >= 10:
        return 'Moderate'
    else:
        return 'Low'

def format_comorbidities(pd: dict, probs: dict) -> str:
    """Format comorbidities list based on patient data and probabilities"""
    comorbidities = []
    if float(probs['Diabetes'].strip('%')) > 25:
        comorbidities.append("Prediabetes")
    if pd.get('hypertension'):
        comorbidities.append("Hypertension")
    if pd.get('CVD_Family_History'):
        comorbidities.append("Family History of CVD")
    if pd.get('is_smoking'):
        comorbidities.append("Smoking")
    return ", ".join(comorbidities) if comorbidities else "None noted"

def format_exercise(hours: float) -> str:
    """Format exercise description"""
    if hours >= 5:
        return "Active"
    elif hours >= 2.5:
        return "Moderately Active"
    elif hours > 0:
        return "Sedentary"
    else:
        return "No Regular Exercise"

def assess_diet_quality(diet: str) -> str:
    """Assess diet quality"""
    if not diet:
        return "Unknown"
    diet = diet.lower()
    if 'mediterranean' in diet:
        return "High Quality"
    elif 'balanced' in diet:
        return "Moderate Quality"
    elif 'fast food' in diet or 'processed' in diet:
        return "Low Quality"
    else:
        return "Unknown"

def check_formulary(medications: List[Medication]) -> str:
    """Check if medications are on standard formulary"""
    # In a real implementation, this would check against a formulary database
    return "Standard formulary" if medications else "No current medications"

def assess_health_literacy(pd: dict) -> str:
    """Assess health literacy level"""
    # Placeholder for actual assessment logic
    return "Intermediate"  # Basic/Intermediate/Advanced

def check_transportation(pd: dict) -> str:
    """Check transportation access"""
    # Placeholder for actual assessment logic
    return "Available"

def parse_probability(prob_str: str) -> float:
    """Convert percentage string to float (0-1)"""
    try:
        return float(prob_str.strip('%')) / 100
    except (ValueError, AttributeError):
        logger.warning(f"Invalid probability string: {prob_str}")
        return 0.0

def get_risk_probabilities(patient_data: dict) -> dict:
    """Get risk probabilities from Bayesian Network API"""
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
        logger.info(f"Requesting risk probabilities from BN API for {gender} patient")
        response = requests.post(
            api_url, 
            json=payload,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"BN API request failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"BN service error: {str(e)}")

def classify_recommendation(text: str) -> str:
    """Classify recommendation type for evaluation"""
    t = text.lower()
    if 'exercise' in t or 'activity' in t:
        return 'Physical Activity'
    if 'diet' in t or 'nutrition' in t or 'food' in t:
        return 'Diet'
    if 'smoking' in t or 'tobacco' in t:
        return 'Smoking Cessation'
    if 'alcohol' in t or 'drinking' in t:
        return 'Alcohol Reduction'
    return 'Other'

def adjust_metrics(data: dict, kind: str) -> dict:
    """Simulate metric adjustments based on recommendation type"""
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
    if kind == 'Alcohol Reduction':
        d['is_alcohol_user'] = False
    return d

def is_effective(orig: dict, new: dict) -> bool:
    """Determine if recommendation would be effective"""
    try:
        o = orig['Health Risk Probabilities']
        n = new['Health Risk Probabilities']
        o_d = parse_probability(o['Diabetes'])
        o_c = parse_probability(o['Heart Disease'])
        n_d = parse_probability(n['Diabetes'])
        n_c = parse_probability(n['Heart Disease'])
        
        # Consider recommendation effective if either risk decreases significantly
        # without significant increase in the other risk
        return ((n_d < o_d - 0.05 and n_c <= o_c + 0.01) or
                (n_c < o_c - 0.05 and n_d <= o_d + 0.01))
    except KeyError as e:
        logger.error(f"Missing risk probability in effectiveness check: {str(e)}")
        return False

def get_patient_medications(patient_id: str) -> List[Medication]:
    """Retrieve patient's current medications with error handling"""
    try:
        logger.info(f"Fetching medications for patient {patient_id}")
        medications = list(medications_col.find({"patientId": patient_id}))
        return [
            Medication(
                medicationName=med.get('medicationName'),
                dosage=med.get('dosage'),
                frequency=med.get('frequency'),
                indication=med.get('indication')
            ) for med in medications
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medications: {str(e)}")
        return []

def get_available_medicines() -> List[Medicine]:
    """Retrieve available medicines from database"""
    try:
        logger.info("Fetching available medicines from database")
        medicines = list(medicines_col.find({}))
        return [
            Medicine(
                name=med.get('name'),
                specialization=med.get('specialization'),
                description=med.get('description', ''),
                contraindications=med.get('contraindications', []),
                monitoring_requirements=med.get('monitoring_requirements', [])
            ) for med in medicines
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medicines database: {str(e)}")
        return []

# Graph nodes with enhanced functionality
def risk_assessment(state: State) -> dict:
    """Assess patient risk using Bayesian Network"""
    logger.info("Starting risk assessment")
    probs = get_risk_probabilities(state['patient_data'])
    return {'risk_probabilities': probs}

def generate_recommendations(state: State) -> dict:
    """Generate personalized recommendations based on patient state"""
    logger.info(f"Generating recommendations for sent_for={state['sent_for']}")
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
            info += f" (Indications: {med.description})"
        if med.contraindications:
            info += f" [Contraindications: {', '.join(med.contraindications)}]"
        meds_info.append(info)

    if sent_for == 0:
        instruction = (
            "Provide up to five lifestyle and behavior change recommendations in 'patient_recommendations'.\n"
            "Additionally, you MUST provide a diet plan tailored for Egyptian patients in 'diet_plan', which must be a dictionary with:\n"
            "- 'description' (string describing the diet, including Egyptian foods)\n"
            "- 'calories' (integer, daily calorie target)\n"
            "- 'meals' (list of strings, example meals)\n"
            "- 'cultural_preferences' (list of Egyptian dietary preferences)\n"
            "- 'restrictions' (list of any dietary restrictions)\n\n"
            
            "You MUST provide a comprehensive exercise plan in 'exercise_plan', which must be a LIST of dictionaries with:\n"
            "- 'type' (e.g., 'aerobic', 'strength training')\n"
            "- 'duration' (minutes per session)\n"
            "- 'frequency' (sessions per week)\n"
            "- 'description' (details about the exercise)\n"
            "- 'intensity' (low/medium/high)\n"
            "- 'precautions' (list of precautions if any)\n\n"
            
            "You MUST provide nutrition targets in 'nutrition_targets', which must include:\n"
            "- 'target_BMI'\n"
            "- 'target_glucose'\n"
            "- 'target_LDL'\n"
            "- 'target_HDL'\n"
            "- 'target_systolic_BP'\n\n"
            
            "Set 'doctor_recommendations' to null.\n"
            "Consider the patient's current medications: {medications}. Ensure no conflicts.\n"
            "Example output format:\n"
            "{{\n"
            "  \"patient_recommendations\": [\"Increase water intake\", \"Reduce sugar\"],\n"
            "  \"diet_plan\": {{\"description\": \"Egyptian diet with ful medames...\", \"calories\": 2000, \"meals\": [\"Ful medames\", \"Grilled chicken\"], \"cultural_preferences\": [\"Egyptian cuisine\"], \"restrictions\": [\"Low salt\"]}},\n"
            "  \"exercise_plan\": [{{\"type\": \"aerobic\", \"duration\": 30, \"frequency\": 5, \"description\": \"Brisk walking\", \"intensity\": \"medium\", \"precautions\": [\"Avoid in extreme heat\"]}}],\n"
            "  \"nutrition_targets\": {{\"target_BMI\": 25.0, \"target_glucose\": 100, \"target_LDL\": 100, \"target_HDL\": 50, \"target_systolic_BP\": 120}},\n"
            "  \"doctor_recommendations\": null\n"
            "}}"
        ).format(medications=", ".join([f"{m.medicationName} ({m.dosage})" for m in medications]))
    elif sent_for == 1:
        instruction = (
            "Generate a clinical decision support output for cardiology management with these requirements:\n\n"
            
            "STRUCTURE:\n"
            "1. Risk Stratification:\n"
            "   - Quantify risk using ESC/ACC guidelines\n"
            "   - Highlight modifiable vs non-modifiable factors\n"
            "   - Include 10-year/30-year ASCVD risk when calculable\n\n"
            
            "2. Diagnostic Workflow:\n"
            "   - Tiered recommendations (STAT/Routine/Consider if...)\n"
            "   - Justify each test with clinical rationale\n"
            "   - Include cost-effectiveness considerations\n"
            "   - Specify optimal timing (fasting/non-fasting, AM/PM)\n\n"
            
            "3. Pharmacotherapy Matrix:\n"
            "   - Current Regimen Analysis:\n"
            "     * Adherence barriers\n"
            "     * Therapeutic duplication\n"
            "     * Potential interactions (DDI/DDI-Check)\n"
            "   - Evidence-Based Additions:\n"
            "     * First-line → Alternatives\n"
            "     * Dosing algorithms (renal/hepatic adjustment)\n"
            "     * Prior authorization requirements\n"
            "     * Cost-saving alternatives\n"
            "   - Deprescribing Opportunities:\n"
            "     * Beers Criteria medications\n"
            "     * Therapeutic redundancy\n\n"
            
            "4. Monitoring Protocol:\n"
            "   - Short-term (1-3 month) safety labs\n"
            "   - Long-term efficacy markers\n"
            "   - Point-of-care testing recommendations\n"
            "   - Remote monitoring options\n\n"
            
            "5. Safety Considerations:\n"
            "   - Black box warnings\n"
            "   - High-risk combinations\n"
            "   - Fall risk assessment\n\n"
            
            "6. Value-Based Care:\n"
            "   - Cost-effective alternatives\n"
            "   - Therapeutic interchange options\n"
            "   - Prior authorization strategies\n\n"
            
            "CLINICAL INPUTS:\n"
            "- Vitals: BP {bp} ({bp_class}), BMI {bmi} ({bmi_class})\n"
            "- Metabolic: Glucose {glucose} ({glucose_status}), HbA1c {a1c}\n"
            "- Lipids: LDL {ldl}, HDL {hdl}, Triglycerides {trig}\n"
            "- Risk Scores:\n"
            "  * ASCVD: {cvd_risk}% ({cvd_risk_class})\n"
            "  * Diabetes: {diabetes_risk}% ({diabetes_risk_class})\n"
            "  * Heart Failure: {hf_risk}%\n"
            "- Comorbidities: {comorbidities}\n"
            "- Lifestyle: {exercise}, {diet_quality} diet, {smoking_status}\n"
            "- Current Medications ({medications_count}):\n{medications_list}\n"
            "- Medication Access: {formulary_status}\n"
            "- Social Determinants: {health_literacy}, {transportation_access}\n\n"
            
            "OUTPUT FORMAT (strict JSON):\n"
            "{{\n"
            "  \"doctor_recommendations\": [\n"
            "    {{\"section\": \"Risk Assessment\", \"content\": [\"...\", \"...\"], \"priority\": \"high\"}},\n"
            "    {{\"section\": \"Diagnostics\", \"content\": [\"...\", \"...\"], \"priority\": \"medium\"}},\n"
            "    {{\"section\": \"Medication Management\", \"content\": [\"...\", \"...\"], \"priority\": \"high\"}}\n"
            "  ],\n"
            "  \"patient_recommendations\": null,\n"
            "  \"diet_plan\": null,\n"
            "  \"exercise_plan\": null,\n"
            "  \"nutrition_targets\": null\n"
            "}}\n\n"
            
            "CLINICAL RULES TO APPLY:\n"
            "1. ESC 2021 CVD Prevention Guidelines\n"
            "2. ADA 2023 Diabetes Standards\n"
            "3. AHA/ACC Heart Failure Guidelines\n"
            "4. Beers Criteria 2023\n"
            "5. Local formulary restrictions\n\n"
            
            "Avoid:\n"
            "- Generic lifestyle advice\n"
            "- Recommendations for tests/meds already in regimen\n"
            "- Contradictions with patient's belief system\n"
            "- Non-formulary medications without alternatives"
        ).format(
            bp=pd.get('Blood_Pressure', 'N/A'),
            bp_class=classify_bp(pd.get('Blood_Pressure')),
            bmi=pd.get('BMI', 'N/A'),
            bmi_class=classify_bmi(pd.get('BMI')),
            glucose=pd.get('glucose', 'N/A'),
            glucose_status=classify_glucose(pd.get('glucose')),
            a1c=pd.get('hemoglobin_a1c', 'N/A'),
            ldl=pd.get('ld_value', 'N/A'),
            hdl=pd.get('cholesterolHDL', 'N/A'),
            trig=pd.get('triglycerides', 'N/A'),
            cvd_risk=probs['Heart Disease'],
            cvd_risk_class=classify_ascvd_risk(float(probs['Heart Disease'].strip('%'))),
            diabetes_risk=probs['Diabetes'],
            diabetes_risk_class=classify_diabetes_risk(float(probs['Diabetes'].strip('%'))),
            hf_risk=probs.get('Heart Failure', '0%'),
            comorbidities=format_comorbidities(pd, probs),
            exercise=format_exercise(pd.get('Exercise_Hours_Per_Week', 0)),
            diet_quality=assess_diet_quality(pd.get('Diet')),
            smoking_status="Current smoker" if pd.get('is_smoking') else "Never smoker",
            medications_list="\n".join([f"- {m.medicationName} {m.dosage}{' ' + m.frequency if m.frequency else ''} ({m.indication or 'Unspecified'})" for m in medications]),
            medications_count=len(medications),
            formulary_status=check_formulary(medications),
            health_literacy=assess_health_literacy(pd),
            transportation_access=check_transportation(pd),
            available_meds="\n".join(meds_info) if meds_info else "No specific cardiology medications in database"
        )
    elif sent_for == 2:
        instruction = (
            "Provide up to three medical action recommendations for an endocrinologist in 'doctor_recommendations'. "
            "Structure as a list of ClinicalRecommendationSection objects with:\n"
            "1. Key metabolic risk factors\n"
            "2. Recommended diagnostic tests with targets\n"
            "3. Medication considerations: \n"
            "   - First list current diabetes/endocrine medications: {medications_list}\n"
            "   - Then suggest potential medication adjustments or additions\n"
            "   - Do NOT recommend medications already being taken\n"
            "   - Check for contraindications with current regimen\n"
            "4. Monitoring plan\n"
            "5. Evidence basis\n\n"
            "Example output format:\n"
            "{{\n"
            "  \"doctor_recommendations\": [\n"
            "    {{\"section\": \"Risk Factors\", \"content\": [\"Elevated glucose {glucose}\", \"HbA1c {hba1c}\"], \"priority\": \"high\"}},\n"
            "    {{\"section\": \"Diagnostics\", \"content\": [\"Fasting glucose (target < 100)\", \"HbA1c (target < 5.7%)\"], \"priority\": \"medium\"}}\n"
            "  ],\n"
            "  \"patient_recommendations\": null,\n"
            "  \"diet_plan\": null,\n"
            "  \"exercise_plan\": null,\n"
            "  \"nutrition_targets\": null\n"
            "}}"
        ).format(
            medications_list="\n".join([f"- {m.medicationName} {m.dosage}{' ' + m.frequency if m.frequency else ''}" for m in medications]),
            glucose=pd.get('glucose', 'N/A'),
            hba1c=pd.get('hemoglobin_a1c', 'N/A'),
            bmi=pd.get('BMI', 'N/A'),
            available_meds="\n".join(meds_info) if meds_info else "No specific endocrinology medications in database"
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid sent_for value')

    prompt = (
        f"Based on the following patient profile and risk probabilities, generate recommendations.\n"
        f"Patient Data: {json.dumps(pd, indent=2)}\n"
        f"Current Medications: {[f'{m.medicationName} {m.dosage}' + (f' ({m.frequency})' if m.frequency else '') for m in medications]}\n"
        f"Diabetes Risk: {probs['Diabetes']}\n"
        f"CVD Risk: {probs['Heart Disease']}\n\n"
        f"{instruction}\n"
        f"Return only the JSON object, without any additional text or explanations."
    )
    
    try:
        logger.info("Invoking LLM for recommendations...")
        response = llm.invoke(prompt)
        
        # Enhanced JSON parsing with better error handling
        try:
            json_str = re.search(r'\{.*\}', response.content, re.DOTALL).group(0)
            json_data = json.loads(json_str)
            
            # Convert any dictionary items in recommendations to proper format
            if 'doctor_recommendations' in json_data and json_data['doctor_recommendations']:
                processed_recs = []
                for rec in json_data['doctor_recommendations']:
                    if isinstance(rec, str):
                        # Convert string to section format
                        processed_recs.append({
                            "section": rec.split(":")[0] if ":" in rec else "Recommendation",
                            "content": [rec],
                            "priority": "medium"
                        })
                    elif isinstance(rec, dict):
                        # Ensure proper section format
                        processed_recs.append({
                            "section": rec.get("section", "Recommendation"),
                            "content": rec.get("content", [str(rec)]),
                            "priority": rec.get("priority", "medium")
                        })
                json_data['doctor_recommendations'] = processed_recs
            
            # Validate and convert using Pydantic model
            recs = Recommendations(**json_data)
            
            if sent_for == 0 and (not recs.diet_plan or not recs.exercise_plan):
                raise ValueError("Missing required recommendation fields")
                
            return {'recommendations': recs}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.error(f"Response content: {response.content}")
            raise HTTPException(status_code=500, detail="Failed to parse recommendation response")
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

def evaluate_recommendations(state: State) -> dict:
    """Evaluate and select most effective patient recommendations"""
    if state['sent_for'] != 0:
        return {'selected_patient_recommendations': []}
    
    logger.info("Evaluating patient recommendations for effectiveness")
    original = state['risk_probabilities']
    selected = []
    
    for rec in state['recommendations'].patient_recommendations or []:
        kind = classify_recommendation(rec)
        if kind != 'Other':
            adj = adjust_metrics(state['patient_data'], kind)
            try:
                new_probs = get_risk_probabilities(adj)
                if is_effective(original, new_probs):
                    selected.append(rec)
            except Exception as e:
                logger.warning(f"Failed to evaluate recommendation '{rec}': {str(e)}")
    
    # Limit to top 3 most effective recommendations
    return {'selected_patient_recommendations': selected[:3]}

def output_results(state: State) -> dict:
    """Format final output results"""
    logger.info("Formatting final output results")
    probs = state['risk_probabilities']['Health Risk Probabilities']
    result = {
        'diabetes_probability': probs['Diabetes'],
        'cvd_probability': probs['Heart Disease'],
        'current_medications': [{
            'medicationName': m.medicationName,
            'dosage': m.dosage,
            'frequency': m.frequency,
            'indication': m.indication
        } for m in state.get('current_medications', [])]
    }
    
    if state['sent_for'] == 0:
        result.update({
            'patient_recommendations': state['selected_patient_recommendations'],
            'diet_plan': state['recommendations'].diet_plan.dict() if state['recommendations'].diet_plan else None,
            'exercise_plan': [ex.dict() for ex in state['recommendations'].exercise_plan] if state['recommendations'].exercise_plan else None,
            'nutrition_targets': state['recommendations'].nutrition_targets.dict() if state['recommendations'].nutrition_targets else None
        })
    else:
        result['doctor_recommendations'] = [
            rec.dict() for rec in state['recommendations'].doctor_recommendations
        ] if state['recommendations'].doctor_recommendations else None
    
    return result

# Build and compile state graph with enhanced configuration
graph_builder = StateGraph(State)
for node in ['risk_assessment', 'generate_recommendations', 'evaluate_recommendations', 'output_results']:
    graph_builder.add_node(node, globals()[node])

graph_builder.add_edge(START, 'risk_assessment')
graph_builder.add_edge('risk_assessment', 'generate_recommendations')
graph_builder.add_edge('generate_recommendations', 'evaluate_recommendations')
graph_builder.add_edge('evaluate_recommendations', 'output_results')
graph_builder.add_edge('output_results', END)

graph = graph_builder.compile()

# FastAPI app with enhanced configuration
app = FastAPI(
    title="Clinical Decision Support System",
    description="API for generating personalized medical recommendations",
    version="1.1.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url=None
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": ENVIRONMENT}

@app.get("/recommendations/{patient_id}", response_model=Dict[str, Any])
async def get_recommendations(patient_id: str, sent_for: int = 0):
    """Generate personalized recommendations for a patient"""
    try:
        logger.info(f"Processing recommendations request for patient {patient_id}")
        oid = ObjectId(patient_id)
    except Exception:
        logger.error(f"Invalid patient ID format: {patient_id}")
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    # Retrieve patient data with enhanced error handling
    try:
        patient = patients_col.find_one({"_id": oid})
        if not patient:
            logger.error(f"Patient not found: {patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found")

        metrics = list(metrics_col.find({"patientId": patient_id}).sort([('createdAt', -1)]).limit(1))
        if metrics:
            patient.update(metrics[0])
    except Exception as e:
        logger.error(f"Failed to retrieve patient data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patient data")

    # Get patient medications and available medicines
    medications = get_patient_medications(patient_id)
    available_medicines = get_available_medicines()

    # Prepare patient data with enhanced validation
    patient_data = {
        "Blood_Pressure": patient.get('bloodPressure'),
        "Age": patient.get('anchorAge'),
        "Exercise_Hours_Per_Week": patient.get('exerciseHoursPerWeek', 0),
        "Diet": patient.get('diet', 'Unknown'),
        "Sleep_Hours_Per_Day": patient.get('sleepHoursPerDay', 0),
        "Stress_Level": patient.get('stressLevel', 0),
        "glucose": patient.get('glucose'),
        "BMI": patient.get('bmi'),
        "hypertension": 1 if patient.get("bloodPressure", "0/0").split('/')[0] > '130' else 0,
        "is_smoking": patient.get('isSmoker', False),
        "hemoglobin_a1c": patient.get('hemoglobinA1c'),
        "diabetesPedigree": patient.get('diabetesPedigree', 0),
        "CVD_Family_History": patient.get('ckdFamilyHistory', False),
        "ld_value": patient.get('cholesterolLDL'),
        "cholesterolHDL": patient.get('cholesterolHDL'),
        "triglycerides": patient.get('triglycerides'),
        "is_alcohol_user": patient.get('isAlcoholUser', False),
        "gender": 'M' if str(patient.get('gender', '')).lower().startswith('m') else 'F',
    }

    initial_state = {
        'patient_data': patient_data,
        'sent_for': sent_for,
        'current_medications': medications,
        'available_medicines': available_medicines
    }
    
    try:
        logger.info(f"Executing recommendation workflow for sent_for={sent_for}")
        result = await graph.ainvoke(initial_state)
        logger.info(f"Successfully generated recommendations for patient {patient_id}")
        return result
    except Exception as e:
        logger.error(f"Recommendation workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level="info",
        timeout_keep_alive=60,
        workers=4 if ENVIRONMENT == "production" else 1
    )
