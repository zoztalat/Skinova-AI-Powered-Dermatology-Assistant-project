"""
Skinova AI Backend Server
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Install requirements:
pip install fastapi uvicorn torch torchvision pillow pandas python-multipart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import io
import os
import sys
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = r"D:\FinalVersion\Skinova AI Dermatology App (3)\myAPP"
MODEL_FILENAME = "best_model2.pth"
MODEL_PATH     = os.path.join(MODEL_DIR, MODEL_FILENAME)
NUM_CLASSES    = 35

CLASS_NAMES = [
    'Acne And Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions',
    'Atopic Dermatitis Photos', 'Ba  Cellulitis', 'Ba Impetigo', 'Benign',
    'Bullous Disease Photos', 'Cellulitis Impetigo And Other Bacterial Infections',
    'Eczema Photos', 'Exanthems And Drug Eruptions', 'Fu Athlete Foot', 'Fu Nail Fungus',
    'Fu Ringworm', 'Hair Loss Photos Alopecia And Other Hair Diseases', 'Heathy',
    'Herpes Hpv And Other Stds Photos', 'Light Diseases And Disorders Of Pigmentation',
    'Lupus And Other Connective Tissue Diseases', 'Malignant',
    'Melanoma Skin Cancer Nevi And Moles', 'Nail Fungus And Other Nail Disease',
    'Pa Cutaneous Larva Migrans', 'Poison Ivy Photos And Other Contact Dermatitis',
    'Psoriasis Pictures Lichen Planus And Related Diseases', 'Rashes',
    'Scabies Lyme Disease And Other Infestations And Bites',
    'Seborrheic Keratoses And Other Benign Tumors', 'Systemic Disease',
    'Tinea Ringworm Candidiasis And Other Fungal Infections', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis Photos', 'Vi Chickenpox', 'Vi Shingles',
    'Warts Molluscum And Other Viral Infections'
]

DRUG_KEYWORD_MAP = {
    'Insulin': 'insulin',
    'Oral Hypoglycemics (e.g., Metformin)': 'metformin',
    'Warfarin / Anticoagulants': 'warfarin',
    'Methotrexate': 'methotrexate',
    'Amiodarone (QT-prolonging drug)': 'amiodarone',
    'Statins (e.g., Simvastatin, Atorvastatin)': 'statin',
    'Calcium-Channel Blockers (e.g., Amlodipine)': 'amlodipine',
    'Beta-Blockers (e.g., Metoprolol)': 'metoprolol',
    'Probenecid': 'probenecid',
    'Pregnancy': 'pregnancy',
}

MEDICAL_RULES_CSV = """Disease_Name,Standard_Treatment,DDI_Contraindications,Safe_Alternative
Acne And Rosacea Photos,"Doxycycline (Oral, subantimicrobial or therapeutic dosing as per guidelines)","Pregnancy (contraindicated), Concurrent high-dose calcium/iron/antacids (reduce absorption), Isotretinoin (avoid combination — risk of intracranial hypertension), Photosensitizing drugs (increased photosensitivity risk)","Topical combination (Benzoyl peroxide + Topical antibiotic or topical retinoid)"
Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions,"Topical 5-Fluorouracil (for actinic keratoses / superficial BCC); refer for suspicious malignant lesions","Concurrent warfarin (monitor INR if systemic exposure or large-area use), Avoid use in pregnancy","Diclofenac topical (or cryotherapy / dermatology referral depending on lesion)"
Atopic Dermatitis Photos,"Topical corticosteroids (first-line). Reserve short-course systemic oral corticosteroids for severe flares only","In systemic steroid use: insulin and oral hypoglycemics (monitor glucose), Use caution with live vaccines when on systemic immunosuppression","Topical calcineurin inhibitor (Tacrolimus ointment) or topical PDE4 inhibitor (as appropriate)"
Ba  Cellulitis,"Amoxicillin-Clavulanate (or local guideline-directed β-lactam or MRSA-covering agent when indicated)","Methotrexate (coadministration may increase methotrexate toxicity — use caution/monitor)","Clindamycin (oral) or appropriate cephalosporin based on local susceptibility"
Ba Impetigo,"Topical Mupirocin for localized lesions; oral cephalexin for extensive/widespread disease",N/A,"Mupirocin ointment (topical) — first choice for limited impetigo"
Benign,"Observation / reassurance; dermatology follow-up if changes",N/A,None
Bullous Disease Photos,"Depends on diagnosis: Dapsone (for dermatitis herpetiformis) or systemic corticosteroids/immunosuppression for autoimmune bullous diseases","G6PD deficiency (contraindicated for dapsone), Concurrent oxidizing agents that increase hemolysis/methemoglobinemia risk","Rituximab or other steroid-sparing immunosuppressants per specialist guidance"
Cellulitis Impetigo And Other Bacterial Infections,"Antibiotic chosen by suspected pathogen and severity (oral clindamycin or β-lactam agents commonly used)","Consider C. difficile risk with clindamycin; check patient allergy and local resistance patterns","Cephalexin or amoxicillin-clavulanate (or vancomycin for severe MRSA IV cases)"
Eczema Photos,"Topical corticosteroid (appropriate potency for site and age)","With very extensive or potent topical steroid use: monitor glucose control in diabetic patients (risk with systemic absorption)","Pimecrolimus or tacrolimus topical (steroid-sparing agents)"
Exanthems And Drug Eruptions,"Identify and stop offending drug if drug eruption; symptomatic care (topical emollients, antihistamines, topical antipruritics)",N/A,Referral to dermatologist/allergist for severe cases
Fu Athlete Foot,"Topical azole (clotrimazole or miconazole) or terbinafine topical depending on agent",N/A,"Terbinafine topical"
Fu Nail Fungus,"Oral terbinafine (first-line for dermatophyte onychomycosis) — check liver function and drug interactions","CYP2D6 and CYP interactions (e.g., metoprolol and other CYP2D6 substrates) — review med list, Existing severe liver disease (relative contraindication)","Oral itraconazole (consider interaction profile) or topical efinaconazole/ciclopirox for limited disease"
Fu Ringworm,"Oral terbinafine or itraconazole for tinea corporis/cruris/onychomycosis as per site; griseofulvin used historically (less preferred)","Griseofulvin interacts with warfarin and enzyme-inducing drugs (use caution), Terbinafine: CYP interactions and hepatotoxicity risk","Topical or oral azoles depending on site/severity (e.g., fluconazole, itraconazole)"
Hair Loss Photos Alopecia And Other Hair Diseases,"Topical Minoxidil (first-line for androgenetic alopecia in many cases)","Topical use: pregnancy/breastfeeding caution (systemic absorption minimal but oral agents contraindicate pregnancy)","Oral finasteride (men only, contraindicated in pregnancy — handle with care)"
Heathy,None,N/A,None
Herpes Hpv And Other Stds Photos,"For HSV: Acyclovir (oral). For varicella/zoster: valacyclovir/acyclovir per indication. For HPV: treatment depends on lesion type (cryotherapy, topical agents, procedural referral).","Probenecid increases acyclovir levels (dose adjustments/monitor may be needed with renal impairment)","Valacyclovir (prodrug with better oral bioavailability for HSV/shingles)"
Light Diseases And Disorders Of Pigmentation,"Topical hydroquinone (short-term) often combined with retinoid; consider sun protection and dermatology referral",N/A,"Topical retinoids, azelaic acid, or chemical peels per specialist guidance"
Lupus And Other Connective Tissue Diseases,"Hydroxychloroquine (long-term disease-modifying for cutaneous/systemic lupus) with ophthalmologic monitoring","Concomitant QT-prolonging drugs (amiodarone) — use caution and monitor ECG, Caution with drugs that affect retinal toxicity risk","Systemic immunomodulators (e.g., methotrexate, azathioprine) per rheumatology/dermatology"
Malignant,"Biopsy and urgent dermatology/oncology referral (management depends on pathology)",N/A,None
Melanoma Skin Cancer Nevi And Moles,"Biopsy/dermatology referral (no routine topical drug treatment)",N/A,None
Nail Fungus And Other Nail Disease,"Topical efinaconazole for toenail onychomycosis (localized) or systemic agents per severity",N/A,"Ciclopirox topical or systemic therapy based on organism and extent"
Pa Cutaneous Larva Migrans,"Ivermectin (oral single dose or short course) or albendazole","Use caution with anticoagulants (monitor), and review other systemic meds — interactions possible","Albendazole (oral) or topical thiabendazole where available"
Poison Ivy Photos And Other Contact Dermatitis,"Topical high-potency steroid for localized severe dermatitis; oral short-course systemic steroid for widespread severe cases","If systemic steroids used: monitor glucose in diabetic patients (insulin/oral hypoglycemics)","Oral antihistamines (e.g., diphenhydramine or non-sedating agents) for pruritus; topical emollients"
Psoriasis Pictures Lichen Planus And Related Diseases,"Topical vitamin D analog (Calcipotriene) ± topical steroid as per site/severity","Calcipotriene: caution if hypercalcemia or concomitant agents that markedly alter calcium homeostasis (rare)","Topical retinoid (Tazarotene) or refer for phototherapy/systemic agents"
Rashes,"Symptomatic: low-potency topical steroid (hydrocortisone) and emollients; identify cause",N/A,"Refer/patch testing/pruritus control depending on cause"
Scabies Lyme Disease And Other Infestations And Bites,"Permethrin 5% cream (scabies). For systemic or crusted scabies consider oral ivermectin in addition",N/A,"Oral ivermectin (for crusted scabies or where topical therapy not feasible)"
Seborrheic Keratoses And Other Benign Tumors,"Cryotherapy / curettage / dermatology referral for symptomatic lesions",N/A,"None (procedural management)"
Systemic Disease,"Referral to appropriate specialist for systemic disease management",N/A,None
Tinea Ringworm Candidiasis And Other Fungal Infections,"Topical antifungals (ketoconazole, miconazole) for skin; oral agents for extensive disease — choose by organism and site","Oral azoles/ketoconazole (systemic) have significant CYP interactions (e.g., with amiodarone, statins) — check med list","Topical miconazole or terbinafine topical for localized skin infections"
Urticaria Hives,"Second-generation antihistamine (Cetirizine, Loratadine) as first-line","Sedation risk when combined with CNS depressants for sedating antihistamines (e.g., first-generation agents)","Loratadine or fexofenadine (non-sedating alternatives)"
Vascular Tumors,"Propranolol (oral) for problematic infantile hemangiomas under specialist supervision","Concurrent calcium-channel blockers (verapamil/diltiazem) and other drugs causing bradycardia/AV block — use caution/monitor","Topical timolol for small superficial hemangiomas (specialist-guided)"
Vasculitis Photos,"Systemic corticosteroids (e.g., prednisolone) for many vasculitides as initial control; follow specialist protocol","If systemic steroids used: monitor glucose in diabetics (insulin/oral hypoglycemics)","Steroid-sparing agents (azathioprine, methotrexate) per specialist guidance"
Vi Chickenpox,"Acyclovir (oral) for adults, immunocompromised, or severe disease if started early; supportive care for healthy children","Probenecid increases acyclovir levels (dose/monitoring considerations)","Supportive care in uncomplicated pediatric cases"
Vi Shingles,"Valacyclovir or acyclovir (start within 72 hours when possible)","Probenecid (affects renal excretion of acyclovir/valacyclovir) — adjust as needed","Acyclovir (if valacyclovir unavailable) or pain control and referral for severe cases"
Warts Molluscum And Other Viral Infections,"Topical salicylic acid for common warts; procedural removal or cryotherapy for resistant lesions",N/A,"Cryotherapy or immunomodulatory topical agents per dermatologist"
"""

# ─── Severity map (diseases that require urgent referral) ─────────────────────
SEVERITY_MAP = {
    'Malignant': 'Severe',
    'Melanoma Skin Cancer Nevi And Moles': 'Severe',
    'Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions': 'Severe',
    'Bullous Disease Photos': 'Moderate',
    'Lupus And Other Connective Tissue Diseases': 'Moderate',
    'Vasculitis Photos': 'Moderate',
    'Systemic Disease': 'Moderate',
}

SEVERE_DISEASES = {'Malignant', 'Melanoma Skin Cancer Nevi And Moles',
                   'Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions'}

# ─── Model ────────────────────────────────────────────────────────────────────
class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model = models.convnext_base(weights=None)
        base_model.classifier = nn.Identity()
        self.base = base_model
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1      = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x));  x = self.dropout1(x)
        x = torch.relu(self.fc2(x));  x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return x

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

try:
    model = CustomConvNeXt(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ FATAL: Could not load model: {e}")
    sys.exit(1)

# ─── Load rules CSV ───────────────────────────────────────────────────────────
rules_df = pd.read_csv(io.StringIO(MEDICAL_RULES_CSV))

# ─── Drug interaction check ───────────────────────────────────────────────────
def check_ddi(diagnosis: str, chronic_meds: List[str]):
    row = rules_df[rules_df['Disease_Name'] == diagnosis]
    if row.empty:
        return {"conflict": False, "standard_treatment": "Referral advised", "conflicts": []}

    std_tx   = str(row['Standard_Treatment'].iloc[0])
    ddi_str  = str(row['DDI_Contraindications'].iloc[0])
    alt_tx   = str(row['Safe_Alternative'].iloc[0])

    # Build contraindication keyword set
    contra_kw = set()
    if ddi_str.upper() not in ('N/A', 'NAN', ''):
        parts = ddi_str.replace('(', ',').replace(')', ',').split(',')
        for p in parts:
            p = p.strip().lower()
            if 'insulin' in p:               contra_kw.add('insulin')
            if 'metformin' in p or 'oral hypoglycemics' in p: contra_kw.add('metformin')
            if 'warfarin' in p or 'anticoagulant' in p:       contra_kw.add('warfarin')
            if 'methotrexate' in p:          contra_kw.add('methotrexate')
            if 'amiodarone' in p:            contra_kw.add('amiodarone')
            if 'statin' in p:                contra_kw.add('statin')
            if 'calcium-channel' in p or 'amlodipine' in p:   contra_kw.add('amlodipine')
            if 'metoprolol' in p or 'beta-blocker' in p:      contra_kw.add('metoprolol')
            if 'probenecid' in p:            contra_kw.add('probenecid')
            if 'pregnancy' in p:             contra_kw.add('pregnancy')

    conflicts = []
    for med in chronic_meds:
        kw = DRUG_KEYWORD_MAP.get(med)
        if kw and kw in contra_kw:
            conflicts.append(med)

    if conflicts:
        return {
            "conflict": True,
            "conflicts": conflicts,
            "standard_treatment": std_tx,
            "safe_alternative": alt_tx,
        }
    return {
        "conflict": False,
        "conflicts": [],
        "standard_treatment": std_tx,
        "safe_alternative": alt_tx,
    }

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Skinova AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (dev mode)
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Skinova AI server is running ✅"}

@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    chronic_meds: str = Form(default="[]"),   # JSON array string
):
    # Parse meds list
    try:
        meds_list: List[str] = json.loads(chronic_meds)
    except Exception:
        meds_list = []

    # Read & predict
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output      = model(input_tensor)
        probs       = F.softmax(output, dim=1)
        conf, idx   = torch.max(probs, 1)

    predicted_class  = CLASS_NAMES[idx.item()]
    confidence       = round(conf.item() * 100, 2)

    # DDI check
    ddi = check_ddi(predicted_class, meds_list)

    # Severity
    severity   = SEVERITY_MAP.get(predicted_class, 'Mild')
    severe_case = predicted_class in SEVERE_DISEASES

    # Get description / treatment from rules
    row = rules_df[rules_df['Disease_Name'] == predicted_class]
    std_tx  = str(row['Standard_Treatment'].iloc[0]) if not row.empty else "Referral advised"
    alt_tx  = str(row['Safe_Alternative'].iloc[0])   if not row.empty else "N/A"

    return {
        "disease":          predicted_class,
        "confidence":       confidence,
        "severity":         severity,
        "severe_case":      severe_case,
        "standard_treatment": std_tx,
        "safe_alternative": alt_tx,
        "conflict_detected": ddi["conflict"],
        "conflicting_drugs": ddi.get("conflicts", []),
    }
