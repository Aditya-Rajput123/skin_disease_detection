from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import base64
from PIL import Image
from io import BytesIO

# --- MODEL LIBRARIES (Ensure these are installed: pip install tensorflow keras numpy Pillow) ---
from tensorflow.keras.models import load_model 
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions
# ---------------------------------------------------------------------------------------------


app = Flask(__name__)

# --- MODEL LOADING AND CONFIGURATION (Loaded once at startup) ---
MODEL_PATH = 'model/trained_model_98.h5'
try:
    # 1. Load the main skin disease model
    model = load_model(MODEL_PATH) 
    print("Skin Disease Model loaded successfully!")
    
    # 1.5 Load MobileNetV2 for Gatekeeping
    gatekeeper_model = MobileNetV2(weights='imagenet')
    print("Gatekeeper Model (MobileNetV2) loaded successfully!")
    
    # 2. Define all your classes here (the list you provided)
    CLASS_NAMES = [
        'Acne and Rosacea Photos', 'Actinic Keratosis...', 'Atopic Dermatitis Photos',
        'Cellulitis Impetigo...', 'Eczema Photos', 'Exanthems and Drug Eruptions',
        'Herpes HPV...', 'Light Diseases...', 'Lupus...', 
        'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos...', 
        'Psoriasis pictures Lichen Planus...', 'Seborrheic Keratoses...', 
        'Systemic Disease', 'Tinea Ringworm...', 'Urticaria Hives', 
        'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum...'
    ]
    
    DISEASE_INFO = {
        'Acne and Rosacea Photos': {
            'description': 'Acne and rosacea are common skin conditions causing redness, bumps, and blemishes. While acne involves clogged pores, rosacea primarily causes persistent redness and visible blood vessels.',
            'symptoms': 'Red bumps, pimples, blackheads, facial redness, visible blood vessels, and skin sensitivity.',
            'home_remedy': 'Apply tea tree oil, aloe vera, or a honey mask. Avoid picking at the skin.',
            'solution': 'Use over-the-counter benzoyl peroxide or salicylic acid. Prescription retinoids or antibiotics may be needed.',
            'precautions': 'Wash your face gently twice a day. Avoid harsh scrubs and heavy, pore-clogging makeup.',
            'origin_causes': 'Caused by clogged pores due to excess oil production, dead skin cells, and bacteria. Hormones and stress can trigger it.',
            'contact_dermat': 'If it is severe, cystic, leaves scars, or does not respond to over-the-counter treatments, consult a dermatologist immediately.',
            'diet_lifestyle': 'Reduce intake of high glycemic index foods and dairy. Manage stress and ensure adequate sleep.',
            'contagious': 'No, neither acne nor rosacea is contagious.',
            'duration': 'Can be persistent. Breakouts may last for a few weeks, while rosacea is a chronic condition that requires long-term management.',
            'doctor_visit_timeline': 'Within 2-4 weeks if OTC treatments fail, or immediately if cystic acne occurs.'
        },
        'Actinic Keratosis...': {
            'description': 'A rough, scaly patch on the skin that develops from years of sun exposure. It is considered precancerous.',
            'symptoms': 'Rough, dry, or scaly patches of skin, usually less than 1 inch in diameter. Flat to slightly raised bumps.',
            'home_remedy': 'There are no effective home remedies for this precancerous condition. Apple cider vinegar is sometimes used but not medically recommended.',
            'solution': 'Medical treatments include cryotherapy (freezing), topical chemotherapy creams (like 5-fluorouracil), or photodynamic therapy.',
            'precautions': 'Strictly use broad-spectrum sunscreen, wear protective clothing, and avoid tanning beds.',
            'origin_causes': 'Primarily caused by long-term, chronic exposure to ultraviolet (UV) rays from the sun.',
            'contact_dermat': 'Yes, contact a dermatologist. It is considered precancerous and should be monitored and treated to prevent progression to skin cancer.',
            'diet_lifestyle': 'A diet rich in antioxidants (vitamins A, C, E) may help protect skin cells from further damage. Avoid excessive sun exposure.',
            'contagious': 'No, it is not contagious.',
            'duration': 'Lesions can persist for years if untreated and may progress to skin cancer.',
            'doctor_visit_timeline': 'Within 1-2 weeks of noticing persistent scaly patches.'
        },
        'Atopic Dermatitis Photos': {
            'description': 'Also known as eczema, it is a condition that makes your skin red and itchy. It is common in children but can occur at any age.',
            'symptoms': 'Dry skin, itching (which may be severe), red to brownish-gray patches, and small, raised bumps.',
            'home_remedy': 'Take colloidal oatmeal baths, apply coconut oil or thick moisturizers immediately after bathing.',
            'solution': 'Use topical corticosteroids, calcineurin inhibitors, or antihistamines to reduce itching.',
            'precautions': 'Use mild, fragrance-free soaps. Avoid known triggers like wool, harsh chemicals, and extreme temperatures.',
            'origin_causes': 'A combination of genetic variations that affect the skin\'s barrier function and environmental immune system triggers.',
            'contact_dermat': 'If the itching affects your sleep, home treatments aren\'t working, or you suspect a skin infection, see a doctor.',
            'diet_lifestyle': 'Identify and avoid food allergens (like dairy or nuts) if they trigger flare-ups. Wear soft, breathable cotton clothing.',
            'contagious': 'No, atopic dermatitis is not contagious.',
            'duration': 'Chronic condition that can flare up periodically throughout life, though many children outgrow it.',
            'doctor_visit_timeline': 'Within 1 week if itching is severe or sleep-disturbing.'
        },
        'Cellulitis Impetigo...': {
            'description': 'Serious bacterial skin infections. Cellulitis affects deeper layers of skin, while impetigo is a highly contagious surface infection.',
            'symptoms': 'Redness, swelling, tenderness, warmth, and sometimes blisters or crusting (honey-colored in impetigo).',
            'home_remedy': 'Keep the area clean and elevated. Apply warm compresses to reduce pain and swelling. Do not attempt to treat solely at home.',
            'solution': 'Prescription oral or topical antibiotics are required to clear the bacterial infection.',
            'precautions': 'Wash cuts and scrapes carefully. Keep wounds bandaged until healed and practice good hand hygiene.',
            'origin_causes': 'Caused by bacteria (usually Streptococcus or Staphylococcus) entering through a break, cut, or insect bite in the skin.',
            'contact_dermat': 'Highly serious. Seek immediate medical attention if you have a rapidly spreading red, swollen, tender rash, especially with a fever.',
            'diet_lifestyle': 'Rest and elevate the affected area to reduce swelling. Maintain good overall hygiene.',
            'contagious': 'Impetigo is highly contagious. Cellulitis is generally not contagious from person to person.',
            'duration': 'With proper antibiotics, it usually clears up in 7 to 14 days.',
            'doctor_visit_timeline': 'Immediately (same day) for cellulitis symptoms.'
        },
        'Eczema Photos': {
            'description': 'A broad term for conditions that cause skin inflammation. It results in itchy, red, and dry skin.',
            'symptoms': 'Intense itching, redness, dry/scaly skin, and sometimes oozing or crusting.',
            'home_remedy': 'Use regular application of petroleum jelly, wet wrap therapy, and cool compresses.',
            'solution': 'Medical treatments include prescription steroid creams, barrier repair moisturizers, and sometimes immunosuppressants.',
            'precautions': 'Moisturize multiple times a day. Wear soft, breathable fabrics like cotton.',
            'origin_causes': 'Often genetic, linked to an overactive immune system responding to internal or external triggers.',
            'contact_dermat': 'Consult a dermatologist if the condition covers a large area of the body or is causing significant discomfort.',
            'diet_lifestyle': 'Manage stress through relaxation techniques. Avoid harsh detergents and highly fragranced products.',
            'contagious': 'No, eczema is not contagious.',
            'duration': 'Often a lifelong, chronic condition with intermittent flare-ups.',
            'doctor_visit_timeline': 'Within 1-2 weeks if a flare-up does not respond to OTC moisturizers.'
        },
        'Exanthems and Drug Eruptions': {
            'description': 'Skin rashes caused by an internal trigger, often a medication or a viral infection.',
            'symptoms': 'Widespread rash, small red bumps, or hives. May be accompanied by itching or fever.',
            'home_remedy': 'Take cool showers and apply calamine lotion or aloe vera to soothe the skin.',
            'solution': 'Stop the suspected medication (under doctor guidance). Use oral antihistamines and topical corticosteroids.',
            'precautions': 'Keep a record of any medications that cause a reaction and inform all your healthcare providers.',
            'origin_causes': 'An allergic reaction to a prescribed or over-the-counter medication, or sometimes due to a viral infection.',
            'contact_dermat': 'Seek emergency care if the rash is accompanied by difficulty breathing, facial swelling, or blistering.',
            'diet_lifestyle': 'Drink plenty of water to help flush the system. Avoid potential trigger medications in the future.',
            'contagious': 'Drug eruptions are not contagious. Viral exanthems can be contagious depending on the underlying virus.',
            'duration': 'Usually resolves within 1 to 2 weeks after stopping the offending medication.',
            'doctor_visit_timeline': 'Immediately if accompanied by fever or breathing difficulty, otherwise within 24-48 hours.'
        },
        'Herpes HPV...': {
            'description': 'Viral infections causing sores or warts. Herpes typically causes blisters, while HPV leads to various types of warts.',
            'symptoms': 'Painful blisters (herpes) or small, fleshy bumps/warts (HPV).',
            'home_remedy': 'Apply a cold compress to sores. For warts, over-the-counter salicylic acid patches can be used.',
            'solution': 'Antiviral medications (oral or topical) for herpes. Cryotherapy or minor surgery for HPV wart removal.',
            'precautions': 'Avoid direct skin-to-skin contact with infected areas. Do not share towels or personal items.',
            'origin_causes': 'Caused by viral infections: Herpes Simplex Virus (HSV) for sores, and Human Papillomavirus (HPV) for warts.',
            'contact_dermat': 'See a doctor if you experience frequent outbreaks, severe pain, or if warts multiply rapidly.',
            'diet_lifestyle': 'A diet rich in Lysine (for herpes) and reducing stress can help prevent outbreaks. Maintain a strong immune system.',
            'contagious': 'Yes, highly contagious through direct skin contact.',
            'duration': 'Herpes outbreaks typically last 1-2 weeks but the virus remains dormant for life. Warts can last months to years.',
            'doctor_visit_timeline': 'Within 2-3 days for new herpes outbreaks or spreading warts.'
        },
        'Light Diseases...': {
            'description': 'Skin reactions triggered by exposure to ultraviolet (UV) radiation or visible light.',
            'symptoms': 'Redness, itching, burning sensation, or hives appearing after sun exposure.',
            'home_remedy': 'Apply cool, damp cloths to the rash. Aloe vera can help soothe sun-irritated skin.',
            'solution': 'Use prescribed corticosteroid creams and oral antihistamines to reduce inflammation.',
            'precautions': 'Avoid sun exposure during peak hours (10 AM to 4 PM). Wear UPF clothing and broad-spectrum sunscreen.',
            'origin_causes': 'An abnormal immune system reaction to sunlight (photosensitivity), sometimes triggered by certain medications.',
            'contact_dermat': 'Consult a dermatologist to identify the exact cause and get a proper sun-protection and treatment plan.',
            'diet_lifestyle': 'Strict avoidance of intense sunlight. Check if any current medications are increasing photosensitivity.',
            'contagious': 'No, sun allergies and photosensitivity are not contagious.',
            'duration': 'Rashes usually resolve within a few days if further sun exposure is avoided.',
            'doctor_visit_timeline': 'Within 1 week if reactions occur frequently.'
        },
        'Lupus...': {
            'description': 'An autoimmune disease that can affect many parts of the body, including the skin, joints, and organs.',
            'symptoms': 'Butterfly-shaped rash on the face, fatigue, joint pain, and sensitivity to sunlight.',
            'home_remedy': 'Rest and avoid stress. However, lupus cannot be managed by home remedies alone.',
            'solution': 'Requires medical management including antimalarial drugs, corticosteroids, or immunosuppressants.',
            'precautions': 'Strict sun protection is crucial as UV light can trigger lupus flares. Maintain a healthy lifestyle.',
            'origin_causes': 'An autoimmune disease where the body\'s immune system mistakenly attacks healthy tissues, including the skin.',
            'contact_dermat': 'Serious condition. Must be managed by a rheumatologist and dermatologist to prevent organ damage.',
            'diet_lifestyle': 'Eat an anti-inflammatory diet. Quit smoking, get plenty of rest, and minimize physical and emotional stress.',
            'contagious': 'No, lupus is an autoimmune condition and is not contagious.',
            'duration': 'A chronic, lifelong condition with periods of flares and remissions.',
            'doctor_visit_timeline': 'ASAP/Within 1 week of noticing systemic symptoms or the characteristic butterfly rash.'
        },
        'Melanoma Skin Cancer Nevi and Moles': {
            'description': 'Melanoma is the most serious type of skin cancer. It develops in the cells (melanocytes) that produce melanin.',
            'symptoms': 'A mole that changes in size, shape, or color. Irregular borders or multiple colors within one mole.',
            'home_remedy': 'None. Do not attempt to remove or treat abnormal moles at home.',
            'solution': 'Surgical excision is the primary treatment. Advanced stages may require immunotherapy or targeted therapy.',
            'precautions': 'Perform monthly self-exams using the ABCDE rule. Always wear sunscreen and protective clothing outdoors.',
            'origin_causes': 'Caused by DNA damage to skin cells, typically from intense, intermittent UV radiation (like severe sunburns).',
            'contact_dermat': 'CRITICAL. Contact a dermatologist immediately if a mole changes in size, shape, color, or starts bleeding.',
            'diet_lifestyle': 'Rigorous sun protection every day. Avoid tanning beds entirely.',
            'contagious': 'No, cancer is not contagious.',
            'duration': 'Cancer can grow progressively if untreated. Moles are lifelong.',
            'doctor_visit_timeline': 'IMMEDIATELY (within 48 hours) for any suspicious mole changes.'
        },
        'Poison Ivy Photos...': {
            'description': 'A type of allergic contact dermatitis caused by contact with the urushiol oil in certain plants.',
            'symptoms': 'Redness, severe itching, swelling, and blisters often appearing in a linear pattern.',
            'home_remedy': 'Wash immediately with soap and water. Apply calamine lotion, oatmeal baths, and cool compresses.',
            'solution': 'Over-the-counter antihistamines. For severe cases, prescription oral or topical corticosteroids are needed.',
            'precautions': 'Learn to identify poison ivy/oak/sumac. Wear long pants and sleeves when in wooded areas.',
            'origin_causes': 'An allergic contact dermatitis caused by an oily resin called urushiol found in the leaves, stems, and roots of these plants.',
            'contact_dermat': 'See a doctor if the rash covers a large portion of your body, affects your face or genitals, or if you have trouble breathing.',
            'diet_lifestyle': 'Wash all clothes and pets that may have come in contact with the plant oils.',
            'contagious': 'The rash itself is not contagious, but the plant oil (urushiol) can be spread to others if it remains on skin or clothes.',
            'duration': 'Typically clears up on its own within 1 to 3 weeks.',
            'doctor_visit_timeline': 'Within 2-3 days if symptoms are spreading or severe.'
        },
        'Psoriasis pictures Lichen Planus...': {
            'description': 'Chronic autoimmune conditions. Psoriasis causes rapid skin cell buildup (plaques), while Lichen Planus involves itchy, flat-topped bumps.',
            'symptoms': 'Red patches covered with silvery scales (psoriasis) or purple, itchy, flat bumps (lichen planus).',
            'home_remedy': 'Apply aloe vera, take Epsom salt baths, and use thick ointments to keep plaques soft.',
            'solution': 'Topical steroids, vitamin D analogues, phototherapy, or systemic biologic medications.',
            'precautions': 'Avoid skin injuries (which can trigger new patches), manage stress, and limit alcohol consumption.',
            'origin_causes': 'An autoimmune condition that speeds up the lifecycle of skin cells, causing them to build up rapidly on the surface.',
            'contact_dermat': 'Consult a doctor for a tailored treatment plan, especially if it causes joint pain (psoriatic arthritis).',
            'diet_lifestyle': 'An anti-inflammatory diet may help. Limit alcohol intake and quit smoking as these trigger flare-ups.',
            'contagious': 'No, neither psoriasis nor lichen planus is contagious.',
            'duration': 'Chronic conditions that flare up and go into remission throughout life.',
            'doctor_visit_timeline': 'Within 1-2 weeks for diagnosis and starting a management plan.'
        },
        'Seborrheic Keratoses...': {
            'description': 'Noncancerous skin growths that are common in older adults. They often appear as waxy or wart-like growths.',
            'symptoms': 'Brown, black, or light tan growths that look "pasted on" the skin.',
            'home_remedy': 'None needed for medical reasons. Do not pick or scratch them off to avoid infection.',
            'solution': 'Usually requires no treatment. If irritated or for cosmetic reasons, a doctor can remove them via freezing (cryosurgery) or scraping.',
            'precautions': 'There is no known prevention. Have a doctor confirm they are not a more serious skin condition.',
            'origin_causes': 'The exact cause is unknown, but they are very common harmless skin growths that appear as people age. Not caused by sun exposure.',
            'contact_dermat': 'See a doctor if many appear suddenly, or if a growth bleeds, grows rapidly, or looks unusual.',
            'diet_lifestyle': 'No specific diet or lifestyle changes affect these benign growths.',
            'contagious': 'No, they are completely non-contagious.',
            'duration': 'Permanent unless removed by a doctor.',
            'doctor_visit_timeline': 'Next routine checkup, unless they change rapidly or bleed.'
        },
        'Systemic Disease': {
            'description': 'Skin manifestations of an underlying internal medical condition, such as diabetes, liver disease, or thyroid issues.',
            'symptoms': 'Unexplained rashes, changes in skin color, or itching that doesn\'t go away.',
            'home_remedy': 'Maintain a balanced diet, stay hydrated, and practice general gentle skincare.',
            'solution': 'Treatment entirely depends on diagnosing and treating the underlying internal condition (e.g., diabetes, thyroid disease).',
            'precautions': 'Get regular health checkups and blood work to monitor internal health.',
            'origin_causes': 'Skin symptoms that are secondary to an internal medical problem affecting the whole body.',
            'contact_dermat': 'Yes, a dermatologist can help identify if a skin rash is a sign of a serious internal illness.',
            'diet_lifestyle': 'Depends heavily on the underlying disease (e.g., managing sugar intake for diabetes).',
            'contagious': 'Usually not contagious, as it stems from an internal systemic issue, unless caused by an infectious disease.',
            'duration': 'Persists until the underlying internal medical condition is effectively managed.',
            'doctor_visit_timeline': 'Within 1 week for comprehensive blood work and diagnosis.'
        },
        'Tinea Ringworm...': {
            'description': 'A fungal infection that develops on the top layer of your skin. It is characterized by a red circular rash.',
            'symptoms': 'Ring-shaped rash, itching, redness, and scaly skin.',
            'home_remedy': 'Keep the area clean and dry. Garlic or tea tree oil have some antifungal properties but OTC creams are better.',
            'solution': 'Use over-the-counter antifungal creams (like clotrimazole or terbinafine) for 2-4 weeks.',
            'precautions': 'Do not share towels, clothing, or combs. Wear flip-flops in public showers and locker rooms.',
            'origin_causes': 'Caused by a highly contagious fungal infection of the skin (dermatophytes).',
            'contact_dermat': 'See a doctor if it doesn\'t improve after two weeks of over-the-counter treatment or if it spreads to the scalp.',
            'diet_lifestyle': 'Change socks and underwear daily. Ensure pets are checked by a vet if they have bald patches.',
            'contagious': 'Yes, highly contagious through skin contact or sharing items.',
            'duration': 'Usually resolves in 2 to 4 weeks with consistent antifungal treatment.',
            'doctor_visit_timeline': 'Within 1 week if OTC antifungal creams show no improvement.'
        },
        'Urticaria Hives': {
            'description': 'Red, itchy welts that result from a skin reaction. They can be triggered by many things, including allergens and stress.',
            'symptoms': 'Batches of red or skin-colored welts (wheals), which vary in size and change shape.',
            'home_remedy': 'Apply cold compresses to soothe itching. Wear loose-fitting cotton clothing.',
            'solution': 'Take non-drowsy oral antihistamines. Severe cases may require a short course of oral steroids.',
            'precautions': 'Identify and avoid your specific triggers (certain foods, medications, stress, or temperature changes).',
            'origin_causes': 'Triggered by the release of histamine in the skin, often due to an allergic reaction, infection, or stress.',
            'contact_dermat': 'Seek emergency care immediately if hives are accompanied by swelling of the lips/throat or difficulty breathing.',
            'diet_lifestyle': 'Keep a food diary to identify allergic triggers. Practice stress reduction techniques.',
            'contagious': 'No, hives are an allergic or immune response and are not contagious.',
            'duration': 'Acute hives last less than 6 weeks. Chronic hives can last for months or years.',
            'doctor_visit_timeline': 'Immediately for breathing issues; otherwise, within 2-3 days if they persist.'
        },
        'Vascular Tumors': {
            'description': 'Abnormal growths of blood vessels that can appear as red or purple marks on the skin.',
            'symptoms': 'Red, purple, or blue bumps or patches on the skin.',
            'home_remedy': 'None. Do not attempt to drain or puncture vascular growths.',
            'solution': 'Many require no treatment (like infantile hemangiomas). Others may be treated with beta-blockers, laser therapy, or surgery.',
            'precautions': 'Protect the area from trauma or friction to prevent bleeding.',
            'origin_causes': 'Caused by an abnormal, dense cluster of rapidly dividing blood vessels.',
            'contact_dermat': 'Consult a dermatologist or pediatrician to monitor the growth, especially if it interferes with vision or breathing, or starts bleeding.',
            'diet_lifestyle': 'No specific diet or lifestyle changes affect these growths.',
            'contagious': 'No, they are completely non-contagious.',
            'duration': 'Many infantile hemangiomas fade away by age 10. Others may be permanent unless removed.',
            'doctor_visit_timeline': 'Within 1-2 weeks for an initial evaluation.'
        },
        'Vasculitis Photos': {
            'description': 'Inflammation of the blood vessels. This can cause the blood vessel walls to thicken, narrow, weaken or scar.',
            'symptoms': 'Red or purple spots (purpura), often on the lower legs. May be accompanied by fever or fatigue.',
            'home_remedy': 'Elevate the legs if the rash is on the lower extremities. Rest and avoid standing for long periods.',
            'solution': 'Medical treatment with corticosteroids or immunosuppressive drugs is required to stop blood vessel inflammation.',
            'precautions': 'Manage any underlying autoimmune conditions or infections. Avoid medications that trigger it.',
            'origin_causes': 'An autoimmune response where the immune system attacks the blood vessels, causing them to leak blood into the skin.',
            'contact_dermat': 'Serious condition. Seek medical attention promptly to evaluate for internal organ involvement.',
            'diet_lifestyle': 'Rest during flare-ups. Eat an anti-inflammatory diet and avoid intense prolonged physical exertion.',
            'contagious': 'No, vasculitis is an autoimmune disorder and is not contagious.',
            'duration': 'Can be acute (lasting weeks) or chronic (lasting years with flare-ups).',
            'doctor_visit_timeline': 'ASAP/Within 24-48 hours for new purpura or spots.'
        },
        'Warts Molluscum...': {
            'description': 'Contagious viral infections of the skin. Warts are caused by HPV, while Molluscum is caused by the molluscum contagiosum virus.',
            'symptoms': 'Small, hard bumps (warts) or small, firm, raised bumps with a central pit (molluscum).',
            'home_remedy': 'Apply over-the-counter salicylic acid or use the duct tape occlusion method for regular warts.',
            'solution': 'Cryotherapy (freezing), cantharidin application, or minor surgical removal by a doctor.',
            'precautions': 'Do not pick at the bumps. Avoid sharing towels and avoid direct contact with others\' lesions.',
            'origin_causes': 'Caused by viral infections: Human Papillomavirus (HPV) for warts, and Molluscum contagiosum virus.',
            'contact_dermat': 'See a doctor if they are painful, spreading rapidly, or located on sensitive areas like the face or genitals.',
            'diet_lifestyle': 'Maintain a strong immune system. Wear shower shoes in public pools and locker rooms.',
            'contagious': 'Yes, highly contagious through direct skin-to-skin contact or touching contaminated surfaces.',
            'duration': 'May resolve on their own in 6 to 12 months, but can persist for years without treatment.',
            'doctor_visit_timeline': 'Within 2-3 weeks if they are spreading or becoming painful.'
        }
    }

except Exception as e:
    print(f"Error loading model: {e}")
    # If the model does not load, set the model variables to None
    model = None 
    gatekeeper_model = None

# Flask Configuration
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload and instance folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('instance', exist_ok=True)


# --- HELPER FUNCTIONS ---

def init_db():
    # Database initialization code (Your existing code)
    conn = sqlite3.connect('instance/skin_disease.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, image_path TEXT, disease_name TEXT, confidence REAL, recommendations TEXT, body_area TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ML PREDICTION LOGIC ---

def ml_predict_logic(image_path, model, gatekeeper):
    if model is None or gatekeeper is None:
        return "Model Error", 0.0, "Model failed to load on startup. Check terminal logs."
    
    try:
        # Load image for processing
        img = image.load_img(image_path, target_size=(224, 224)) # MobileNetV2 prefers 224x224
        img_array_gk = image.img_to_array(img)
        img_array_gk = np.expand_dims(img_array_gk, axis=0)
        img_array_gk = mobilenet_preprocess(img_array_gk)
        
        # --- STAGE 1: GATEKEEPER (Enhanced to detect non-skin images) ---
        gk_predictions = gatekeeper.predict(img_array_gk)
        decoded_gk = decode_predictions(gk_predictions, top=3)[0]
        
        # Check for forbidden categories (non-skin/non-medical)
        # MobileNetV2 (ImageNet) classes that often represent humans or random objects
        forbidden_terms = ['person', 'face', 'man', 'woman', 'guy', 'girl', 'boy', 'groom', 'mask', 'sunglasses', 'wig']
        
        is_forbidden = False
        for _, label, prob in decoded_gk:
            # If the gatekeeper is very confident it's a person/face/etc
            if any(term in label.lower() for term in forbidden_terms) and prob > 0.4:
                is_forbidden = True
                break
        
        if is_forbidden:
            return "Wrong Image", 0.0, "The captured image appears to be a face or person. Please capture a clear, close-up photo of the affected skin area only."
        
        # --- STAGE 2: SKIN DISEASE MODEL ---
        # Load image at 128x128 for the custom model
        img_skin = image.load_img(image_path, target_size=(128, 128))
        x_rgb = image.img_to_array(img_skin)
        
        # NOTE: Further testing reveals RGB RAW (0-255) performs best across various conditions
        img_array_skin = np.expand_dims(x_rgb, axis=0)
        
        # NOTE: Model was trained on RAW pixel values (0-255), NOT normalized.
        
        # Make prediction
        predictions = model.predict(img_array_skin)
        
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        print(f"DEBUG: Predicted {CLASS_NAMES[predicted_class_index]} with {confidence:.2f}% confidence.")
        
        # Because the model is now seeing the pixel range it was trained on,
        # its confidence will behave normally (very high for obvious features).
        # We can restore a robust safety threshold of 50%.
        if confidence < 50.0:
            return "Unrecognized Image", 0.0, "The uploaded image does not appear to match any of our trained skin diseases with acceptable confidence. Please ensure you upload a clear, focused picture of a skin condition."
        
        disease_name = CLASS_NAMES[predicted_class_index]
        recommendations = f"Keep the area clean. Based on {disease_name} prediction, seek immediate advice from a dermatologist."
        
        return disease_name, confidence, recommendations
        
    except Exception as e:
        return f"Processing Error: {e}", 0.0, "Could not process image."

# Initialize database on startup
with app.app_context():
    init_db()


# --- FLASK ROUTES (Login, Signup, Home, Logout) ---

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

# Your login, signup, and logout functions remain unchanged...

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = sqlite3.connect('instance/skin_disease.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('instance/skin_disease.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                    (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
    
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Example to fetch recent predictions for display on home page
    conn = sqlite3.connect('instance/skin_disease.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5''',
             (session['user_id'],))
    recent_predictions = c.fetchall()
    conn.close()

    return render_template('home.html', username=session.get('username'), recent_predictions=recent_predictions)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

# --- ML PREDICTION ROUTE (Replaces the Mock/Duplicated Code) ---

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    image_path = None
    
    # 1. Handle file upload (from form)
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(f"{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
    
    # 2. Handle camera capture (from base64 data)
    elif 'image_data' in request.form:
        image_data = request.form['image_data']
        image_data = image_data.split(',')[1]
        filename = f"{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_capture.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
        except:
             return jsonify({'error': 'Invalid base64 data received'}), 400

    if not image_path:
        return jsonify({'error': 'No image provided or file type not allowed'}), 400
    
    # Get body area from request
    body_area = request.form.get('body_area', 'General')

    # 3. Perform ML Prediction
    disease_name, confidence, recommendations = ml_predict_logic(image_path, model, gatekeeper_model)
    
    # 4. Save prediction to database
    conn = sqlite3.connect('instance/skin_disease.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions (user_id, image_path, disease_name, confidence, recommendations, body_area)
                  VALUES (?, ?, ?, ?, ?, ?)''',
              (session['user_id'], os.path.basename(image_path), disease_name, confidence, recommendations, body_area))
    prediction_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # 5. Return JSON result
    return jsonify({
        'success': True,
        'prediction_id': prediction_id,
        'disease': disease_name,
        'confidence': round(confidence, 2),
        'recommendations': recommendations,
        'image_url': url_for('static', filename='uploads/' + os.path.basename(image_path))
    })

# --- HISTORY AND RESULT ROUTES ---

@app.route('/result/<int:prediction_id>')
def result(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('instance/skin_disease.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM predictions WHERE id = ? AND user_id = ?''',
             (prediction_id, session['user_id']))
    prediction = c.fetchone()
    conn.close()
    
    if not prediction:
        flash('Prediction not found', 'error')
        return redirect(url_for('home'))
    
    result_data = {
        'id': prediction[0],
        'image_path': url_for('static', filename='uploads/' + os.path.basename(prediction[2])),
        'disease': prediction[3],
        'confidence': round(prediction[4], 2),
        'recommendations': prediction[5],
        'body_area': prediction[7],
        'created_at': prediction[6]
    }
    
    # Try to append detailed info if available
    try:
        if prediction[3] in DISEASE_INFO:
            result_data['details'] = DISEASE_INFO[prediction[3]]
    except:
        pass
    
    return render_template('result.html', result=result_data)

@app.route('/history')
def history():
    # Your history function logic... (Returns JSON data)
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('instance/skin_disease.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 10''',
             (session['user_id'],))
    predictions = c.fetchall()
    conn.close()
    
    history_data = [{
        'id': p[0],
        'image_path': url_for('static', filename='uploads/' + os.path.basename(p[2])),
        'disease': p[3],
        'confidence': round(p[4], 2),
        'created_at': p[6]
    } for p in predictions]
    
    return jsonify(history_data)

if __name__ == '__main__':
    app.run(debug=True)