from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import google.generativeai as genai
from flask_cors import CORS


genai.configure(api_key="AIzaSyD1FPKl0lENNaIw8JGtMBzPXopVDIqcab8")
model = genai.GenerativeModel("gemini-1.5-flash")


app = Flask(__name__)
CORS(app)


loaded_model = load_model('stone_cyst_tumor_model.h5')
loaded_model.compile() 


class_labels = ["Cyst", "Normal", "Stone", "Tumor"]

# Function to generate the prevention report using Gemini LLM
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        - Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        - Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        - Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        - Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        - Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        - Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        - Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        - Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        - Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        - Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        - Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        - In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        - Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        - Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        - Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        - Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        - Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        - Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        - Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        - Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        - Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        - Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        - Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        - Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        - Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        - Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk}
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# API route to upload image and get prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded."}), 400
        
       
        image = request.files['image']
        
        
        image_path = 'uploaded_image.png'
        image.save(image_path)
        
       
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "Failed to load image."}), 400
        
       
        img_resized = cv2.resize(img, (200, 200))
        img_normalized = img_resized / 255.0
        img_reshaped = img_normalized.reshape(1, 200, 200, 1)
        
       
        predictions = loaded_model.predict(img_reshaped)
        
        # Get predicted label
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        
        
        risk_score = f"{predictions[0][predicted_class] * 100:.2f}%"
        
        # Generate the prevention report
        age = request.form.get('age', 30)  # Default age is 30, replace with dynamic input if needed
        risk = "Kidney"
        report = generate_prevention_report(risk, predicted_label, age)
        
        # Return the results as a JSON response
        return jsonify({
            "risk_name": risk,
            "predicted_class": predicted_label,
            "risk_score": risk_score,
            "wellness_report": report
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
