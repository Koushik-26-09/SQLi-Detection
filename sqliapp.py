import gradio as gr
import joblib
import numpy as np

# Load pre-trained models
xgb = joblib.load("C:\\Users\\koush\\OneDrive\\Desktop\\MiniProject\\xgb_model.pkl")
vectorizer = joblib.load("C:\\Users\\koush\\OneDrive\\Desktop\\MiniProject\\tfidf_vectorizer.pkl")
encoder = joblib.load("C:\\Users\\koush\\OneDrive\\Desktop\\MiniProject\\encoder.pkl")

# Prediction function with confidence and HTML output
def predict(query):
    query_cleaned = query.strip().lower()
    query_vector = vectorizer.transform([query_cleaned]).toarray()
    query_encoded = encoder.predict(query_vector)
    prediction = xgb.predict(query_encoded)
    proba = xgb.predict_proba(query_encoded)[0][1]  # probability of class 1 (Malicious)

    # Confidence levels
    confidence = round(proba * 100, 2)

    # Risk Level Based on Confidence
    if prediction[0] == 1:  # Malicious
        label_html = "<span style='color:red; font-weight:bold;'>🛑 Malicious</span>"
        if confidence > 70:
            risk = "🔴 High"
        elif confidence > 40:
            risk = "🟡 Medium"
        else:
            risk = "🟢 Low"
    else:  # Normal
        label_html = "<span style='color:lightgreen; font-weight:bold;'>✅ Normal</span>"
        if confidence > 70:
            risk = "🔴 High"
        elif confidence > 40:
            risk = "🟡 Medium"
        else:
            risk = "🟢 Low"

    return label_html, confidence, risk

# Build the UI
with gr.Blocks(title="SQL Injection Detection", theme=gr.themes.Base(primary_hue="red", secondary_hue="gray")) as demo:
    gr.Markdown("""
        <h1 style="color:#e60000;">🔍 SQL Injection Detection</h1>
        <p style="color:#f2f2f2;">
            Detect potentially malicious SQL input using a trained machine learning model.<br>
            Helps safeguard applications from unauthorized database manipulation.
        </p>
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="🧾 Enter SQL Query", placeholder="e.g. ' OR 1=1 --", lines=1)
            example_inputs = gr.Examples(
                examples=[
                    "' OR 1=1 --",
                    "SELECT * FROM users WHERE username='SASTRA'",
                    "1) or benchmark(10000000,MD5(1))#",
                    "Normal user input"
                ],
                inputs=input_text
            )
            submit_btn = gr.Button("🚨 Detect")
            clear_btn = gr.Button("🧹 Clear")

        with gr.Column():
            result = gr.HTML(label="🔍 Detection Result")
            confidence_slider = gr.Slider(minimum=0, maximum=100, label="📊 Confidence (%)", interactive=False)
            risk_level = gr.Textbox(label="📊 Risk Level", interactive=False)

    submit_btn.click(fn=predict, inputs=input_text, outputs=[result, confidence_slider, risk_level])
    clear_btn.click(fn=lambda: ("", 0, ""), inputs=None, outputs=[result, confidence_slider, risk_level])

    gr.Markdown("<p style='color: #ff9999;'>💡 <strong>Security Tip:</strong> Always sanitize inputs and use parameterized queries to avoid SQL injection attacks!</p>")

# Launch
demo.launch(share=True)
