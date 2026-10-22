import gradio as gr
import pandas as pd
import joblib

print("Step 1: Imports successful")

model = joblib.load("house_price_model.pkl")
print("Step 2: Model loaded successfully")

def predict(location, sqft, bath, bedrooms):
    data = pd.DataFrame([{
        "location": location.lower(),
        "total_sqft": sqft,
        "bath": bath,
        "bedrooms": bedrooms
    }])

    prediction = model.predict(data)[0]

    return f"Estimated House Price: ₹ {prediction:.2f} Lakhs"

print("Step 3: Predict function created")

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Location", placeholder="Example: whitefield"),
        gr.Number(label="Total Square Feet"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Bedrooms"),
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="🏠 Bengaluru House Price Prediction",
    description="Enter the house details to estimate the selling price."
)

print("Step 4: Interface created")

if __name__ == "__main__":
    print("Step 5: Launching Gradio...")
    demo.launch(debug=True)