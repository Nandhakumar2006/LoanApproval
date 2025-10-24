import gradio as gr
import numpy as np

import joblib

pt = joblib.load("yeojohnson_transformer.joblib")
rf_model = joblib.load("random_forest_compressed.joblib")

def predict_loan(
    person_age,
    person_emp_exp,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    person_income,
    person_gender,
    person_education,
    person_home_ownership,
    loan_intent,
    previous_loan_defaults_on_file
):

    person_income_transformed = pt.transform(np.array(person_income).reshape(-1, 1))[0][0]

    gender_female = 1 if person_gender == "Female" else 0
    gender_male = 1 if person_gender == "Male" else 0

    educ_levels = ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    educ_encoded = [1 if person_education == lvl else 0 for lvl in educ_levels]

    home_types = ["MORTGAGE", "OTHER", "OWN", "RENT"]
    home_encoded = [1 if person_home_ownership == h else 0 for h in home_types]

    intents = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    intent_encoded = [1 if loan_intent == i else 0 for i in intents]

    prev_no = 1 if previous_loan_defaults_on_file == "No" else 0
    prev_yes = 1 if previous_loan_defaults_on_file == "Yes" else 0

    features = [
        person_age,
        person_emp_exp,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        person_income_transformed,   # <-- transformed value passed to model
        gender_female,
        gender_male,
        *educ_encoded,
        *home_encoded,
        *intent_encoded,
        prev_no,
        prev_yes
    ]

    X = np.array(features).reshape(1, -1)
    pred = rf_model.predict(X)[0]
    prob = rf_model.predict_proba(X)[0][1]

    if pred == 1:
        status = f"✅ **Loan Approved**"
        confidence = f"Confidence: {prob*100:.1f}%"
        style = "color: green; font-weight: bold;"
    else:
        status = f"❌ **Loan Not Approved (Default Risk)**"
        confidence = f"Confidence: {(1-prob)*100:.1f}%"
        style = "color: red; font-weight: bold;"

    return f"<div style='{style}; font-size: 18px;'>{status}<br><span style='font-size:16px;'>{confidence}</span></div>"

# ------------------- UI -------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")) as demo:
    gr.Markdown(
        """
        <h1 style='text-align:center; color:#4F46E5;'>💳 Loan Approval Predictor</h1>
        <p style='text-align:center; font-size:16px;'>
        Enter applicant details below and get an instant AI-powered loan approval prediction.<br>
        Powered by a <b>Random Forest Machine Learning Model</b> trained with Yeo–Johnson transformation.
        </p>
        <hr>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            person_age = gr.Number(label="Age")
            person_emp_exp = gr.Number(label="Employment Experience (yrs)")
            person_income = gr.Number(label="Annual Income ($)", info="Enter actual annual income value")  # <-- changed label
            credit_score = gr.Number(label="Credit Score")
            cb_person_cred_hist_length = gr.Number(label="Credit History Length (yrs)")

        with gr.Column(scale=1):
            loan_amnt = gr.Number(label="Loan Amount ($)")
            loan_int_rate = gr.Number(label="Interest Rate (%)")
            loan_percent_income = gr.Number(label="Loan Percent of Income")
            loan_intent = gr.Dropdown(
                ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
                label="Loan Intent"
            )
            person_home_ownership = gr.Dropdown(
                ["MORTGAGE", "OTHER", "OWN", "RENT"], label="Home Ownership Type"
            )

    with gr.Row():
        person_gender = gr.Radio(["Male", "Female"], label="Gender")
        person_education = gr.Dropdown(
            ["Associate", "Bachelor", "Doctorate", "High School", "Master"], label="Education Level"
        )
        previous_loan_defaults_on_file = gr.Radio(["No", "Yes"], label="Previous Loan Defaults")

    with gr.Row():
        submit_btn = gr.Button("🔍 Predict Loan Approval", variant="primary")

    output_box = gr.HTML(label="Prediction Result")

    submit_btn.click(
        fn=predict_loan,
        inputs=[
            person_age,
            person_emp_exp,
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length,
            credit_score,
            person_income,
            person_gender,
            person_education,
            person_home_ownership,
            loan_intent,
            previous_loan_defaults_on_file
        ],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()
