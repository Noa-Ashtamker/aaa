# === app11.py - Streamlit App ===
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and column order
saved = joblib.load("final_catboost_model_with_cols.pkl")
model = saved['model']
model_columns = saved['columns']

st.set_page_config(page_title="חיזוי עבירה שנייה", page_icon="🏗", layout="centered")
st.title("סיווג האם האיתור יהפוך למנהלית")
st.markdown("יש למלא את כל השדות הבאים:")

with st.form("prediction_form"):
    district = st.selectbox("📍 מחוז", ["Center", "Jerusalem", "North", "South"], index=None, placeholder="בחר מחוז")
    q1 = st.selectbox("📆 רבעון איתור ראשון", ["Q1", "Q2", "Q3", "Q4"], index=None, placeholder="בחר רבעון")

    types = [
        "Earthworks and clearance", "Site preparation", "Roads and approaches",
        "Drilling and foundations", "Base for columns", "Infrastructure",
        "Skeleton – beginning", "Skeleton – advanced", "Skeleton – general",
        "new floor", "concrete floor", "main structure", "light structures",
        "mobile structures", "add-ons and reinforcements", "termination/disposal"
    ]
    type1 = st.selectbox("🧱 אופי איתור ראשון", types, index=None, placeholder="בחר אופי")

    land_options = [
        "Agricultural area", "Beach/ River", "Industrial & Employment",
        "Nature & Conservation", "Tourism & Commerce", "Unknown & Other",
        "Urban & Residential", "Village"
    ]
    land_use = st.selectbox("🗺 ייעוד קרקע", land_options, index=None, placeholder="בחרי ייעוד")

    structure1 = st.selectbox("🏗 סוג מבנה איתור ראשון", ["קל", "קשיח"], index=None, placeholder="בחר סוג")
    city_area = st.selectbox("🏙 אזור עירוני", ["כן", "לא"], index=None, placeholder="בחר אזור")
    jewish = st.selectbox("🕍 אזור יהודי", ["כן", "לא"], index=None, placeholder="בחר אזור")

    submitted = st.form_submit_button("חשב תוצאה")
    reset = st.form_submit_button("איפוס הטופס")

if submitted:
    if None in [district, q1, type1, land_use, structure1, city_area, jewish]:
        st.warning("אנא מלא את כל השדות לפני ביצוע חיזוי.")
    else:
        features = {
            'District_Center': int(district == 'Center'),
            'District_Jerusalem': int(district == 'Jerusalem'),
            'District_North': int(district == 'North'),
            'District_South': int(district == 'South'),

            'Quarter_Update_1_Q1': int(q1 == 'Q1'),
            'Quarter_Update_1_Q2': int(q1 == 'Q2'),
            'Quarter_Update_1_Q3': int(q1 == 'Q3'),
            'Quarter_Update_1_Q4': int(q1 == 'Q4'),
        }

        for t in types:
            features[f"Potential_Type_1_Grouped_{t}"] = int(type1 == t)

        for land in land_options:
            features[f"District_land_designation_{land}"] = int(land_use == land)

        features['Kal_Kashiah_1'] = int(structure1 == "קשיח")
        features['city_erea'] = int(city_area == "כן")
        features['jewish_e'] = int(jewish == "כן")

        # Fill second-detection fields with NaN
        for col in model_columns:
            if col not in features:
                features[col] = np.nan

        input_df = pd.DataFrame([features])[model_columns]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("✔️ האיתור צפוי להפוך לעבירה מנהלית")
        else:
            st.info("ℹ️ האיתור יישאר מודיעיני")

elif reset:
    st.experimental_set_query_params(reset=str(np.random.randint(0, 100000)))
    st.experimental_rerun()
