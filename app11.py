import streamlit as st
import pandas as pd
import numpy as np
import joblib

# טוען את המודל
model = joblib.load("final_catboost_model.pkl")

st.set_page_config(page_title="חיזוי עבירה שנייה", page_icon="🏗", layout="centered")
st.title("סיווג האם האיתור יהפוך למנהלית")
st.markdown("יש למלא את כל השדות הבאים:")

with st.form("prediction_form"):
    district = st.selectbox("📍 מחוז", ["Center", "Jerusalem", "North", "South"], index=None, placeholder="בחרי מחוז")

    q1 = st.selectbox("📆 רבעון איתור ראשון", ["Q1", "Q2", "Q3", "Q4"], index=None, placeholder="בחרי רבעון")

    types = [
        "Earthworks and clearance", "Site preparation", "Roads and approaches",
        "Drilling and foundations", "Base for columns", "Infrastructure",
        "Skeleton – beginning", "Skeleton – advanced", "Skeleton – general",
        "new floor", "concrete floor", "main structure", "light structures",
        "mobile structures", "add-ons and reinforcements", "termination/disposal"
    ]
    type1 = st.selectbox("🧱 אופי איתור ראשון", types, index=None, placeholder="בחרי אופי")

    land_options = [
        "Agricultural area", "Beach/ River", "Industrial & Employment",
        "Nature & Conservation", "Tourism & Commerce", "Unknown & Other",
        "Urban & Residential", "Village"
    ]
    land_use = st.selectbox("🗺 ייעוד קרקע", land_options, index=None, placeholder="בחרי ייעוד")

    structure1 = st.selectbox("🏗 סוג מבנה איתור ראשון", ["בחר", "קל", "קשיח"])
    city_area = st.selectbox("🏙 אזור עירוני", ["בחר", "כן", "לא"])
    jewish = st.selectbox("🕍 אזור יהודי", ["בחר", "כן", "לא"])

    submitted = st.form_submit_button("חשב תוצאה")
    reset = st.form_submit_button("איפוס הטופס")

if submitted:
    if "בחר" in [district, q1, type1, land_use, structure1, city_area, jewish]:
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

        # העמודות שלא קיימות בטופס
        for col in ["Kal_Kashiah_2"] + [f"Potential_Type_2_Grouped_{t}" for t in types]:
            features[col] = np.nan

        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✔️ האיתור צפוי להפוך לעבירה מנהלית")
        else:
            st.info("ℹ️ האיתור יישאר מודיעיני")

elif reset:
    st.rerun()
