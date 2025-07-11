
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# טוען את המודל
model = joblib.load("final_catboost_model.pkl")

st.set_page_config(page_title="חיזוי עבירה שנייה", page_icon="🏗", layout="centered")

st.title("חיזוי האם האיתור יהפוך לעבירה מנהלית")

st.markdown("יש למלא את כל השדות הבאים:")

with st.form("prediction_form"):
    structure_type = st.selectbox("🏗 סוג מבנה ראשון", ["Yes", "No"], index=None, placeholder="בחרי סוג")
    city_area = st.selectbox("🏙 אזור עירוני", ["Yes", "No"], index=None, placeholder="בחרי אזור")
    jewish_area = st.selectbox("🕍 אזור יהודי", ["Yes", "No"], index=None, placeholder="בחרי אזור")
    district = st.selectbox("📍 מחוז", [
        "District_Center", "District_Jerusalem", "District_North", "District_South"
    ], index=None, placeholder="בחרי מחוז")
    land_designation = st.selectbox("🗺 ייעוד קרקע", [
        "District_land_designation_Agricultural area",
        "District_land_designation_Beach/ River",
        "District_land_designation_Industrial & Employment",
        "District_land_designation_Nature & Conservation",
        "District_land_designation_Tourism & Commerce",
        "District_land_designation_Unknown & Other",
        "District_land_designation_Urban & Residential",
        "District_land_designation_Village"
    ], index=None, placeholder="בחרי ייעוד")

    potential_type = st.selectbox("🧱 אופי איתור ראשון", [
        'Construction Violation', 'No Permit', 'Construction Completion', 'New Construction Start',
        'Illegal Use', 'Other Violation', 'Structure Placement', 'Construction Without Permit',
        'Expansion', 'Illegal Use With Permit', 'Construction Not According To Permit',
        'New Construction', 'Mobile Structures', 'Use Violation', 'Illegal Use By Plan',
        'Continued Construction'
    ], index=None, placeholder="בחרי אופי")

    submitted = st.form_submit_button("חשב תוצאה")
    reset = st.form_submit_button("איפוס הטופס")

if submitted:
    if None in [structure_type, city_area, jewish_area, district, land_designation, potential_type]:
        st.error("יש למלא את כל השדות לפני החישוב.")
    else:
        # יצירת עמודות מקודדות
        input_dict = {}

        for d in ["District_Center", "District_Jerusalem", "District_North", "District_South"]:
            input_dict[d] = 1 if d == district else 0

        for l in [
            "District_land_designation_Agricultural area",
            "District_land_designation_Beach/ River",
            "District_land_designation_Industrial & Employment",
            "District_land_designation_Nature & Conservation",
            "District_land_designation_Tourism & Commerce",
            "District_land_designation_Unknown & Other",
            "District_land_designation_Urban & Residential",
            "District_land_designation_Village"
        ]:
            input_dict[l] = 1 if l == land_designation else 0

        for p in [
            'Construction Violation', 'No Permit', 'Construction Completion', 'New Construction Start',
            'Illegal Use', 'Other Violation', 'Structure Placement', 'Construction Without Permit',
            'Expansion', 'Illegal Use With Permit', 'Construction Not According To Permit',
            'New Construction', 'Mobile Structures', 'Use Violation', 'Illegal Use By Plan',
            'Continued Construction'
        ]:
            col = "Potential_Type_1_Grouped_" + p
            input_dict[col] = 1 if p == potential_type else 0

        input_dict["Kal_Kashiah_1"] = 1 if structure_type == "Yes" else 0
        input_dict["city_erea"] = 1 if city_area == "Yes" else 0
        input_dict["jewish_e"] = 1 if jewish_area == "Yes" else 0

        # אלו העמודות שלא נכללות בטופס – נשמרות כ-NaN
        for col in ["Kal_Kashiah_2"] + [f"Potential_Type_2_Grouped_{p}" for p in [
            'Construction Violation', 'No Permit', 'Construction Completion', 'New Construction Start',
            'Illegal Use', 'Other Violation', 'Structure Placement', 'Construction Without Permit',
            'Expansion', 'Illegal Use With Permit', 'Construction Not According To Permit',
            'New Construction', 'Mobile Structures', 'Use Violation', 'Illegal Use By Plan',
            'Continued Construction'
        ]]:
            input_dict[col] = np.nan

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        result_text = "✔️ האיתור צפוי להפוך לעבירה מנהלית" if prediction == 1 else "ℹ️ האיתור יישאר מודיעיני"
        st.success(result_text)
elif reset:
    st.experimental_rerun()
