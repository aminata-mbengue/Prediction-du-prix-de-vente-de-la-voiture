# -*- coding: utf-8 -*-
"""
app.py — Application Streamlit
Prédiction du prix de vente d'une voiture d'occasion
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction prix de vente voiture", page_icon="🚗")

MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"
ENCODERS_PATH = "encoders.joblib"


@st.cache_resource
def load_artifacts():
    for path in (MODEL_PATH, ENCODERS_PATH):
        if not os.path.exists(path):
            st.error(f"Fichier introuvable : {path}. Vérifie qu'il est bien présent dans le repo GitHub.")
            st.stop()

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    # encoders.joblib peut avoir été sauvegardé comme liste [enc0, enc1, enc2]
    # (ordre : Fuel_Type, Seller_Type, Transmission) plutôt que comme dict
    if isinstance(encoders, list):
        cat_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
        encoders = dict(zip(cat_cols, encoders))

    return model, encoders


best_model, encoders = load_artifacts()

FEATURE_COLS = ['Kms_Driven', 'Present_Price', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Age']


def Pred_func(kms_driven, present_price, fuel_type, seller_type, transmission, age):
    """Prédiction simple à partir des caractéristiques d'une voiture."""
    fuel_enc = encoders['Fuel_Type'].transform([fuel_type])[0]
    seller_enc = encoders['Seller_Type'].transform([seller_type])[0]
    trans_enc = encoders['Transmission'].transform([transmission])[0]

    x_new = np.array([[
        kms_driven,
        present_price,
        fuel_enc,
        seller_enc,
        trans_enc,
        age
    ]], dtype=float)

    # Pas de scaler.transform ici : le modèle a été entraîné sur des valeurs brutes
    prediction = best_model.predict(x_new)[0]
    return round(float(prediction), 2)


def Pred_func_csv(df_in):
    """Prédiction multiple à partir d'un DataFrame CSV importé."""
    predictions = []
    for _, row in df_in.iterrows():
        y_pred = Pred_func(
            row['Kms_Driven'],
            row['Present_Price'],
            row['Fuel_Type'],
            row['Seller_Type'],
            row['Transmission'],
            row['Age']
        )
        predictions.append(y_pred)
    df_out = df_in.copy()
    df_out['Predicted Selling_Price'] = predictions
    return df_out


st.title("🚗 Prédiction du prix de vente d'une voiture d'occasion")

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        kms = st.number_input("Kms parcourus (Kms_Driven)", min_value=0, value=45000, step=1000)
    with col2:
        price = st.number_input("Prix catalogue neuf en k$ (Present_Price)", min_value=0.0, value=7.0, step=0.1)

    col3, col4, col5 = st.columns(3)
    with col3:
        fuel = st.selectbox("Type de carburant", options=list(encoders['Fuel_Type'].classes_))
    with col4:
        seller = st.selectbox("Type de vendeur", options=list(encoders['Seller_Type'].classes_))
    with col5:
        trans = st.selectbox("Transmission", options=list(encoders['Transmission'].classes_))

    age = st.number_input("Âge du véhicule (années)", min_value=0, max_value=30, value=6, step=1)

    if st.button("Prédire", type="primary"):
        result = Pred_func(kms, price, fuel, seller, trans, age)
        st.success(f"Prix de vente prédit : **{result} k$**")

with tab2:
    st.markdown(
        "Importer un fichier CSV avec les colonnes : "
        "`Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age`"
    )
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURE_COLS if c not in df_in.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le CSV : {missing}")
        else:
            df_out = Pred_func_csv(df_in)
            st.dataframe(df_out)
            st.download_button(
                "Télécharger le fichier avec les prédictions",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
