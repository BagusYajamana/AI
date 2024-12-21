import streamlit as st
import pandas as pd
import datetime
import xgboost as xgb


def main():
    html_header = """
        <div style="background-color:#1E90FF; padding:16px; border-radius:10px;">
            <h1 style="color:white; text-align:center;">Car Price Prediction App</h1>
        </div>
    """

    html_footer = """
        <div style="background-color:#1E90FF; padding:10px; border-radius:10px; margin-top:20px;">
            <h4 style="color:white; text-align:center;">Powered by XGBoost | Streamlit</h4>
        </div>
    """

    page_background = """
        <style>
            body {
                background-color: #E8F0F2;
            }
            .stButton button {
                background-color: #1E90FF; 
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
            }
        </style>
    """

    st.markdown(page_background, unsafe_allow_html=True)
    st.markdown(html_header, unsafe_allow_html=True)

    try:
        model = xgb.XGBRegressor()
        #--->check<---
        model.load_model(r"C:\\Users\\yajam\\xgb_model.json")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.markdown("### Enter the details below to estimate your car's resale value:")

    present_price = st.number_input(
        "Current Market Price of the Car (in Rp):",
        min_value=0.0,
        step=100000.0,
        format="%.2f",
        help="Enter the current market price of the car in Indonesian Rupiah."
    )
    kms_driven = st.number_input(
        "Kilometers Driven:",
        min_value=0,
        step=100,
        help="Enter the total kilometers the car has been driven."
    )

    fuel_type = st.selectbox("Fuel Type:", ['Petrol', 'Diesel', 'Electric'], help="Select the fuel type of the car.")
    fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'Electric': 2}
    fuel_type_encoded = fuel_type_mapping[fuel_type]

    seller_type = st.selectbox("Seller Type:", ['Dealer', 'Individual'], help="Specify whether you're a dealer or an individual seller.")
    seller_type_encoded = 0 if seller_type == 'Dealer' else 1

    transmission_type = st.selectbox("Transmission Type:", ['Manual', 'Automatic'], help="Select the transmission type of the car.")
    transmission_encoded = 0 if transmission_type == 'Manual' else 1

    owner_count = st.number_input(
        "Number of Previous Owners:",
        min_value=0,
        step=1,
        format="%d",
        help="Enter the number of previous owners the car has had."
    )

    current_year = datetime.datetime.now().year
    purchase_year = st.number_input(
        "Year of Purchase:",
        min_value=2000,
        max_value=current_year,
        step=1,
        help="Enter the year the car was purchased."
    )
    car_age = current_year - purchase_year

    input_data = pd.DataFrame({
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type_encoded],
        'Seller_Type': [seller_type_encoded],
        'Transmission': [transmission_encoded],
        'Owner': [owner_count],
        'Age': [car_age]
    })

    if st.button("Predict Car Price"):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Resale Value: Rp {prediction:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.markdown(html_footer, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
