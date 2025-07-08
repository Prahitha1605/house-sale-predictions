
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import locale

# Function to format numbers in Indian system
def format_indian_number(number):
    try:
        # Set locale to Indian English
        locale.setlocale(locale.LC_ALL, 'en_IN')
        # Format as currency without INR symbol (add ₹ manually)
        formatted = locale.currency(number, grouping=True, international=False)
        # Remove trailing ".00" if present
        if formatted.endswith(".00"):
            formatted = formatted[:-3]
        return f"{formatted}"
    except locale.Error:
        # Fallback to custom formatting
        integer_part = int(number)
        decimal_part = f"{number:.2f}".split('.')[-1]
        s = str(integer_part)
        if len(s) <= 3:
            return f"₹{s}.{decimal_part}"
        first = s[-3:]
        rest = s[:-3]
        formatted = ""
        while rest:
            formatted = "," + rest[-2:] + formatted
            rest = rest[:-2]
        return f"₹{rest[1:] if rest else ''}{first}.{decimal_part}"

# Set page title
st.title("Chennai House Price Prediction")

# Load the model and encoders
try:
    with open('chennai_home_prices_model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pickle', 'rb') as file:
        encoders = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# Define expected columns (include BEDROOM_SQFT_RATIO and BATHROOM_SQFT_RATIO)
columns = [
    "AREA", "INT_SQFT", "N_BEDROOM", "N_BATHROOM", 
    "SALE_COND", "PARK_FACIL", "BUILDTYPE", "STREET", 
    "MZZONE", "QS_ROOMS", "QS_BATHROOM", "QS_BEDROOM", 
    "PROPERTY_AGE", "AREA_PRICE_LEVEL", 
    "BEDROOM_SQFT_RATIO", "BATHROOM_SQFT_RATIO"
]

# Define categorical options (cleaned lists)
area_options = ['Adyar', 'Anna Nagar', 'Chrompet', 'KK Nagar', 'Karapakkam', 'TNagar', 'Velachery']
sale_cond_options = ['Normal', 'AbNormal', 'Partial', 'AdjLand', 'Family']
park_facil_options = ['Yes', 'No']
buildtype_options = ['Commercial', 'House', 'Others']
street_options = ['Paved', 'Gravel', 'NoAccess']

# Create a form for user inputs
with st.form("prediction_form"):
    st.header("Enter Property Details")
    
    # User-friendly inputs
    area = st.selectbox("Area", area_options)
    int_sqft = st.number_input("Interior Sqft", min_value=0, step=1, value=1000)
    n_bedroom = st.number_input("Number of Bedrooms", min_value=0, step=1, value=2)
    n_bathroom = st.number_input("Number of Bathrooms", min_value=0, step=1, value=1)
    sale_cond = st.selectbox("Sale Condition", sale_cond_options)
    park_facil = st.selectbox("Parking Facility", park_facil_options)
    buildtype = st.selectbox("Build Type", buildtype_options)
    street = st.selectbox("Street Type", street_options)
    mzzone = st.selectbox("Zoning Type", ['A', 'C', 'I', 'RH', 'RL', 'RM'])
    property_age = st.number_input("Property Age (years)", min_value=0, step=1, value=5)
    
    # Submit button
    submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Calculate derived features
        bedroom_sqft_ratio = n_bedroom / int_sqft if int_sqft > 0 else 0
        bathroom_sqft_ratio = n_bathroom / int_sqft if int_sqft > 0 else 0
        bedroom_sqft_ratio = min(bedroom_sqft_ratio, 1.0)
        bathroom_sqft_ratio = min(bathroom_sqft_ratio, 1.0)
        
        # Define AREA_PRICE_LEVEL mapping
        area_price_mapping = {
            'Karapakkam': 1.0,
            'Anna Nagar': 15.0,
            'Adyar': 4.0,
            'Velachery': 9.0,
            'Chrompet': 5.0,
            'KK Nagar': 10.0,
            'TNagar': 13.0
        }
        
        # Prepare input data
        try:
            data = {
                "AREA": float(encoders['AREA'].transform([area])[0]),
                "INT_SQFT": float(int_sqft),
                "DIST_MAINROAD": 100.0,  # Placeholder value
                "N_BEDROOM": float(n_bedroom),
                "N_BATHROOM": float(n_bathroom),
                "SALE_COND": float(encoders['SALE_COND'].transform([sale_cond])[0]),
                "PARK_FACIL": float(encoders['PARK_FACIL'].transform([park_facil])[0]),
                "BUILDTYPE": float(encoders['BUILDTYPE'].transform([buildtype])[0]),
                "STREET": float(encoders['STREET'].transform([street])[0]),
                "MZZONE": float(encoders['MZZONE'].transform([mzzone])[0]),
                "QS_ROOMS": 3.5,
                "QS_BATHROOM": 3.5,
                "QS_BEDROOM": 3.5,
                "PROPERTY_AGE": float(property_age),
                "BEDROOM_SQFT_RATIO": float(bedroom_sqft_ratio),
                "BATHROOM_SQFT_RATIO": float(bathroom_sqft_ratio),
                "AREA_PRICE_LEVEL": area_price_mapping.get(area, 1.0)
            }
        except ValueError as e:
            st.error(f"Error encoding input: {e}. Please ensure all inputs match training data categories.")
            st.stop()

        try:
            # Create DataFrame with proper column order
            input_data = pd.DataFrame([data], columns=columns)
            
            # Debugging: Show input data
            with st.expander("View Input Data"):
                st.dataframe(input_data)
            
            # Make prediction
            prediction = model.predict(input_data)
            raw_prediction = prediction[0]
            
            # Check if prediction is reasonable
            if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                st.error("Model returned invalid prediction. Please check input values.")
                st.write(f"Raw prediction: {raw_prediction}")
            else:
                predicted_price = float(np.exp(raw_prediction))
                if predicted_price > 100000000:  # 10 crores
                    st.warning("Predicted price seems high. Please verify input values.")
                st.success(f"Predicted Price: {format_indian_number(predicted_price)}")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Debug Info:")
            st.write(f"Input data shape: {input_data.shape}")
            st.write(f"Input columns: {list(input_data.columns)}")
            if hasattr(model, 'feature_names_in_'):
                st.write(f"Model expects: {list(model.feature_names_in_)}")
