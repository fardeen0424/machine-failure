import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objs as go

# Explicitly import scikit-learn components
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_models():
    """
    Load saved models with comprehensive error handling
    """
    try:
        # Attempt to load models
        lr_model_path = 'logistic_regression_model.pkl'
        nn_model_path = 'neural_network_model.h5'
        scaler_path = 'data_scaler.pkl'
        
        # Import joblib here to avoid early import issues
        import joblib
        
        # Load models
        lr_model = joblib.load(lr_model_path)
        nn_model = tf.keras.models.load_model(nn_model_path)
        scaler = joblib.load(scaler_path)
        
        return lr_model, nn_model, scaler
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure all model files are present and correct.")
        return None, None, None

def predict_machine_failure(input_data, lr_model, nn_model, scaler):
    """
    Make predictions using both models
    """
    # Validate input
    if lr_model is None or nn_model is None or scaler is None:
        st.error("Models not loaded properly")
        return None
    
    try:
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Logistic Regression Prediction
        lr_pred_prob = lr_model.predict_proba(scaled_data)[:, 1]
        lr_pred = (lr_pred_prob > 0.5).astype(int)[0]
        
        # Neural Network Prediction
        nn_pred_prob = nn_model.predict(scaled_data)[0][0]
        nn_pred = (nn_pred_prob > 0.5).astype(int)
        
        return {
            'lr_prob': float(lr_pred_prob[0]),
            'lr_pred': int(lr_pred),
            'nn_prob': float(nn_pred_prob),
            'nn_pred': int(nn_pred)
        }
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.title("Machine Failure Prediction System By Sameer NB")
    
    # Load models
    lr_model, nn_model, scaler = load_models()
    
    if lr_model is None or nn_model is None or scaler is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Sidebar for input features
    st.sidebar.header("Machine Sensor Inputs")
    
    # Define feature input configurations
    feature_configs = [
        # Continuous Features
        ('UDI', 0, 10000, 5000, 100),  # Added UDI
        ('Air temperature [K]', 250.0, 350.0, 300.0, 1.0),
        ('Process temperature [K]', 250.0, 350.0, 300.0, 1.0),
        ('Rotational speed [rpm]', 1000, 3000, 1500, 50),
        ('Torque [Nm]', 0.0, 200.0, 50.0, 1.0),
        ('Tool wear [min]', 0, 300, 50, 1)
    ]
    
    # Additional binary features
    binary_features = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # Create input sliders for continuous features
    feature_inputs = {}
    for name, min_val, max_val, default_val, step in feature_configs:
        feature_inputs[name] = st.sidebar.slider(
            name, 
            min_value=min_val, 
            max_value=max_val, 
            value=default_val, 
            step=step
        )
    
    # Create input toggles for binary features
    for feature in binary_features:
        feature_inputs[feature] = st.sidebar.checkbox(feature, value=False)
    
    # Prepare input data
    input_df = pd.DataFrame([feature_inputs])
    
    # Ensure input data matches original training data column order
    original_columns = [
        'UDI',  # Added UDI
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]', 
        'Torque [Nm]', 
        'Tool wear [min]',
        'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
    ]
    
    # Reorder and select columns to match original training data
    input_df = input_df[original_columns]
    
    # Prediction button
    if st.sidebar.button("Predict Machine Failure"):
        # Get predictions
        predictions = predict_machine_failure(input_df, lr_model, nn_model, scaler)
        
        if predictions:
            # Display prediction results
            st.header("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Logistic Regression")
                st.metric(
                    "Prediction", 
                    "Failure" if predictions['lr_pred'] == 1 else "No Failure",
                    f"Probability: {predictions['lr_prob']:.2%}"
                )
            
            with col2:
                st.subheader("Neural Network")
                st.metric(
                    "Prediction", 
                    "Failure" if predictions['nn_pred'] == 1 else "No Failure",
                    f"Probability: {predictions['nn_prob']:.2%}"
                )
            
            # Probability Comparison Bar Chart
            prob_data = {
                'Model': ['Logistic Regression', 'Neural Network'],
                'Failure Probability': [predictions['lr_prob'], predictions['nn_prob']]
            }
            prob_df = pd.DataFrame(prob_data)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Model'], 
                    y=prob_df['Failure Probability'],
                    text=[f"{p:.2%}" for p in prob_df['Failure Probability']],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title='Model Prediction Probabilities',
                yaxis_title='Failure Probability',
                height=400
            )
            
            st.plotly_chart(fig)
    
    # Feature Explanation
    st.sidebar.markdown("### Feature Explanations")
    feature_explanations = {
        'UDI': 'Unique Device Identifier',
        'Air temperature [K]': 'Ambient Air Temperature',
        'Process temperature [K]': 'Machine Process Temperature',
        'Rotational speed [rpm]': 'Machine Rotational Speed',
        'Torque [Nm]': 'Rotational Force Measurement',
        'Tool wear [min]': 'Tool Wear Duration',
        'TWF': 'Tool Wear Failure',
        'HDF': 'Heat Dissipation Failure',
        'PWF': 'Power Failure',
        'OSF': 'Overspeed Failure',
        'RNF': 'Random Failure'
    }
    
    for feature, explanation in feature_explanations.items():
        st.sidebar.text(f"{feature}: {explanation}")
    
    # Additional information
    st.sidebar.info(
        "This app predicts machine failure using Logistic Regression "
        "and Neural Network models based on sensor data."
    )

# Run the app
if __name__ == "__main__":
    main()