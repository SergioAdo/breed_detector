# app.py

import streamlit as st
from fastai.vision.all import load_learner, PILImage
import pandas as pd
import os

# Load the trained model with caching to prevent reloading on every interaction
@st.cache_resource
def load_model():
    return load_learner('dog_breed_classifier.pkl')

learn = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the classification mode:",
                            ["Single Image Classification", "Multiple Images Classification"])

def single_image_classification():
    st.title("🖼️ Single Dog Breed Classifier")
    st.write("Upload an image of a dog, and the model will classify its breed along with the top 3 predictions.")

    # File uploader for a single file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the image
            img = PILImage.create(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)

            # Make prediction
            pred, pred_idx, probs = learn.predict(img)
            breed = str(pred)
            confidence = probs[pred_idx] * 100

            # Extract top 3 predictions
            top_n = 3
            top_probs, top_idxs = probs.topk(top_n)
            top_preds = [str(learn.dls.vocab[i]) for i in top_idxs]
            top_confidences = [f"{p * 100:.2f}%" for p in top_probs]
            
            if breed == 'Error':
                st.write("Oooops!! That does not look like a dog 😅")
                st.write("Please try another picture")
                
            else:

            # Display results
                st.write(f"**Top Prediction:** {breed} ({confidence:.2f}% confidence)")

            # Display additional predictions
                if top_n > 1 and breed != "Error":
                    additional_preds = ", ".join([f"{p} ({c})" for p, c in zip(top_preds[1:], top_confidences[1:])])
                    st.write(f"**Also Looks Like:** {additional_preds}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

def multiple_images_classification():
    st.title("📂 Multiple Dog Breed Classifier")
    st.write("Upload multiple images of dogs, and the model will classify each breed along with the top 3 predictions.")

    # File uploader allows multiple files
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Initialize lists to store results
        predictions_list = []
        confidence_list = []
        second_preds_list = []
        second_confidences = []
        third_preds_list = []
        third_confidences = []
        image_names = []

        for uploaded_file in uploaded_files:
            try:
                # Display the image
                img = PILImage.create(uploaded_file)
                st.image(img, caption='Uploaded Image.', use_column_width=True)

                # Make prediction
                pred, pred_idx, probs = learn.predict(img)
                breed = str(pred)
                confidence = probs[pred_idx] * 100

                # Extract top 3 predictions
                top_n = 3
                top_probs, top_idxs = probs.topk(top_n)
                top_preds = [str(learn.dls.vocab[i]) for i in top_idxs]
                top_confidences = [p * 100 for p in top_probs]

                # Collect results
                image_names.append(uploaded_file.name)
                predictions_list.append(breed)
                confidence_list.append(f"{confidence:.2f}%")

                # Handle additional predictions
                if top_n > 1:
                    second_pred = top_preds[1]
                    second_conf = f"{top_confidences[1]:.2f}%"
                    third_pred = top_preds[2] if top_n > 2 else "N/A"
                    third_conf = f"{top_confidences[2]:.2f}%" if top_n > 2 else "N/A"

                    second_preds_list.append(second_pred)
                    second_confidences.append(second_conf)
                    third_preds_list.append(third_pred)
                    third_confidences.append(third_conf)
                else:
                    second_preds_list.append("N/A")
                    second_confidences.append("N/A")
                    third_preds_list.append("N/A")
                    third_confidences.append("N/A")

                # Display results
                st.write(f"**Top Prediction:** {breed} ({confidence:.2f}% confidence)")
                additional_preds = ", ".join([
                    f"{second_pred} ({second_conf})",
                    f"{third_pred} ({third_conf})"
                ]) if top_n > 1 else "N/A"
                st.write(f"**Also Looks Like:** {additional_preds}")
                st.markdown("---")  # Separator between images

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        # Display all results in a table
        if image_names:
            st.header("📊 Classification Results")
            results_df = pd.DataFrame({
                'Image': image_names,
                'Top Prediction': predictions_list,
                'Confidence': confidence_list,
                'Second Prediction': second_preds_list,
                'Second Confidence': second_confidences,
                'Third Prediction': third_preds_list,
                'Third Confidence': third_confidences
            })
            st.table(results_df)

            # Download button for CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='classification_results.csv',
                mime='text/csv',
            )

# Main Execution
if app_mode == "Single Image Classification":
    single_image_classification()
elif app_mode == "Multiple Images Classification":
    multiple_images_classification()
