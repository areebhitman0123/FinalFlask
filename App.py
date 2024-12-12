import os
from flask import Flask, render_template, request, send_file
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid GUI-related issues
app = Flask(__name__)  # Initialize the Flask application

# Helper function to save plots
def save_plot_image(fig, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_image_path = os.path.join(folder_path, filename)
    fig.savefig(plot_image_path)
    return plot_image_path

# Common function for image processing
def process_image(uploaded_file):
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if test_image is not None:
        test_image = test_image.astype("float") / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        return test_image
    return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            test_image = process_image(uploaded_file)

            if test_image is not None:
                # Load the pre-trained model
                loaded_model = keras.models.load_model('path_to_model.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold predictions to create binary masks
                threshold = 0.5
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmentation
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    contour_length = sum(cv2.arcLength(contour, True) for contour in contours)

                    pix_to_mm = 0.08458333  # Set pixel-to-mm conversion factor
                    length_in_mm = contour_length * pix_to_mm

                    # Plotting the results
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)
                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {length_in_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'femur_plot.png')

                    return render_template('result_femur.html', femur_length=length_in_mm, image_path='static/plot_images/femur_plot.png')

    return render_template('result_femur.html', error='Invalid file or image processing failed')

@app.route('/Femur.html', methods=['GET', 'POST'])
def femur_page():
    return render_template('Femur.html')

if __name__ == '__main__':
    app.run('0.0.0.0')


