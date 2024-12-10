import os
from flask import Flask, render_template, request, send_file
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid GUI-related issues
app = Flask(__name__)  # Initialize the Flask application

# Function to estimate the length of an ellipse perimeter
PIXEL_TO_MM = 0.197889  # Example pixel to mm conversion factor

# Function to estimate the length of an ellipse perimeter in mm
def estimate_ellipse_length(major_axis, minor_axis, pixel_to_mm):
    a = max(major_axis, minor_axis)  # Major axis length in pixels
    b = min(major_axis, minor_axis)  # Minor axis length in pixels
    perimeter_pixels = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))  # Perimeter calculation in pixels
    perimeter_mm = perimeter_pixels * pixel_to_mm  # Convert perimeter to mm
    return perimeter_mm

# Function to draw ellipse on the image
def draw_ellipse(image, mask):
    modified_image = image.copy()  # Create a copy of the image

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate and draw ellipse on the image
    for contour in contours:
        if len(contour) >= 5:  # Ensure there are enough points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)  # Fit an ellipse to the contour
            cv2.ellipse(modified_image, ellipse, (255, 255, 255), 3)  # Draw the ellipse in white color

            major_axis = ellipse[1][0]  # Major axis length in pixels
            minor_axis = ellipse[1][1]  # Minor axis length in pixels
            estimated_length = estimate_ellipse_length(major_axis, minor_axis, PIXEL_TO_MM)  # Estimate the ellipse perimeter in mm

    return modified_image, major_axis, minor_axis, estimated_length

# Function to preprocess the predicted mask
def preprocess_mask(predicted_mask, threshold_value):
    _, binary_mask = cv2.threshold(predicted_mask, threshold_value, 255, cv2.THRESH_BINARY)  # Threshold the mask to make it binary
    return binary_mask

# Function to predict mask using the loaded model
def predict_mask(image, model_path):
    model = keras.models.load_model(model_path)  # Load the trained model
    predicted_mask = model.predict(image)  # Predict the mask for the input image
    return predicted_mask

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

@app.route('/plot_image')
def plot_image():
    return send_file('static/plot_image.png', mimetype='image/png')  # Send the saved plot image

@app.route('/ac', methods=['POST'])
def test_ac():
    if 'file' not in request.files:
        return render_template('result.html', error='No file provided')  # Error if no file is uploaded

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result.html', error='No selected file')  # Error if no file is selected

    # Read the image
    image_stream = uploaded_file.read()  # Read the uploaded file
    nparr = np.frombuffer(image_stream, np.uint8)  # Convert the file to a numpy array
    test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Decode the image from the numpy array

    # Convert to BGR for display
    test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)  # Convert grayscale image to BGR for display

    # Normalize the image
    test_image = test_image.astype("float") / 255.0  # Normalize the image to range [0, 1]
    test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions to match model input shape

    model_path = 'C:/Users/SCOM/Downloads/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/models/AC_MODEL.h5'

    # Predict the mask
    predicted_mask = predict_mask(test_image, model_path)  # Predict the mask using the model

    # Check if the predicted_mask is not None before proceeding
    if predicted_mask is not None:
        # Preprocess the predicted mask
        binary_mask = preprocess_mask(predicted_mask[0], threshold_value=0.5)  # Preprocess the predicted mask

        # Draw ellipse on the image and get measurements
        image_with_ellipse, major_axis, minor_axis, estimated_length = draw_ellipse(test_image_for_display, binary_mask)  # Draw ellipse and get measurements

        # Save the plot as an image
        plot_image_path = save_plot_as_image_ac(image_with_ellipse, binary_mask)  # Save the plot as an image

        # Render result.html with plot and measurements
        return render_template('result.html', plot_image_path=plot_image_path, major_axis=major_axis, minor_axis=minor_axis, estimated_length=estimated_length)  # Display results

    else:
        return render_template('result.html', error='Model prediction failed')  # Error if model prediction fails

# Function to save plot as image AC
def save_plot_as_image_ac(image_with_ellipse, binary_mask):
    # Increase the size of the entire figure
    plt.figure(figsize=(12, 6))  # Set the figure size

    # Original Image with Ellipse
    plt.subplot(1, 2, 1)  # Create a subplot for the original image with ellipse
    plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))  # Display the image with ellipse
    plt.axis('off')  # Turn off the axis
    plt.title("Image with Ellipse")  # Set the title

    # Predicted Mask
    plt.subplot(1, 2, 2)  # Create a subplot for the predicted mask
    plt.imshow(binary_mask, cmap='gray')  # Display the predicted mask
    plt.axis('off')  # Turn off the axis
    plt.title("Predicted Mask")  # Set the title

    # Save the plot as an image
    plot_image_path = 'static/plot_image.png'  # Define the path to save the plot image
    plt.savefig(plot_image_path)  # Save the figure
    plt.close()  # Close the plot

    return plot_image_path  # Return the path to the saved plot image



@app.route('/bpd', methods=['POST'])
def test_bpd():
    if 'file' not in request.files:
        return render_template('result.html', error='No file provided')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result.html', error='No selected file')

    # Read the image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Convert to BGR for display
    test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)

    # Normalize the image
    test_image = test_image.astype("float") / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    model_path = 'C:/Users/SCOM/Downloads/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/models/BDP_MODEL.h5'

    # Predict the mask
    predicted_mask = predict_mask(test_image, model_path)

    # Check if the predicted_mask is not None before proceeding
    if predicted_mask is not None:
        # Preprocess the predicted mask
        binary_mask = preprocess_mask(predicted_mask[0], threshold_value=0.5)

        # Draw ellipse on the image and get measurements
        image_with_ellipse, major_axis, minor_axis, estimated_length = draw_ellipse(test_image_for_display, binary_mask)

        # Continue with the rest of your code
        image_with_line = image_with_ellipse.copy()
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                center_x, center_y = map(int, center)

                major_axis, minor_axis = map(int, axes)

                # Calculate top and bottom points of the major axis
                top_point = (center_x, center_y - major_axis // 2)
                bottom_point = (center_x, center_y + major_axis // 2)

                # Draw a line from top to bottom on the ellipse
                cv2.line(image_with_line, top_point, bottom_point, (255, 0, 0), 2)

                # Calculate the length of the line
                length = np.sqrt((bottom_point[0] - top_point[0]) ** 2 + (bottom_point[1] - top_point[1]) ** 2)  * 0.3169


        # Save the plot as an image
        plot_image_path = save_plot_as_image(image_with_ellipse, binary_mask, image_with_line)

        # Render result.html with plot and measurements
        return render_template('result.html', plot_image_path=plot_image_path, major_axis=major_axis, minor_axis=minor_axis, estimated_length=estimated_length, length = length )

    else:
        return render_template('result.html', error='Model prediction failed')

# Function to save plot as image BPD
def save_plot_as_image(image_with_ellipse, binary_mask, image_with_line):
    # Increase the size of the entire figure
    plt.figure(figsize=(24, 8))

    # Original Image with Ellipse
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image with Ellipse")

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.title("Predicted Mask")

    # Detected Ellipse with Line
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Ellipse with Line')

    # Save the plot as an image
    plot_image_path = 'static/plot_image.png'
    plt.savefig(plot_image_path)
    plt.close()

    return plot_image_path

#FEMUR Starts Here!

def save_plot_image(fig, directory, filename):
    file_path = os.path.join(directory, filename)
    fig.savefig(file_path)
    plt.close(fig)  # Close the plot to release resources
    return file_path


#Voluson-E6

@app.route('/volusonE6', methods=['POST'])
def test_voluson_e6():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()  # Read the image data from the uploaded file
            nparr = np.frombuffer(image_stream, np.uint8)  # Convert the image data to a numpy array
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Decode the image from the numpy array

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)  # Convert the image to BGR for display
                test_image = test_image.astype("float") / 255.0  # Normalize the image to range [0, 1]
                test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions to match model input shape

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/SCOM/Desktop/Fetal-Health-Anaysis-FYP--Revised-2-jasonify-API-/models/vle6_updated.h5')  # Load the trained model

                # Make predictions
                predictions = loaded_model.predict(test_image)  # Predict the mask for the input image

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # Set a threshold value for binary mask
                thresholded_predictions = (predictions > threshold).astype(np.uint8)  # Apply threshold to predictions

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)  # Apply the mask to the image

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)  # Apply the mask to the image

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the mask

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)  # Sum the lengths of the contours

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)  # Create a blank image for contours
                    cv2.drawContours(contour_image, contours, -1, (255), 1)  # Draw contours on the image

                    pix = 0.08458333 # Pixel to mm conversion factor
                    pix_mm = contour_length * pix  # Convert contour length to mm

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Create a subplot with 4 columns

                    axs[0].imshow(image, cmap='gray')  # Show the original image
                    axs[0].set_title('Original Image')  # Set title for the original image
                    axs[0].axis('off')  # Turn off axis for the original image

                    axs[1].imshow(segmented_head, cmap='gray')  # Show the segmented head image
                    axs[1].set_title('Segmented Femur')  # Set title for the segmented head image
                    axs[1].axis('off')  # Turn off axis for the segmented head image

                    axs[2].imshow(contour_image, cmap='gray')  # Show the contour image
                    axs[2].set_title('Contour Area')  # Set title for the contour image
                    axs[2].axis('off')  # Turn off axis for the contour image

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)  # Display the measured length
                    axs[3].axis('off')  # Turn off axis for the length display

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'voluson_e6_plot.png')  # Save the plot image

                    # Render the result page with the specific details
                    return render_template('result_femur.html', machine='test_volusonE6', femur_length=pix_mm, image_path='static/plot_images/voluson_e6_plot.png', plot_path=plot_image_path)  # Display the result page with details

    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')  # Error if no valid file or image processing fails



#Voluson-S10
@app.route('/volusonS10', methods=['POST'])
def test_voluson_S10():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/SCOM/Downloads/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/models/VLS10.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.08458333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'voluson_s10_plot.png')

                    # Render the result page with the specific details
                    # Inside the test_voluson_s10 route
                    machine_name = request.args.get('machine', default='test_volusonS10')
                    return render_template('result_femur.html', machine=machine_name, femur_length=pix_mm,image_path='static/plot_images/voluson_s10_plot.png',plot_path=plot_image_path)



    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')



#Voluson-S8

@app.route('/volusonS8', methods=['POST'])
def test_voluson_s8():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/SCOM/Downloads/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/models/vls8.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.08458333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'voluson_s8_plot.png')

                    # Render the result page with the specific details
                    # Inside the test_voluson_s8 route
                    machine_name = request.args.get('machine', default='test_volusonS8')
                    return render_template('result_femur.html', machine=machine_name, femur_length=pix_mm,image_path='static/plot_images/voluson_s8_plot.png',plot_path=plot_image_path)



    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')


#ALOKA

@app.route('/aloka', methods=['POST'])
def test_aloka():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/SCOM/Downloads/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/Fetal-Health-Anaysis-FYP--Revised-2-NON-JASONIFY-/models/Aloka_model.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.06858333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'aloka_plot.png')

                    # Render the result page with the specific details

                    machine_name = request.args.get('machine', default='test_aloka')
                    return render_template('result_femur.html', machine=machine_name, femur_length=pix_mm,image_path='static/plot_images/aloka_plot.png',plot_path=plot_image_path)



    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')


@app.route('/Femur.html', methods=['GET', 'POST'])
def femur_page():
    machine = request.args.get('machine', 'default_machine_value')
    return render_template('Femur.html', machine=machine)


if __name__ == '__main__':
    app.run(debug=True)
