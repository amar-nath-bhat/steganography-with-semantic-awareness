from model import DataGenerator, give_color_to_seg_img, UNet, val_gen
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stegano import lsb
from itertools import product
from random import randint

# Load the model and weights
model = UNet()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
model.load_weights("seg_model.hdf5")

# Set up variables
max_show = 5
imgs, segs = next(val_gen)
pred = model.predict(imgs)

# Define a custom generator based on segment locations
def segment_generator(segment_mask):
    """
    Generator function to select pixels only in specific segments.
    
    :param segment_mask: 2D numpy array with 1s indicating target segment pixels, 0s elsewhere.
    :return: Generator that yields single integer indices for pixels within the target segments.
    """
    rows, cols = segment_mask.shape
    for x, y in product(range(rows), range(cols)):
        if segment_mask[x, y] == 1:  # Only yield indices within the target segment
            yield x * cols + y  # Convert (x, y) to a single integer index

# Set the target label and secret message

secret_message = "The quick brown fox jumps over the lazy dog. " * 50  # Large message
target_label = 3 # Randomly select a target label

# Embed the secret message in each image
for i in range(max_show):
    # Get the segmentation mask for the target segment
    
    target_segment = segs[i, ..., target_label]  # Extract the target label mask for image i
    target_segment = (target_segment > 0).astype(int)  # Convert to binary mask

    # Generate color-coded segmentation images for visualization
    _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
    _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))
    
    # Overlay segmentation on original image
    predimg = cv2.addWeighted(imgs[i] / 255, 0.5, _p, 0.5, 0)
    trueimg = cv2.addWeighted(imgs[i] / 255, 0.5, _s, 0.5, 0)
    
    # Display the images
    plt.figure(figsize=(6, 6))
    plt.subplot(121)
    plt.title("Prediction")
    plt.imshow(predimg)
    pred_filename = f"output/pred/pred_{i}.png"
    plt.savefig(pred_filename, dpi=150)
    
    img = imgs[i]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL compatibility
    img_pil = Image.fromarray(img_rgb)
    
    # Overlay target segment on original image for visualization
    target_segment_colored = img_rgb.copy()
    target_segment_colored[target_segment == 1] = [255, 0, 0]  # Color the segment in red

    # Blend the colored segment with the original image
    overlay_img = cv2.addWeighted(img_rgb, 0.7, target_segment_colored, 0.3, 0)

    # Display the image with the highlighted segment
    plt.imshow(overlay_img)
    plt.title("Original Image with Target Segment Highlighted")
    plt.axis("off")
    plt.savefig(f"output/overlay/overlay_{i}.png", dpi=150)

    # Use the custom generator for steganography
    custom_generator = segment_generator(target_segment)

    # Embed the secret message into the actual image
    secret_image = lsb.hide(img_pil, secret_message, generator=custom_generator)

    # Save and display the modified image
    output_filename = f"output/steg/output_image_with_secret_{i}.png"
    secret_image.save(output_filename)
    
    # Display the image using OpenCV
    output_image = cv2.imread(output_filename)
    # cv2.imshow("Steganography Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Retrieve the secret message
    custom_generator_reveal = segment_generator(target_segment)
    retrieved_message = lsb.reveal(output_filename, generator=custom_generator_reveal)
    print(f"Retrieved Message from Image {i}: {retrieved_message}")
    
    # compute the accuracy of the retrieved message
    accuracy = sum([1 for x, y in zip(secret_message, retrieved_message) if x == y]) / len(secret_message)
    # print(f"Accuracy of the retrieved message: {accuracy * 100:.2f}%")
    
print("Embedding and retrieval completed.")

import numpy as np
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import pandas as pd

# Assume `imgs` contains the input images, and `segs` contains the ground truth segmentations.
# `pred` should be the output predictions from your model for `imgs`.

# Generate predictions from the model
pred = model.predict(imgs)  # Use your trained model to predict segmentation on imgs

def compute_metrics(y_true, y_pred):
    """
    Compute segmentation metrics.
    
    Args:
    y_true (np.array): Ground truth mask (flattened for each image).
    y_pred (np.array): Predicted mask (flattened for each image).
    
    Returns:
    dict: Dictionary containing IoU, Dice Coefficient, Pixel Accuracy, Mean Accuracy, Precision, Recall, and F1 Score.
    """
    metrics = {}

    # Flatten arrays for per-pixel metric calculations
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute metrics
    metrics['IoU'] = jaccard_score(y_true_flat, y_pred_flat, average='macro')
    metrics['Dice Coefficient'] = f1_score(y_true_flat, y_pred_flat, average='macro')
    metrics['Pixel Accuracy'] = np.mean(y_true_flat == y_pred_flat)
    metrics['Mean Accuracy'] = np.mean([np.mean(y_pred[y_true == cls] == cls) for cls in np.unique(y_true)])
    metrics['Precision'] = precision_score(y_true_flat, y_pred_flat, average='macro')
    metrics['Recall'] = recall_score(y_true_flat, y_pred_flat, average='macro')
    metrics['F1 Score'] = f1_score(y_true_flat, y_pred_flat, average='macro')
    
    return metrics

# Loop over each image in the batch, compute metrics, and average
metrics_list = []
for i in range(len(imgs)):
    # Convert ground truth and prediction to binary labels if needed
    y_true = np.argmax(segs[i], axis=-1)  # Convert to single channel with class labels
    y_pred = np.argmax(pred[i], axis=-1)  # Convert to single channel with class labels
    
    # Compute metrics for the current pair
    metrics = compute_metrics(y_true, y_pred)
    metrics_list.append(metrics)

# Average metrics across all images in the batch
average_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0].keys()}

# Display metrics in a DataFrame
metrics_df = pd.DataFrame([average_metrics], index=["Model"])
print(metrics_df)
