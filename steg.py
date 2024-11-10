from model import DataGenerator, give_color_to_seg_img, UNet, val_gen
import cv2
import matplotlib.pyplot as plt
import numpy as np


model = UNet()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
# Load the weights
model.load_weights("seg_model.hdf5")

# Validate
max_show = 1
imgs, segs = next(val_gen)
pred = model.predict(imgs)

from PIL import Image
import cv2
from stegano import lsb
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

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

# Define the target label for embedding
target_label = 3 # Specify the segment label you want to use for embedding

# Define the secret message
secret_message = "Hi this is Amarnath Bhat, this is a test message for steganography. Please ensure saftey of the data."

# Process images and embed messages in specific segments
for i in range(max_show):
    # Get the segmentation mask for the target segment in this image
    target_segment = segs[i, ..., target_label]  # Extract the target label mask for image i

    # Convert to binary mask (if not already binary)
    target_segment = (target_segment > 0).astype(int)

    # Generate color-coded segmentation images for visualization
    _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
    _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))

    # Overlay segmentation on original image
    predimg = cv2.addWeighted(imgs[i] / 255, 0.5, _p, 0.5, 0)
    trueimg = cv2.addWeighted(imgs[i] / 255, 0.5, _s, 0.5, 0)

    # Display and save prediction and ground truth images
    plt.figure(figsize=(6, 6))
    plt.subplot(121)
    plt.title("Prediction")
    plt.imshow(predimg)
    pred_filename = f"output/pred/pred_{i}.png"
    plt.savefig(pred_filename, dpi=150)
    # plt.axis("off")
    # plt.subplot(122)
    # plt.title("Original")
    # plt.imshow(trueimg)
    # plt.axis("off")
    # plt.tight_layout()
    
    plt.show()
    img = Image.open(pred_filename)
    
    # Use the custom generator for steganography
    custom_generator = segment_generator(target_segment)
    
    # Embed the secret message into the image in target segment pixels
    secret_image = lsb.hide(img, secret_message, generator=custom_generator)

    # Save the modified image with the embedded message
    output_filename = f"output/steg/output_image_with_secret_{i}.png"
    secret_image.save(output_filename)

    # Create a new generator instance for reveal
    custom_generator_reveal = segment_generator(target_segment)
    
    # Retrieve the secret message from the modified image
    retrieved_message = lsb.reveal(output_filename, generator=custom_generator_reveal)
    print(f"Retrieved Message from Image {i}: {retrieved_message}")

print("Embedding and retrieval completed.")
