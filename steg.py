from utils import give_color_to_seg_img
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stegano import lsb
import os

def visualize_and_save(imgs, pred, segs, i, output_dir="output"):
    """
    Visualize and save prediction and ground truth overlays.
    """
    os.makedirs(f"{output_dir}/pred", exist_ok=True)
    os.makedirs(f"{output_dir}/overlay", exist_ok=True)

    _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
    _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))
    
    predimg = cv2.addWeighted(imgs[i] / 255, 0.5, _p, 0.5, 0)
    trueimg = cv2.addWeighted(imgs[i] / 255, 0.5, _s, 0.5, 0)
    
    plt.figure(figsize=(6, 6))
    plt.subplot(121)
    plt.title("Prediction")
    plt.imshow(predimg)
    plt.axis("off")
    plt.savefig(f"{output_dir}/pred/pred_{i}.png", dpi=150)
    plt.close()

def highlight_target_segment(img_rgb, target_segment, i, output_dir="output"):
    """
    Highlight and save the target segment over the original image.
    """
    os.makedirs(f"{output_dir}/overlay", exist_ok=True)
    
    target_segment_colored = img_rgb.copy()
    target_segment_colored[target_segment == 1] = [255, 0, 0]
    overlay_img = cv2.addWeighted(img_rgb, 0.7, target_segment_colored, 0.3, 0)
    
    plt.imshow(overlay_img)
    plt.title("Image with Target Segment")
    plt.axis("off")
    plt.savefig(f"{output_dir}/overlay/overlay_{i}.png", dpi=150)
    plt.close()

def embed_secret(img_rgb, target_segment, secret_message, i, segment_generator, output_dir="output"):
    """
    Embed the secret message into the image using LSB steganography.
    """
    os.makedirs(f"{output_dir}/steg", exist_ok=True)

    img_pil = Image.fromarray(img_rgb)
    custom_generator = segment_generator(target_segment)
    secret_image = lsb.hide(img_pil, secret_message, generator=custom_generator)

    output_path = f"{output_dir}/steg/output_image_with_secret_{i}.png"
    secret_image.save(output_path)
    return output_path

def retrieve_secret(img_path, target_segment, segment_generator):
    """
    Retrieve the hidden message from the image using the same segment-based generator.
    """
    custom_generator_reveal = segment_generator(target_segment)
    retrieved_message = lsb.reveal(img_path, generator=custom_generator_reveal)
    return retrieved_message

def compute_accuracy(original_msg, retrieved_msg):
    """
    Compute accuracy between the original and retrieved message.
    """
    if not retrieved_msg:
        return 0.0
    correct = sum(1 for x, y in zip(original_msg, retrieved_msg) if x == y)
    return correct / len(original_msg)

# def process_images(imgs, segs, pred, secret_message, target_label, segment_generator, output_dir="output"):
#     """
#     Main loop to process a batch of images.
#     """
#     max_show = len(imgs)

#     for i in range(max_show):
#         target_segment = segs[i, ..., target_label]
#         target_segment = (target_segment > 0).astype(int)

#         visualize_and_save(imgs, pred, segs, i, output_dir)

#         img_bgr = imgs[i]
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         highlight_target_segment(img_rgb, target_segment, i, output_dir)

#         steg_img_path = embed_secret(img_rgb, target_segment, secret_message, i, segment_generator, output_dir)
#         retrieved_message = retrieve_secret(steg_img_path, target_segment, segment_generator)

#         accuracy = compute_accuracy(secret_message, retrieved_message)
#         print(f"[Image {i}] Retrieved Accuracy: {accuracy*100:.2f}%")

#     print("Embedding and retrieval completed.")

# Usage
# process_images(imgs, segs, pred, secret_message, target_label, segment_generator)
