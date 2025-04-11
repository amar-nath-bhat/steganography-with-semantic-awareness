import numpy as np 
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from utils import give_color_to_seg_img, DataGenerator, compute_metrics, segment_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from model import UNet
from steg import embed_secret, retrieve_secret

# Parameters
EPOCHS = 20
BATCH_SIZE = 10
HEIGHT = 256
WIDTH = 256
N_CLASSES = 13
MAX_SHOW = 5
TARGET_LABEL = 3
SECRET_MESSAGE = "The quick brown fox jumps over the lazy dog. " * 50

# Paths
TRAIN_FOLDER = "input"
VALID_FOLDER = "input"
os.makedirs("output/masked", exist_ok=True)
os.makedirs("output/pred", exist_ok=True)
os.makedirs("output/overlay", exist_ok=True)
os.makedirs("output/steg", exist_ok=True)

def load_data():
    train_gen = DataGenerator(TRAIN_FOLDER, batch_size=BATCH_SIZE, classes=N_CLASSES, width=WIDTH, height=HEIGHT)
    val_gen = DataGenerator(VALID_FOLDER, batch_size=BATCH_SIZE, classes=N_CLASSES, width=WIDTH, height=HEIGHT)
    return train_gen, val_gen

def visualize_sample_image(img, seg):
    mask = give_color_to_seg_img(np.argmax(seg, axis=-1), n_classes=N_CLASSES)
    masked_image = cv2.addWeighted(img/255, 0.5, mask, 0.5, 0)

    fig, axs = plt.subplots(1, 3, figsize=(20,20))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(mask)
    axs[1].set_title('Segmentation Mask')
    axs[2].imshow(masked_image)
    axs[2].set_title('Masked Image')
    plt.savefig('output/masked/masked_image.png')

def build_and_train_model(train_gen, val_gen):
    model = UNet()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    checkpoint = ModelCheckpoint('seg_model.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    num_train = len(os.listdir(TRAIN_FOLDER))
    num_val = len(os.listdir(VALID_FOLDER))
    train_steps = num_train // BATCH_SIZE + 1
    val_steps = num_val // BATCH_SIZE + 1

    # model.fit(train_gen, validation_data=val_gen, steps_per_epoch=train_steps,
    #           validation_steps=val_steps, epochs=EPOCHS, callbacks=[checkpoint])
    return model

def evaluate_model(model, val_gen):
    imgs, segs = next(val_gen)
    preds = model.predict(imgs)

    metrics_list = []
    for i in range(len(imgs)):
        y_true = np.argmax(segs[i], axis=-1)
        y_pred = np.argmax(preds[i], axis=-1)
        metrics = compute_metrics(y_true, y_pred)
        metrics_list.append(metrics)

    average_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0].keys()}
    metrics_df = pd.DataFrame([average_metrics], index=["Model"])
    print("\nSegmentation Evaluation Metrics:")
    print(metrics_df)

    return imgs, segs, preds

def perform_steganography(imgs, segs, preds):
    accuracies = []

    for i in range(MAX_SHOW):
        target_segment = segs[i, ..., TARGET_LABEL]
        target_segment = (target_segment > 0).astype(int)

        output_filename = embed_secret(
            img_rgb=imgs[i],
            target_segment=target_segment,
            secret_message=SECRET_MESSAGE,
            i=i,
            segment_generator=segment_generator
        )

        retrieved_message = retrieve_secret(
            img_path=output_filename,
            target_segment=target_segment,
            segment_generator=segment_generator
        )

        match_count = sum([1 for x, y in zip(SECRET_MESSAGE, retrieved_message) if x == y])
        accuracy = match_count / len(SECRET_MESSAGE)
        accuracies.append(accuracy)
        print(f"[Image {i}] Accuracy: {accuracy:.4f}")

    print(f"\nAverage Steganography Accuracy: {np.mean(accuracies) * 100:.2f}%")

# Main runner
if __name__ == "__main__":
    train_gen, val_gen = load_data()
    imgs, segs = next(train_gen)
    visualize_sample_image(imgs[0], segs[0])

    print("Training UNet model...")
    model = build_and_train_model(train_gen, val_gen)

    print("Loading best weights...")
    model.load_weights("seg_model.hdf5")

    print("Evaluating model...")
    imgs, segs, preds = evaluate_model(model, val_gen)

    print("Performing steganography on predictions...")
    perform_steganography(imgs, segs, preds)
