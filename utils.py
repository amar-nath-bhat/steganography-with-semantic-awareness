import os
import numpy as np
import seaborn as sns
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from itertools import product

# Helper Functions
def LoadImage(name, path):
    img = Image.open(os.path.join(path, name))
    img = np.array(img)
    
    image = img[:,:256]
    mask = img[:,256:]
    
    return image, mask


def bin_image(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask

def getSegmentationArr(image, classes, width, height):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, : , 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c ).astype(int)
    return seg_labels

def give_color_to_seg_img(seg, n_classes):
    
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

def DataGenerator(path, batch_size, classes, width, height):
    files = os.listdir(path)
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i+batch_size]
            imgs=[]
            segs=[]
            for file in batch_files:
                image, mask = LoadImage(file, path)
                mask_binned = bin_image(mask)
                labels = getSegmentationArr(mask_binned, classes, width, height)

                imgs.append(image)
                segs.append(labels)

            yield np.array(imgs), np.array(segs)

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