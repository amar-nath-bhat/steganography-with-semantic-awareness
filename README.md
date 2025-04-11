# Segmentation-Based Steganography with U-Net

This project combines semantic segmentation and steganography to hide and retrieve information from images. A U-Net model is used for semantic segmentation, while hidden messages are embedded and retrieved using the segmentation mask as a guide.

---

## 🧠 Features

- ✅ Semantic segmentation using U-Net
- 🔒 Data hiding (steganography) based on segmented features
- 🧪 Metric evaluation (IoU, Dice, Accuracy, F1, Precision, Recall)
- 📊 Visual analysis of predictions
- 🔁 Custom data generator and preprocessing pipeline

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main packages used:

- `tensorflow`
- `numpy`
- `opencv-python`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `Pillow`
- `stegano`

---

## 📈 Evaluation Metrics

Metrics computed:

- **IoU (Jaccard Index)**
- **Dice Coefficient**
- **Pixel Accuracy**
- **Mean Class Accuracy**
- **Precision, Recall, F1 Score**

---

## 📸 Visualization

Saved to `output/masked/masked_image.png`:

- Original image
- Segmentation mask (color-coded)
- Blended masked output

---

## 🛠️ Customization

You can easily adapt:

- Number of segmentation classes (default `13`)
- Image dimensions (`256x256`)
- Batch size & epochs in `main.py`
