# Multimodal Fake News Detection: Integrating BERT and ResNet50

A research-grade Deep Learning project that targets the detection of complex online misinformation by integrating and cross-referencing textual and visual data streams. 

Rather than relying purely on a text headline or an isolated image, this architecture processes cross-modal data to identify visual cues that either confirm or explicitly contradict textual claims. This provides a significantly higher barrier against deceptive media than standard unimodal baselines.

---

## 🚀 Getting Started & Execution Guide

### ⚠️ Critical Warnings Before Running
* **Execution Time:** Training this model over the full dataset takes a significant amount of time and computational resources. Even running evaluations or smaller samples requires noticeable processing overhead.
* **Hardware Requirements:** It is **highly recommended** to run this project on a machine equipped with an NVIDIA GPU. 
* **Dependencies:** Proper execution requires standard Deep Learning libraries and specialized compute platforms. Crucially, **CUDA** must be installed and configured correctly on your system to leverage GPU acceleration; running this strictly on a CPU will result in extremely slow performance.

### 📋 Prerequisites & Installation
1. Ensure you have Python 3.8+ and the appropriate NVIDIA CUDA Toolkit installed.
2. Install all required libraries and dependencies by running:
   ```bash
   pip install -r requirements.txt

## 📊 Final Experimental Results & Evaluation

The following metrics represent the finalized, peak performance of the deep multi-modal network over a complete **8-epoch training run** on the unseen public test partition of the **Fakeddit dataset**:

* **Classification Accuracy:** `82.834%`
* **Precision:** `80.151%`
* **Recall:** `87.386%`
* **F1-Score:** `83.613%`

**THIS RESULT ABOVE ARE ACHIEVED USING A DATASET OF 106GB, IN THIS REPOSITORY, A SMALLER SAMPLE IS USED, HENCE RESULT MAY DIFFER. KEEP IN MIND THAT THE STATIC RESULT OF THE CODE IS ONLY IN THE EARLY STAGES OF TRAINING, RESULTS MAY DIFFER. **

### 🔑 Technical Breakdown
The standout achievement of this architecture is its high **87.386% Recall**. In trust-and-safety machine learning pipelines, maximizing recall is vital because it minimizes false negatives—ensuring that deceptive or harmful misinformation rarely leaks past the filter undetected.

---

## 🚀 Model Architecture & Fusion Strategy

The network processes text and image modalities simultaneously via separate feature extraction pathways before joining them in a combined classification space:

1. **Text Processing Pipeline (NLP):** Utilizes a pre-trained `bert-base-uncased` transformer model to capture semantic context from headlines, extracting the `[CLS]` token's pooler representation.
2. **Visual Feature Extraction (Computer Vision):** Leverages a pre-trained `ResNet50` backbone (trained on ImageNet-1K), stripped of its final classification layer, leaving a dense 2,048-dimensional representation vector.
3. **Cross-Modal Fusion:** Concatenates the extracted 768-dimensional textual features and 2,048-dimensional visual features into a unified 2,816-dimensional array.
4. **Classification Head:** Passes the combined multi-modal vector through a sequence of dense layers (`Linear(2,816 -> 256)` -> `ReLU` -> `Dropout(0.3)` -> `Linear(256 -> 1)`) finalized by a `Sigmoid` activation function.

---

## 👥 Project Contributions

This project was developed through a collaborative effort, with responsibilities divided across data preparation, machine learning engineering, and architectural design as follows:

* **David Alexander Haryono**
  * Engineered the multi-modal fusion architecture, concatenating the textual and visual embeddings into a unified classification space.
  * Configured the NLP pipeline using the pre-trained `bert-base-uncased` transformer to extract dense textual feature representations.
  * Constructed and fine-tuned the final dense classification head layers, optimizing dropout rates to reduce overfitting.

* **Evan Samuel Widjaja**
  * Developed the computer vision feature extraction pipeline utilizing the pre-trained `ResNet50` backbone.
  * Managed dataset handling, preprocessing, and sampling pipelines for the massive 106GB Fakeddit source data.
  * Configured the end-to-end training loops and executed validation metrics to monitor model accuracy, precision, and recall.

