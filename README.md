# 🛡️😃 Security Evaluation of a Face Recognition System

## Project Overview

This project investigates the vulnerability of face recognition models to adversarial examples by analyzing their classification performance under controlled perturbations and evaluating potential countermeasures. A dedicated test set was created by selecting 100 identities from the VGGFace2 dataset, with each identity represented by ten images, and experiments were performed using the Inception ResNet V1 model pre-trained on this dataset. After establishing baseline performance on clean inputs, various adversarial attacks were applied to assess their impact on accuracy. To evaluate the transferability of these adversarial examples, a second model based on the ResNet50 architecture was tested with inputs crafted to deceive the first model. Finally, several defense mechanisms were implemented and tested to measure their effectiveness in mitigating the performance degradation caused by these attacks.

> **For methodology, results, and discussions, see [AI4C-report-gr04.pdf](./AI4C-report-gr04.pdf)**

---

## 📁 Project Structure
```plaintext
├── AI4C-report-gr04.pdf # Full project report
├──📁NN2/
│ ├──📁datasets/
│ ├──📁models/
│ ├── demo.py
│ ├── extractor.py
│ ├── LICENSE
│ ├── README.md
│ ├── resnet50_scratch_weight.pkl
│ ├── trainer.py
│ ├── utils.py
├──📁dataset/
│ ├──📁testset/
│ │ └── test_set.csv
│ ├──📁trainset/
│ │ ├──📁adversarial_samples/
│ │ ├──📁clean_samples/
│ ├──📁vggface2_train/
│ │ ├──📁train/
│ ├── identity_meta.csv
│ ├── rcmail_vggface_labels_v2.npy
├── 1)model_baseline_eval_nn1.ipynb # Baseline evaluation (Inception ResNet V1)
├── 2)adversarial_attacks_nn1.ipynb # Adversarial attacks (FGSM, BIM, PGD, DeepFool, CW)
├── 3)attack_transferability_nn2.ipynb # Attack transferability analysis (NN1→NN2)
├── 4)preprocessing_defense_nn1.ipynb # Defence: adversarial training & preprocessing
├── 5)detector_defense_nn1.ipynb # Defence: adversarial sample detection
├── utils.py
├── requirements.txt
├── detector_model.pth
├── detector_params.pkl
├── y_pred_nn1.pt
├── y_pred_nn2.pt
├── y_true.pt
└── README.md
```
---

## 🧪 Main Notebooks & Workflow

### 1️⃣ Baseline Evaluation - `1)model_baseline_eval_nn1.ipynb`
The notebook lays the foundation for the project by establishing a rigorous baseline for face recognition performance on a challenging, balanced test set. The process begins with the setup and careful selection of 100 identities from the **VGGFace2** dataset, ensuring a perfect split between male and female subjects. For each identity, ten representative images are chosen, creating a test set of 1000 unique faces. This selection, supported by metadata such as gender and name, guarantees not only diversity but also fairness in the evaluation.

Images are not simply used in their raw form. Instead, they are aligned and cropped using **MTCNN**, which detects facial landmarks and produces standardised 160x160 face crops. This alignment step is essential to reduce the influence of pose, illumination, and expression, making the recognition task more challenging yet more consistent.

With the test set ready, the notebook then loads the **Inception ResNet V1** model (referred to as NN1), pre-trained on VGGFace2. This model is set to evaluation mode and used to extract 512-dimensional embeddings for each aligned face. Rather than directly using the full neural network for end-to-end classification, a linear classification layer is attached to the embeddings, enabling closed-set identification across the selected identities. The classifier is trained and tested entirely on these extracted features.

Evaluation focuses on top-1 accuracy, measuring how often the predicted identity matches the ground truth. In the reported experiments, the model achieves outstanding results, correctly classifying 986 out of 1000 test images (98.6% accuracy). The notebook does not stop at a simple metric: it saves both the true and predicted labels, enabling in-depth analysis of misclassifications. These rare errors are spread across different identities and are visualised to help understand whether they are due to visual similarities, data quality, or other factors.

### 2️⃣ Adversarial Attack Generation - `2)adversarial_attacks_nn1.ipynb`
The notebook explores the robustness of the baseline face recognition model (NN1, Inception ResNet V1) under adversarial attack scenarios. Building on the previously established baseline accuracy, this notebook systematically generates and evaluates adversarial examples using a variety of state-of-the-art attack algorithms, each designed to test the model’s susceptibility from different perspectives.

After loading the pre-aligned test set and ground-truth labels, the notebook wraps the NN1 model within the **Adversarial Robustness Toolbox (ART)**, enabling it to interface seamlessly with a suite of adversarial attack implementations. This set-up supports both **error-generic (untargeted)** and **error-specific (targeted)** attacks, allowing for a thorough and controlled exploration of model vulnerabilities.

The core of the analysis consists of generating adversarial examples with the following methods: **Fast Gradient Sign Method (FGSM)**, **Basic Iterative Method (BIM)**, **Projected Gradient Descent (PGD)**, **DeepFool**, and **Carlini & Wagner (CW) L∞**. For each attack, the notebook explores how varying the attack parameters—such as the maximum perturbation (epsilon), the step size, the number of iterations, and the number of random initialisations—impacts the model's ability to correctly classify faces. Both generic and targeted variants are tested where supported: in the generic setting, the adversary aims for any misclassification, while in the targeted setting, the attack is crafted to impersonate a specific identity.

For each experiment, adversarial samples are generated and saved, and the resulting predictions are compared against the original ground-truth. The model’s drop in accuracy is quantitatively tracked, and, in the case of targeted attacks, the success rate of forced impersonations is also computed. Results are visualised through **security evaluation curves (SECs)**, which map **model accuracy** (or **targeted success**) as a function of attack strength or configuration. This provides a clear and intuitive sense of how quickly performance deteriorates under increasingly powerful adversarial perturbations.

The notebook also includes detailed visualisations that juxtapose clean and adversarial images, highlighting how even imperceptible perturbations can dramatically alter the model’s predictions. Across all methods, findings are reported for a wide range of parameters, enabling a nuanced comparison of which attacks are most damaging and under what conditions.

### 3️⃣ Attack Transferability - `3)attack_transferability_nn2.ipynb`
The notebook explores a phenomenon in adversarial machine learning known as **transferability**—that is, the ability of adversarial examples crafted to deceive one model to also fool different, independently trained models. This property poses a significant threat in real-world scenarios, as it enables attackers to compromise systems without having access to the precise internal details of the deployed model.

The analysis begins by reusing the adversarial examples previously generated for NN1 (InceptionResNetV1), which were crafted using several popular attack algorithms (FGSM, BIM, PGD, and Carlini & Wagner L∞), both in generic and targeted modes. Instead of generating new perturbations, the notebook adapts these “attacker-crafted” images for evaluation on an alternative model—NN2, which is based on the ResNet-50 architecture. This setup simulates a **grey-box** scenario, where the attacker only needs access to a surrogate model to mount an effective attack.

The process starts by loading the test dataset and aligning preprocessing to ensure compatibility between the adversarial images (originally shaped for NN1) and the input requirements of NN2. This includes resizing, normalization, and other transformations needed for the new architecture.

For each attack method, the notebook evaluates both the **Error Generic** (untargeted) and **Error Specific** (targeted) settings:
- **Error Generic**: The adversarial examples are designed simply to cause any misclassification. The evaluation measures the drop in accuracy of NN2 when presented with these transferred perturbations, compared to clean accuracy.
- **Error Specific**: Here, the attacks are constructed to force the model into predicting a specific target identity (e.g., impersonating “Fernando_Torres”). In addition to accuracy, the targeted success rate—i.e., the percentage of adversarial inputs that the model misclassifies as the attacker’s chosen identity—is computed.

To provide a thorough and nuanced view of transferability, each attack is tested across a range of parameters (such as the strength of perturbation, step size, number of iterations, and random initializations). The notebook visualizes results using **Security Evaluation Curves (SEC)**, showing how NN2’s performance degrades as the attacks generated on NN1 become stronger. For targeted attacks, the SECs include both the overall accuracy and the targeted success rate, offering insight into how easily NN2 can be manipulated towards a specific output.

Visualizations are provided for selected individuals, allowing for a side-by-side comparison of clean and adversarial predictions, further illustrating the real impact of the attacks.

### 4️⃣ Defence: Adversarial Training & Preprocessing - `4)preprocessing_defense_nn1.ipynb`
The notebook implements a **pre-processing defense strategy** for NN1 (InceptionResnetV1), enhancing its robustness against adversarial attacks through proactive input transformations.

The analysis begins by setting up the **test dataset** (100 identities, 1000 images) and initializing the **NN1 model**, which leverages the InceptionResnetV1 architecture pre-trained on VGGFace2. **GPU acceleration** is enabled if available to improve performance.

Key pre-processing techniques include:
- **JPEG Compression**: Compresses the input to reduce fine-grained adversarial noise.
- **Feature Squeezing**: Lowers the bit-depth to limit the attack surface.
- **Spatial Smoothing**: Applies a median filter to suppress localized perturbations.

These techniques are integrated in the helper function `preprocess_adversarial_inputs`, ensuring consistent input normalization and transformation. The pre-processed samples are then evaluated by the original NN1 model without requiring retraining or architectural changes.

The notebook evaluates the defense in two primary scenarios:
- **Clean Data**: Verifying that the defense does not significantly degrade NN1’s clean accuracy. Results confirm a slight drop, from 98.6% to approximately 98.3%.
- **Adversarial Attacks**: Testing the defense against adversarial samples previously generated for NN1 (FGSM, BIM, PGD, Carlini & Wagner L∞) in both generic and targeted settings.

For each attack type:
- **Error Generic**: The defense significantly improves robustness, particularly for iterative attacks (BIM, PGD, CW). Security Evaluation Curves (SEC) show that accuracy degrades more gradually with the defense enabled.
- **Error Specific**: The targeted success rate is substantially reduced, especially for iterative attacks, while overall accuracy is maintained at high levels.

Visualization includes side-by-side comparisons of clean, adversarial, and pre-processed images to illustrate how the defense mitigates perturbations.

These findings are consistent with the report’s data (Tables 7, 8, and 9), highlighting the effectiveness of the pre-processing pipeline as a proactive defense mechanism. However, the report also notes that purely input-level defenses have limitations against simpler attacks like FGSM at high ε, reinforcing the need for complementary strategies like **adversarial sample detection**.

### 5️⃣ Defence: Adversarial Sample Detection - `5)detector_training_nn1.ipynb`
This notebook implements an advanced **detector-based defense strategy** for NN1 (InceptionResnetV1), focusing on explicitly identifying adversarial samples rather than just improving model robustness.

The process begins by constructing a **balanced detector dataset**:
- **Clean samples**: Extracted from the VGGFace2 training set, ensuring no overlap with the test identities.
- **Adversarial samples**: Generated using FGSM, BIM, and PGD attacks in both generic (untargeted) and targeted scenarios, reflecting real-world adversarial challenges.

The adversarial and clean samples are preprocessed using **MTCNN alignment** to ensure consistency with the face recognition pipeline. These aligned datasets are stored and later loaded for training.

For the detector model, a **ResNet-50 architecture** is adapted for binary classification (clean vs adversarial). Only the final layers are fine-tuned to maximize efficiency. The model uses standard data normalization (mean and standard deviation of ImageNet) and is trained on the adversarial dataset with diverse attack intensities and parameters (ε, step size, iterations, random initializations), ensuring broad coverage of the attack landscape.

During training:
- **Batch size**: 32
- **Epochs**: 30
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam, with learning rate 0.01

Post-training, the detector’s weights and configuration are saved (`detector_model.pth` and `detector_params.pkl`).

For evaluation, the detector’s ability to distinguish adversarial samples from clean data is tested:
- **Clean Test Data**: Detector correctly flags very few images as adversarial (low false positives), maintaining trust in clean predictions.
- **Adversarial Test Data**: Detector accurately identifies a significant proportion of adversarial examples, demonstrating effective detection.

The detector’s performance is further analyzed using **Security Evaluation Curves (SECs)**:
- **FGSM**: Detection improves as ε increases, but targeted attacks remain more challenging.
- **BIM and PGD**: Detector robustly identifies iterative attacks, even at moderate perturbation levels.
- **Targeted vs Untargeted**: Detector’s effectiveness is consistent across both error generic and error specific scenarios.

These results align with the report’s findings (Table 10 and Figures 13, 14, 15), confirming that explicit adversarial detection **complements input-level defenses** and enhances overall system security. However, the report also highlights that targeted attacks at low ε still pose challenges, reinforcing the need for further hybrid or ensemble defenses.

This approach represents a significant step forward in **practical adversarial robustness**, addressing the limitations of purely pre-processing defenses and enabling explicit rejection of malicious inputs.

---

## 🚀 Setup & Installation

### 1. Requirements

- Python 3.9+
- facenet-pytorch
- torch
- numpy
- matplotlib
- adversarial-robustness-toolbox

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

The project requires the [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), which is **not included** in this repository due to licensing and size constraints.

- **Download the VGGFace2 dataset** (after registration) from the [official website](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).
- **After downloading, place the entire content of the VGGFace2 "train" split inside:** 📁dataset/vggface2/train/

---

## 📚 References

- [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [Facenet-PyTorch](https://github.com/timesler/facenet-pytorch)
- [VGGFace2-pytorch (NN2)](https://github.com/cydonia999/VGGFace2-pytorch)
- For a full bibliography and additional readings, please see the [AI4C-report-gr04.pdf](./AI4C-report-gr04.pdf).

---

## 👨‍💻 Authors

- **Agostino Cardamone** — [a.cardamone7@studenti.unisa.it](mailto:a.cardamone7@studenti.unisa.it)
- **Chiara Ferraioli** — [c.ferraioli30@studenti.unisa.it](mailto:c.ferraioli30@studenti.unisa.it)
- **Asja Antonucci** — [a.antonucci5@studenti.unisa.it](mailto:a.antonucci5@studenti.unisa.it)
