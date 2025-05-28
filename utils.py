import os
import sys
import shutil
import logging
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Linear, CrossEntropyLoss, Sequential
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Union, List, Optional, Tuple, Dict, Callable
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker

from skimage.transform import resize
from torchvision import models

from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.datasets import ImageFolder
from facenet_pytorch import MTCNN, InceptionResnetV1

from art import config
from art.estimators.classification import PyTorchClassifier
from art.defences.detector.evasion import BinaryInputDetector
from art.utils import to_categorical
from art.defences.preprocessor import FeatureSqueezing, SpatialSmoothing, JpegCompression

from sklearn.metrics import accuracy_score

# --------------------------------------------------------
# Logging setup
# --------------------------------------------------------
import logging
logger = logging.getLogger('AIC_Logger')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
logger.info('Logging is set up.')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def collate_fn(x):
    """
    Custom collate function for DataLoader to return a single element (first item).

    This is used to flatten batches of size 1 to a single sample, which is sometimes
    required for datasets or evaluation loops that expect an individual sample
    rather than a list or tuple.

    Parameters
    ----------
    x : list
        List of items (usually a single-element list from DataLoader).

    Returns
    -------
    object
        The first (and usually only) item from the input list.

    Examples
    --------
    >>> collate_fn([sample])
    sample
    """
    return x[0]

def evaluate_model(model: nn.Module, dataloader: DataLoader, LABELS: np.ndarray) -> Tuple[list, list]:
    """
    Evaluates a PyTorch model on a dataset and returns the true and predicted labels.

    The function switches the model to evaluation mode, disabling dropout and batch statistics updates.
    For each input sample provided by the DataLoader, the model produces a prediction which is then
    mapped from a class index to a human-readable label using the provided LABELS array.

    This function is useful to compare model predictions against ground truth labels, either under
    normal conditions or when evaluating robustness against adversarial examples.

    Returns two parallel lists of strings: the first contains the ground truth labels, the second the
    corresponding predicted labels.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    dataloader : DataLoader
        A DataLoader that provides input images and their corresponding label indices.
    LABELS : np.ndarray
        An array of class names used to map output indices to human-readable labels.

    Returns
    -------
    Tuple[list, list]
        A tuple containing:
        - y_true: list of ground truth class labels (str),
        - y_pred: list of predicted class labels (str).

    Examples
    --------
    >>> y_true, y_pred = evaluate_model(model, test_loader, np.array(['cat', 'dog']))
    >>> print(y_true[0], y_pred[0])
    'dog' 'cat'
    """
    # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
        
    # Initialize the variables
    y_true = []
    y_pred = []
        
    # Iterate over the test set
    with torch.no_grad():
        for sample in tqdm(dataloader, total=len(dataloader), desc='Evaluating the model', leave=False):
            # Get image and label from the sample
            image, label = sample[0].to(device), sample[1].to(device)
            
            # Predicted output from the model
            image = torch.unsqueeze(image, 0) if len(image.shape) == 3 else image
            output = model(image).cpu()
            pred_class = int(torch.argmax(output, dim=1).item())
        
            # Append true and predicted labels to the lists
            y_true.append(dataloader.dataset.idx_to_class[label.item()])
            y_pred.append(LABELS[pred_class])          
            
    return y_true, y_pred

def nn2_transform(img: torch.Tensor) -> torch.Tensor:
    """
    Applies VGGFace2-style preprocessing to a single image tensor.

    The image is first converted from normalized RGB to [0, 1], then to BGR,
    followed by channel-wise mean subtraction. The result is a float tensor
    shaped as [C, H, W], compatible with models trained on VGGFace2.

    Parameters
    ----------
    img : torch.Tensor
        A single image tensor in [-1, 1] or [0, 1] format, shape [C, H, W].

    Returns
    -------
    torch.Tensor
        Preprocessed image ready for NN2 input.

    Examples
    --------
    >>> img_pre = nn2_transform(input_tensor)
    >>> img_pre.shape
    torch.Size([3, 224, 224])
    """
    mean = np.array([91.4953, 103.8827, 131.0912]) 
    
    img = transforms.ToPILImage()((img.clone().detach() + 1.0) / 2.0)
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]                           # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean
    img = img.transpose(2, 0, 1)                    # C x H x W
    img = torch.from_numpy(img).float()
    return img

def nn2_untransform(img : np.ndarray) -> np.ndarray:
    """
    Reverses the NN2 preprocessing to recover a displayable RGB image.

    The image is restored by adding the VGGFace2 mean and converting from BGR to RGB.
    The result is clipped, cast to uint8, and reshaped to [H, W, C] format for visualization.

    Parameters
    ----------
    img : np.ndarray
        Preprocessed image in shape [C, H, W] and float32 format.

    Returns
    -------
    np.ndarray
        Reconstructed image in [H, W, C] format as uint8 RGB.

    Examples
    --------
    >>> restored = nn2_untransform(preprocessed_img)
    >>> plt.imshow(restored)
    """
    mean = np.array([91.4953, 103.8827, 131.0912]) 
    
    img = img.transpose(1, 2, 0)
    img += mean
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img

def conversion_nn1_to_nn2(
    x_adv: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
    """
    Converts and preprocesses a batch of images from NN1 (FaceNet/InceptionResnetV1, 160x160) format
    to NN2 (VGGFace2/ResNet, 224x224) format, suitable for evaluation with models trained on VGGFace2.

    The function performs the following steps:
    - Accepts either a NumPy array or PyTorch tensor of images in shape [N, C, H, W].
    - If input is a NumPy array, converts it to a PyTorch tensor.
    - Resizes each image from 160x160 to 224x224 using bilinear interpolation.
    - Applies VGGFace2-style preprocessing to each image: scaling, RGB to BGR conversion,
      and mean subtraction.

    This is commonly used when transferring adversarial examples or clean images from a model
    trained on FaceNet format (160x160, [-1,1], RGB) to another model trained on VGGFace2 (224x224, BGR, mean-subtracted).

    Parameters
    ----------
    x_adv : np.ndarray or torch.Tensor
        Batch of input images in NN1 format, shape [N, C, 160, 160], values in [-1, 1] or [0, 1].

    Returns
    -------
    torch.Tensor
        Batch of images in NN2 preprocessed format, shape [N, 3, 224, 224], dtype float32.

    Examples
    --------
    >>> # x_adv: shape [N, 3, 160, 160], values in [-1, 1]
    >>> x_nn2 = conversion_nn1_to_nn2(x_adv)
    >>> x_nn2.shape
    torch.Size([N, 3, 224, 224])
    """
    if isinstance(x_adv, np.ndarray):
        x_adv = torch.tensor(x_adv, dtype=torch.float32)

    x_resized = F.interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)

    return torch.stack([nn2_transform(x) for x in x_resized])

def preprocess_adversarial_inputs(
    x_adv: np.ndarray,
    jpeg_quality: int = 50,
    bit_depth: int = 4,
    window_size: int = 11,
    clip_values: Tuple[float, float] = (0, 1),
    channels_first: bool = True,
    labels: Optional[np.ndarray] = None,
    model: Optional[Callable] = None,
    label_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies JPEG compression, Feature Squeezing, and Spatial Smoothing
    to adversarial examples.

    Parameters
    ----------
    x_adv : np.ndarray
        Adversarial examples in range [-1, 1] or [0, 1], shape (N, C, H, W).
    jpeg_quality : int
        JPEG compression quality (default 50).
    bit_depth : int
        Bit depth for Feature Squeezing (default 4).
    window_size : int
        Window size for Spatial Smoothing (default 11).
    clip_values : tuple
        Min/max values for input clipping.
    channels_first : bool
        Whether input has channel-first format (True: [N, C, H, W]).
    labels : np.ndarray, optional
        Labels needed for ART transformations. If None and model is provided, predicted labels will be used.
    model : callable, optional
        A model used to generate predicted labels if labels are not provided.
    label_map : list or array, optional
        Used to convert model outputs to label strings.

    Returns
    -------
    x_processed : np.ndarray
        Processed adversarial examples, in range [-1, 1].
    """
    # Normalize if needed
    if np.min(x_adv) < 0:
        x_norm = (x_adv + 1.0) / 2.0
    else:
        x_norm = x_adv

    # Apply JPEG
    jpeg = JpegCompression(clip_values=clip_values, quality=jpeg_quality, channels_first=channels_first)
    x_compressed, _ = jpeg(x_norm)

    # Estimate labels if not provided
    if labels is None:
        if model is None or label_map is None:
            raise ValueError("Either 'labels' or both 'model' and 'label_map' must be provided.")
        y_pred_logits = model.predict(x_compressed)
        labels = label_map[np.argmax(y_pred_logits, axis=1)]

    # Apply Feature Squeezing
    fs = FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth)
    x_squeezed, _ = fs(x_compressed, labels)

    # Apply Spatial Smoothing
    ss = SpatialSmoothing(clip_values=clip_values, window_size=window_size, channels_first=channels_first)
    x_smoothed, _ = ss(x_squeezed, labels)

    # Restore to [-1, 1]
    x_final = x_smoothed * 2.0 - 1.0

    return x_final

def conversion_nn1_to_detector(
    x_array: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32
    ):
    """
    Converts and preprocesses a batch of images from NN1 (FaceNet/InceptionResnetV1) format
    to the input format required by a detector network (typically ImageNet-pretrained ResNet or similar).

    The function performs the following operations:
    - Accepts either a NumPy array or PyTorch tensor of images in shape [N, C, H, W] with values in [-1, 1].
    - Converts input to a PyTorch tensor if needed.
    - Resizes images from 160x160 to 224x224 via bilinear interpolation.
    - Scales images from [-1, 1] to [0, 1].
    - Converts images from [N, C, H, W] to [N, H, W, C] (channels-last) for normalization.
    - Normalizes each channel using ImageNet mean and standard deviation (applied to values in [0,1]).
    - Restores images to [N, C, H, W] format (channels-first), type float32.

    This is typically used to transfer images (adversarial or clean) from a FaceNet pipeline to a standard detector
    that expects ImageNet normalization and input shape.

    Parameters
    ----------
    x_array : np.ndarray or torch.Tensor
        Batch of input images, shape [N, 3, 160, 160], values in [-1, 1].
    batch_size : int, optional
        Batch size for preprocessing (default: 32).

    Returns
    -------
    np.ndarray
        Batch of processed images, shape [N, 3, 224, 224], dtype float32, normalized as required by typical detectors.

    Examples
    --------
    >>> # x_array: [N, 3, 160, 160], values in [-1, 1]
    >>> x_detector = conversion_nn1_to_detector(x_array)
    >>> x_detector.shape
    (N, 3, 224, 224)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformed_batches = []
    length = len(x_array)
    
    if isinstance(x_array, np.ndarray):
        x_array = torch.tensor(x_array, dtype=torch.float32)
        
    for i in range(0, length, batch_size):
        xb = x_array[i:i+batch_size]
        xb = F.interpolate(xb, size=(224,224), mode='bilinear', align_corners=False)
        xb = (xb + 1) / 2
        xb = xb.permute(0,2,3,1).cpu().numpy()
        xb = (xb - mean) / std
        xb = np.transpose(xb, (0,3,1,2)).astype('float32')
        transformed_batches.append(xb)

    return np.concatenate(transformed_batches, axis=0)

def build_dataloader_from_adversarial(
    x_adv: Union[np.ndarray, torch.Tensor],
    y_true: Union[List[str], List[int], np.ndarray, torch.Tensor],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    num_workers: int = 0
) -> DataLoader:
    """
    Prepares a PyTorch DataLoader from adversarial images and associated labels for evaluation with a second model (NN2).

    The function ensures that adversarial samples are transformed according to the preprocessing required
    by NN2. This includes resizing to 224x224, RGB to BGR conversion, mean subtraction, and label formatting.
    Labels in string format are converted to their corresponding integer indices.

    It is typically used for feeding NN2 with preprocessed adversarial inputs to assess robustness or transferability.

    Parameters
    ----------
    x_adv : np.ndarray or torch.Tensor
        Batch of adversarial images, shape [N, C, H, W].
    y_true : list or array
        True labels as strings or integers.
    class_to_idx : dict
        Mapping from class names to integer indices.
    idx_to_class : dict
        Mapping from indices to class names.
    num_workers : int, optional
        Number of subprocesses used by the DataLoader (default is 0).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader containing preprocessed images and integer labels.

    Examples
    --------
    >>> loader = build_dataloader_from_adversarial_nn2(x_adv, y_true, class_to_idx, idx_to_class)
    >>> for x, y in loader:
    >>>     print(x.shape, y)
    torch.Size([3, 224, 224]) tensor(2)
    """
        
    if isinstance(x_adv, np.ndarray):
        x_adv = torch.tensor(x_adv, dtype=torch.float32)

    if isinstance(y_true[0], str):
        y_tensor = torch.tensor([class_to_idx[label] for label in y_true])
    elif isinstance(y_true, (np.ndarray, list)):
        y_tensor = torch.tensor(y_true)
    else:
        y_tensor = y_true

    dataset = TensorDataset(x_adv, y_tensor)
    dataset.idx_to_class = idx_to_class

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

def generate_one_hot_target_and_plot_image(target_name, LABELS, class_to_idx, x_samples, all_images):
    """
    Generates one-hot encoded labels for a targeted attack and displays the first clean image
    of the target class.

    This function is typically used when setting up a targeted adversarial attack scenario:
    it creates a one-hot label array for the entire batch, corresponding to the desired target class,
    and shows the first image belonging to the target class for visual reference.

    Parameters
    ----------
    target_name : str
        Name of the target class (e.g., 'Fernando_Torres').
    LABELS : list or np.ndarray
        List or array of all class names, as used for human-readable mapping and one-hot encoding.
    class_to_idx : dict
        Mapping from class names to integer indices (as produced by ImageFolder).
    x_samples : np.ndarray or torch.Tensor
        Batch of preprocessed images, shape [N, C, H, W], typically in range [-1, 1].
    all_images : np.ndarray or torch.Tensor
        Full dataset of images (in the same preprocessing and shape), used for visualizing the clean
        reference image for the selected target class.

    Returns
    -------
    np.ndarray
        One-hot encoded labels, shape (N, num_classes), where each row is the one-hot vector
        of the target class, repeated for all samples.

    Examples
    --------
    >>> LABELS = np.array(['Alice', 'Bob', 'Charlie'])
    >>> class_to_idx = {'Alice': 0, 'Bob': 1, 'Charlie': 2}
    >>> one_hot = generate_one_hot_target_and_plot_image('Bob', LABELS, class_to_idx, x_samples, all_images)
    Target (clean): Bob  # (and a plot appears)
    >>> one_hot.shape
    (N, 3)
    """
    if target_name not in class_to_idx:
        raise ValueError(f"Classe '{target_name}' non trovata in class_to_idx.")

    target_idx = class_to_idx[target_name]
    target_class = np.where(np.array(LABELS) == target_name)[0]

    if len(target_class) == 0:
        raise ValueError(f"Classe '{target_name}' non trovata in LABELS.")

    one_hot = to_categorical([target_class[0]], nb_classes=len(LABELS))[0]
    one_hot_targeted_label = np.tile(one_hot, (len(x_samples), 1))

    global_sample_idx = target_idx * 10 + 0
    target_image = all_images[global_sample_idx]

    img = ((target_image + 1) / 2).transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(img, 0, 1))
    plt.title(f"Target (clean): {target_name}", fontsize=14, fontweight='bold', backgroundcolor='#87cefa')
    plt.axis('off')
    plt.show()

    return one_hot_targeted_label

def print_basic_metrics(y_true: List[str],
                        y_pred: List[str],
                        x_orig: Optional[np.ndarray] = None,
                        x_adv: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """
    Prints a summary of prediction performance and, optionally, adversarial perturbation magnitude.

    The function compares the predicted labels against the ground truth and reports the number of
    correct and incorrect classifications. If original and adversarial images are provided, it computes
    the maximum pixel-wise difference between the two to quantify the strength of the attack.

    The function handles inputs in either [-1, 1] or [0, 1] range by automatically rescaling if needed.

    This is a quick diagnostic utility to evaluate the effectiveness of adversarial attacks and
    visualize the accuracy impact alongside perturbation severity.

    Parameters
    ----------
    y_true : list of str
        Ground truth class labels.
    y_pred : list of str
        Model-predicted class labels.
    x_orig : np.ndarray, optional
        Original input images (shape [N, C, H, W]), in [-1, 1] or [0, 1] range.
    x_adv : np.ndarray, optional
        Adversarial input images in the same format as `x_orig`.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the number of correct and incorrect predictions.

    Examples
    --------
    >>> correct, incorrect = print_basic_metrics(y_true, y_pred, x_orig, x_adv)
    Total samples      : 50
    ✓ Correctly class. : 42
    ✗ Misclassified    : 8
    ↪ Max perturbation : 0.0543
    """

    def rescale_if_needed(x):
        if np.min(x) < 0:
            return (x + 1) / 2
        return x

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    incorrect = total - correct

    print(f"Total samples      : {total}")
    print(f"✓ Correctly class. : {correct}")
    print(f"✗ Misclassified    : {incorrect}")

    if x_orig is not None and x_adv is not None:
        x_orig_01 = rescale_if_needed(x_orig)
        x_adv_01 = rescale_if_needed(x_adv)

        if isinstance(x_adv_01, torch.Tensor):
            perturbation = (x_adv_01 - x_orig_01).abs()
            max_perturb = torch.max(perturbation).item()
        else:
            perturbation = np.abs(x_adv_01 - x_orig_01)
            max_perturb = np.max(perturbation)

        print(f"↪ Max perturbation : {max_perturb:.4f}")

    return correct, incorrect

def print_detector_results(flag_adv, total_samples, label="Test data", attack_name=None):
    """
    Prints a summary of detector results for a test set, including the number and percentage
    of samples flagged as adversarial or clean.

    This utility function helps to quickly assess the effectiveness of an adversarial detector,
    displaying how many samples are flagged as adversarial versus clean, with the option to specify
    the name of the attack if applicable.

    Parameters
    ----------
    flag_adv : int
        Number of samples flagged as adversarial by the detector.
    total_samples : int
        Total number of samples evaluated.
    label : str, optional
        Label for the test set (e.g., "Adversarial", "Clean", or a custom description).
    attack_name : str or None, optional
        Name of the attack applied (e.g., "FGSM", "PGD"), or None if not applicable.

    Returns
    -------
    None

    Examples
    --------
    >>> print_detector_results(27, 100, label="Adversarial", attack_name="FGSM")
    Adversarial (FGSM) (100 images):
      Detector flagged as adversarial: 27 (27.0%)
      Detector flagged as clean: 73 (73.0%)

    >>> print_detector_results(0, 50, label="Clean")
    Clean (50 images):
      Detector flagged as adversarial: 0 (0.0%)
      Detector flagged as clean: 50 (100.0%)
    """
    perc_adv = flag_adv / total_samples * 100
    perc_clean = (total_samples - flag_adv) / total_samples * 100

    if attack_name is not None:
        print(f"{label} ({attack_name}) ({total_samples} images):")
    else:
        print(f"{label} ({total_samples} images):")
    print(f"  Detector flagged as adversarial: {flag_adv} ({perc_adv:.1f}%)")
    print(f"  Detector flagged as clean: {total_samples - flag_adv} ({perc_clean:.1f}%)")
    print()
    
def plot_detector_security_evaluation_curve(
    range,
    nb_flag_adv,
    nb_missclass,
    N_SAMPLES,
    title='Security Evaluation Curve',
    xlabel='Epsilon',
    ylabel='Detection / Misclassification (%)',
    bg_color="#e6f2ff",
    legend1='Adversarial samples detected (%)',
    legend2='Classifier misclassification rate (%)',
    marker1='o',
    marker2='x',
    color1='blue',
    color2='red',
    ymin=0,
    ymax=110,
    x_points=None
):
    """
    Plots the Security Evaluation Curve for an adversarial detector and classifier, showing detection rates
    and misclassification rates as a function of attack strength or a chosen parameter (e.g., epsilon).

    This function is useful for visualising the trade-off between adversarial detection and classifier robustness
    across different attack intensities or defences. It displays two curves: the percentage of adversarial
    samples detected, and the percentage of samples misclassified by the classifier, both normalised to the
    total number of test samples.

    Parameters
    ----------
    range : list or array-like
        Range of parameter values (e.g., epsilon for adversarial attacks). The value 0 will be automatically
        prepended for display purposes.
    nb_flag_adv : list or array-like
        Number of samples flagged as adversarial by the detector for each parameter value (excluding 0).
    nb_missclass : list or array-like
        Number of misclassified samples for each parameter value (excluding 0).
    N_SAMPLES : int
        Total number of test samples (used for percentage normalisation).
    title : str, optional
        Title of the plot (default: 'Security Evaluation Curve').
    xlabel : str, optional
        Label for the X axis (default: 'Epsilon').
    ylabel : str, optional
        Label for the Y axis (default: 'Detection / Misclassification (%)').
    bg_color : str, optional
        Background colour of the plot (default: '#e6f2ff').
    legend1 : str, optional
        Legend for the detection curve (default: 'Adversarial samples detected (%)').
    legend2 : str, optional
        Legend for the misclassification curve (default: 'Classifier misclassification rate (%)').
    marker1 : str, optional
        Marker style for the detection curve (default: 'o').
    marker2 : str, optional
        Marker style for the misclassification curve (default: 'x').
    color1 : str, optional
        Colour for the detection curve (default: 'blue').
    color2 : str, optional
        Colour for the misclassification curve (default: 'red').
    ymin : float, optional
        Minimum value for the Y axis (default: 0).
    ymax : float, optional
        Maximum value for the Y axis (default: 110).
    x_points : int or None, optional
        Number of X points to display; if None, defaults to the length of `range` plus one.

    Returns
    -------
    None

    Examples
    --------
    >>> epsilons = [0.01, 0.02, 0.05, 0.1]
    >>> detected = [8, 25, 55, 93]
    >>> misclassified = [4, 18, 33, 65]
    >>> plot_detector_security_evaluation_curve(
    ...     epsilons, detected, misclassified, 100,
    ...     title="Detector vs Classifier Robustness"
    ... )
    """
    # Prepare the data
    range_plot = [0] + list(range)
    nb_flag_adv_plot = np.array([0] + list(nb_flag_adv)) / N_SAMPLES * 100
    nb_missclass_plot = np.array([0] + list(nb_missclass)) / N_SAMPLES * 100

    # Dynamic number of points
    if x_points is None:
        x_points = len(range_plot)

    fig, ax = plt.subplots(figsize=(7, 5))
    curves = [
        (np.array(range_plot), nb_flag_adv_plot, marker1, color1, legend1),
        (np.array(range_plot), nb_missclass_plot, marker2, color2, legend2)
    ]
    plot_multiple_sec_curves(
        ax,
        curves,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        bg_color=bg_color,
        ymin=ymin,
        ymax=ymax
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
    plt.show()

def plot_predicted_images(
    x_test: torch.Tensor,
    y_true: list,
    y_pred: list,
    title: str = "Model Predictions: Clean vs Adversarial",
    id_test_images: list = [1, 4, 32, 33, 59, 73],
    image_idx: int = 3,
    save_path: str = None,
    x_adv: Optional[torch.Tensor] = None,
    y_pred_adv: Optional[list] = None
):
    """
    Displays clean and (optionally) adversarial images together with labels for visual assessment.

    This function creates a visual comparison of model predictions on both clean and adversarial images.
    The first row shows clean images with the true label (and optionally the predicted label).
    If adversarial images are provided, a second row is displayed, showing adversarial images with the
    predicted label beneath each image. The function can optionally save the figure to a file.

    Parameters
    ----------
    x_test : torch.Tensor
        Batch of clean test images of shape [N, C, H, W].
    y_true : list of str
        Ground truth class labels corresponding to each test image.
    y_pred : list of str
        Model predictions for the clean images.
    title : str, optional
        Title for the plot (default: "Model Predictions: Clean vs Adversarial").
    id_test_images : list of int, optional
        List of identity indices to display (default: [1, 4, 32, 33, 59, 73]).
    image_idx : int, optional
        Index of the sample within each identity group to display (default: 3).
    save_path : str or None, optional
        Path to save the plotted figure (default: None, does not save).
    x_adv : torch.Tensor, optional
        Batch of adversarial images of shape [N, C, H, W]. If provided, adversarial images will be displayed in the second row.
    y_pred_adv : list of str, optional
        Model predictions for the adversarial images.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_predicted_images(x_test, y_true, y_pred)
    # Displays only clean images

    >>> plot_predicted_images(x_test, y_true, y_pred, x_adv=x_adv, y_pred_adv=y_pred_adv)
    # Displays both clean and adversarial images

    >>> plot_predicted_images(x_test, y_true, y_pred, save_path="predictions.png")
    # Saves the plot to "predictions.png"
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def invert_standardization(image: np.ndarray) -> np.ndarray:
        return (image + 1) / 2 if image.min() < 0 else image

    def to_numpy(img_tensor):
        img = img_tensor.detach().cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
        return invert_standardization(img)

    x_test = to_numpy(x_test)
    if x_adv is not None:
        x_adv = to_numpy(x_adv)

    num_images = len(id_test_images)
    show_adv = x_adv is not None and y_pred_adv is not None
    num_rows = 2 if show_adv else 1

    fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images * 3, 5.5 if show_adv else 3))
    fig.subplots_adjust(top=0.85)
    fig.suptitle(title, fontsize=15, fontweight='bold', backgroundcolor='#87cefa')

    if num_images == 1:
        axes = [[axes[0]], [axes[1]]] if show_adv else [axes]

    for i, idx in enumerate(id_test_images):
        # Clean image (top row)
        ax_clean = axes[0][i] if show_adv else axes[i]
        clean_img = x_test[10 * idx + image_idx].transpose(1, 2, 0)
        clean_img = np.clip(clean_img, 0, 1)
        true_label = y_true[10 * idx + image_idx]

        ax_clean.imshow(clean_img)
        ax_clean.axis('off')
        ax_clean.text(0.5, 1.08,
                      f"True: {true_label}",
                      transform=ax_clean.transAxes,
                      ha='center', va='bottom',
                      fontsize=8, color='black', fontweight='bold')
        
        if not show_adv:
            pred_label = y_pred[10 * idx + image_idx]
            pred_color = 'green' if pred_label == true_label else 'red'
            ax_clean.text(0.5, 1.01,
                          f"Pred: {pred_label}",
                          transform=ax_clean.transAxes,
                          ha='center', va='bottom',
                          fontsize=8, color=pred_color, fontweight='bold')

        # Adversarial image (bottom row)
        if show_adv:
            ax_adv = axes[1][i]
            adv_img = x_adv[10 * idx + image_idx].transpose(1, 2, 0)
            adv_img = np.clip(adv_img, 0, 1)
            pred_label = y_pred_adv[10 * idx + image_idx]
            pred_color = 'green' if pred_label == true_label else 'red'

            ax_adv.imshow(adv_img)
            ax_adv.axis('off')
            ax_adv.text(0.5, 1.01,
                        f"Pred: {pred_label}",
                        transform=ax_adv.transAxes,
                        ha='center', va='bottom',
                        fontsize=8, color=pred_color, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_multiple_sec_curves(ax, curves, title, xlabel, ylabel, bg_color='#e6f2ff', ymin=0, ymax=1):
    """
    Plots multiple Security Evaluation Curves on a shared axis.

    This function accepts a list of curves (each with its own X and Y values,
    marker, color, and label) and plots them on the same matplotlib Axes.
    The Y axis is automatically scaled with dynamic margins. A legend is
    displayed if more than one curve is plotted or if the single curve has a label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the plot on.
    curves : list of tuples
        Each element must be a tuple: (x_values, y_values, marker, color, label).
    title : str
        Title of the plot.
    xlabel : str
        Label for the X axis.
    ylabel : str
        Label for the Y axis.
    bg_color : str
        Background color of the plot (e.g., "#f0f0f0").
    ymin : float, optional
        Minimum Y axis limit (default is 0).
    ymax : float, optional
        Maximum Y axis limit (default is 1).

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> curves = [
    >>>     ([0.1, 0.2, 0.3], [0.9, 0.7, 0.4], 'o', 'blue', 'Model A'),
    >>>     ([0.1, 0.2, 0.3], [0.95, 0.85, 0.5], 'x', 'red', 'Model B')
    >>> ]
    >>> plot_multiple_sec_curves(ax, curves, "SEC Curves", "Epsilon", "Accuracy", "#ffffff")
    >>> plt.show()
    """
    if not curves:
        raise ValueError("La lista 'curves' è vuota.")

    for x, y, marker, color, label in curves:
        ax.plot(x, y, marker=marker, color=color, linewidth=2, label=label)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    ax.set_facecolor(bg_color)
    ax.grid(True)
    ax.margins(x=0, y=0)
    
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    if len(curves) > 1:
        ax.legend(loc='best')
    else:
        if curves[0][4]:
            ax.legend(loc='best')