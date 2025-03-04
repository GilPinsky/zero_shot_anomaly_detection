import cv2
import cc3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from typing import Tuple, List, Optional
from utils.aligner import CaseImagePairAligner


class DefectDetector:
    """A class for detecting defects by comparing inspected images with reference images."""

    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 4,
        erosion_kernel_size: Tuple[int, int] = (5, 5),
        blur_kernel_size: Tuple[int, int] = (7, 7),
        blur_sigma: int = 4,
        min_pool_kernel_size: int = 11,
    ):
        """Initialize the defect detector.

        Args:
            model: The neural network model for feature extraction.
                  Must have an encoder attribute to extract features.
            threshold (float): Threshold value for the features difference (0-1).
            erosion_kernel_size (tuple): Size of kernel used for eroding valid region mask to
                                       handle edge effects resulting from the network.
            blur_kernel_size (tuple): Size of Gaussian blur kernel for noise reduction in images.
            blur_sigma (int): Standard deviation of Gaussian blur kernel.
            min_pool_kernel_size (int): Size of min pooling kernel for suppressing noise in
                                      the feature difference map.
        """
        self.model = model
        self.model.eval()

        self.threshold = threshold
        self.erosion_kernel_size = erosion_kernel_size
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.min_pool_kernel_size = min_pool_kernel_size

    def preprocess_images(
        self, reference_img: np.ndarray, inspected_img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess and align the image pair.
        
        Args:
            reference_img (np.ndarray): Reference image without defects.
            inspected_img (np.ndarray): Image being inspected for defects.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Preprocessed and aligned reference image tensor
                - Preprocessed inspected image tensor  
                - Valid region mask tensor
        """
        # Align images
        aligned_reference_img, valid_region = CaseImagePairAligner.align_images_affine(
            reference_img, inspected_img
        )

        # Erode valid region - to compensate for network edge artifacts
        valid_region = cv2.erode(
            valid_region.astype(np.uint8),
            np.ones(self.erosion_kernel_size, np.uint8)
        )

        # Preprocess images
        aligned_reference_img = cv2.GaussianBlur(
            aligned_reference_img.astype(np.float32) / 255,
            self.blur_kernel_size,
            self.blur_sigma
        )
        inspected_img = cv2.GaussianBlur(
            inspected_img.astype(np.float32) / 255,
            self.blur_kernel_size,
            self.blur_sigma
        )

        # Apply valid region mask
        aligned_reference_img *= valid_region
        inspected_img *= valid_region

        # Convert to tensors
        return (
            torch.from_numpy(aligned_reference_img).float(),
            torch.from_numpy(inspected_img).float(),
            torch.from_numpy(valid_region)
        )

    @torch.no_grad()
    def compute_feature_difference(
        self,
        reference_img: torch.Tensor,
        inspected_img: torch.Tensor,
        valid_region: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute the feature difference between reference and inspected images.

        This method extracts deep features from both images using the model's encoder,
        interpolates them to the target size, and computes their normalized difference.
        The difference is then processed with min-pooling to identify significant feature
        discrepancies that may indicate defects.

        Args:
            reference_img (torch.Tensor): The aligned reference image tensor
            inspected_img (torch.Tensor): The inspected image tensor
            valid_region (torch.Tensor): Binary mask indicating valid image regions
            target_size (Tuple[int, int]): Desired output size for feature maps

        Returns:
            torch.Tensor: Feature difference map highlighting areas of dissimilarity
                         between the reference and inspected images
        """
        # Extract features
        feats_inspected = self.model.encoder(inspected_img.unsqueeze(0).unsqueeze(0))[2]
        feats_reference = self.model.encoder(reference_img.unsqueeze(0).unsqueeze(0))[2]

        # Interpolate to target size
        feats_inspected = F.interpolate(feats_inspected, size=target_size, mode='bilinear')
        feats_reference = F.interpolate(feats_reference, size=target_size, mode='bilinear')

        # Compute difference and apply min pooling
        feats_diff = (
            valid_region.unsqueeze(0).unsqueeze(0) *
            (feats_inspected - feats_reference)
        ).norm(dim=[0, 1])

        # Fix: Adjust padding to be centered and account for kernel offset
        pad_size = self.min_pool_kernel_size // 2
        feats_diff = F.pad(
            feats_diff,
            (pad_size, pad_size, pad_size, pad_size),
            mode='constant',
            value=float('inf')
        )
        feats_diff = F.max_pool2d(
            -feats_diff.unsqueeze(0),
            kernel_size=self.min_pool_kernel_size,
            stride=1,
            padding=0
        )[0]
        return -feats_diff

    def detect_defects(
        self,
        reference_img: np.ndarray,
        inspected_img: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], np.ndarray]:
        """Detect defects by comparing reference and inspected images using feature-based analysis.

        This method performs the following steps:
        1. Preprocesses and aligns the input images
        2. Computes deep feature differences between aligned images
        3. Generates a binary prediction mask based on feature differences
        4. Identifies connected components and their centroids

        Args:
            reference_img (np.ndarray): Reference image without defects to compare against.
                Expected to be a grayscale image with shape (H, W).
            inspected_img (np.ndarray): Image being inspected for potential defects.
                Should match reference image dimensions.

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]], np.ndarray]: A tuple containing:
                - prediction_mask: Binary mask where True indicates detected defects
                - centroids: List of (x,y) coordinates for the center of each detected defect
                - feats_diff: Feature difference map showing areas of dissimilarity
        """
        # Preprocess images
        aligned_reference, inspected, valid_region = self.preprocess_images(
            reference_img, inspected_img
        )

        # Compute feature difference
        feats_diff = self.compute_feature_difference(
            aligned_reference, inspected, valid_region, reference_img.shape
        )

        # Generate prediction mask
        prediction_mask = (feats_diff > self.threshold) * valid_region

        # Find connected components and compute centroids
        cc = cc3d.connected_components(prediction_mask.cpu().numpy())
        stats = cc3d.statistics(cc.astype(np.uint32))
        centroids = stats["centroids"][1:][:, ::-1]

        return prediction_mask.cpu().numpy(), centroids, feats_diff.cpu().numpy()

    def visualize_results(
        self,
        inspected_img: np.ndarray,
        feats_diff: np.ndarray,
        prediction_mask: np.ndarray,
        centroids: List[Tuple[float, float]],
        case_name: Optional[str] = None
    ) -> plt.Figure:
        """Visualize detection results by creating a figure with three subplots.

        Shows the feature difference map, prediction mask with labeled defect centroids,
        and the original inspected image.

        Args:
            inspected_img (np.ndarray): The original inspected image
            feats_diff (np.ndarray): Feature difference map between reference and inspected images
            prediction_mask (np.ndarray): Binary mask indicating detected defect locations
            centroids (List[Tuple[float, float]]): List of defect centroids as (x,y) coordinates
            case_name (Optional[str]): Name of the case to display as figure title.
                                     Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure containing the three visualization subplots
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

        # Plot feature difference
        ax1.imshow(feats_diff)
        ax1.set_title('Features Difference')

        # Plot prediction mask with centroids
        ax2.imshow(prediction_mask)
        ax2.set_title('Defect Predictions')
        to_char = lambda i: chr(65 + i)
        for i, (x, y) in enumerate(centroids):
            ax2.text(x + 20, y + 20, f'Defect {to_char(i)}', color='red')

        # Plot inspected image
        ax3.imshow(inspected_img)
        ax3.set_title('Inspected Image')

        if case_name:
            fig.suptitle(f'{case_name}:')
        plt.tight_layout()
        return fig