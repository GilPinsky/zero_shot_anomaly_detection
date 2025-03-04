import cv2
import numpy as np
from typing import Tuple


class CaseImagePairAligner:
    """A class for aligning pairs of images using affine transformations.

    This class provides methods to align a moving (target) image with a reference image by:
    1. Estimating an initial affine transformation using SIFT features and RANSAC
    2. Refining the transformation using Enhanced Correlation Coefficient (ECC) maximization
       across multiple pyramid levels
    3. Applying the final transformation to align the images

    The alignment process is robust to noise and can handle moderate differences between
    the images by combining feature-based and intensity-based registration methods.
    """

    @staticmethod
    def estimate_affine_transform_ransac(
        img_ref: np.ndarray,
        img_mov: np.ndarray,
        num_levels: int = 3,
        blur_kernel_size: Tuple[int, int] = (3, 3),
        blur_sigma: float = 0.5,
        ransac_reproj_threshold: float = 1.0,
        ransac_confidence: float = 0.999,
        ransac_max_iters: int = 4000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate initial affine transform between two images using SIFT features and RANSAC.

        This method performs the following steps:
        1. Preprocesses images with Gaussian blur to reduce noise
        2. Detects SIFT keypoints and computes descriptors
        3. Matches features between images using brute force matching
        4. Estimates affine transform using RANSAC on matched point pairs
        5. Adjusts transform parameters for pyramid levels

        Args:
            img_ref: Reference image to align to
            img_mov: Moving image that will be transformed
            num_levels: Number of pyramid levels for scaling translation components
            blur_kernel_size: Size of Gaussian blur kernel for preprocessing
            blur_sigma: Standard deviation of Gaussian blur for preprocessing
            ransac_reproj_threshold: Maximum allowed reprojection error in RANSAC
            ransac_confidence: Confidence level for RANSAC estimation (0-1)
            ransac_max_iters: Maximum number of RANSAC iterations

        Returns:
            A tuple containing:
                - Initial affine transform matrix (2x3 float32 array)
                - Preprocessed reference image (float32 array)
                - Preprocessed moving image (float32 array)
        """
        # Apply Gaussian blur to reduce noise
        blurred_images = [
            cv2.GaussianBlur(img, blur_kernel_size, blur_sigma)
            for img in (img_ref, img_mov)
        ]
        img_ref, img_mov = blurred_images

        # Initialize SIFT detector and compute keypoints & descriptors
        sift = cv2.SIFT_create()
        keypoints_descriptors = [
            sift.detectAndCompute(img, None) for img in (img_ref, img_mov)
        ]
        (kp1, des1), (kp2, des2) = keypoints_descriptors

        # Match features using BFMatcher with L2 norm
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)

        # Extract corresponding point coordinates
        point_pairs = [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in matches]
        src_pts = np.float32([p[0] for p in point_pairs]).reshape(-1, 1, 2)
        dst_pts = np.float32([p[1] for p in point_pairs]).reshape(-1, 1, 2)

        # Estimate initial affine transform using RANSAC
        warp_matrix, _ = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            ransacReprojThreshold=ransac_reproj_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iters
        )

        # Create copy of warp matrix and adjust for pyramid levels
        warp_matrix_c = (
            warp_matrix.copy().astype(np.float32)
            if warp_matrix is not None
            else np.eye(2, 3, dtype=np.float32)
        )
        if warp_matrix is not None:
            warp_matrix_c[:, 2] /= 2 ** (num_levels - 1)

        # Convert images to float32 for pyramid construction
        gray_ref, gray_mov = [img.astype(np.float32) for img in (img_ref, img_mov)]

        return warp_matrix_c, gray_ref, gray_mov

    @staticmethod
    def refine_affine_transform_ecc(
        reference_img: np.ndarray,
        inspected_img: np.ndarray,
        initial_warp: np.ndarray,
        num_pyramid_levels: int = 3,
        warp_mode: int = cv2.MOTION_AFFINE,
        max_iterations: int = 4000,
        eps: float = 1e-8
    ) -> np.ndarray:
        """Refine affine transform using ECC algorithm across image pyramid levels.

        Args:
            reference_img: Reference image to align to
            inspected_img: Image to be aligned
            initial_warp: Initial affine transform matrix from RANSAC
            num_pyramid_levels: Number of pyramid levels for multi-scale refinement
            warp_mode: Type of transform for ECC
            max_iterations: Maximum number of ECC iterations
            eps: ECC convergence threshold

        Returns:
            Refined affine transform matrix. If refinement fails, returns initial_warp.
        """
        # Blur images to reduce noise and improve ECC convergence
        reference_img = cv2.GaussianBlur(reference_img, (7, 7), 5)
        inspected_img = cv2.GaussianBlur(inspected_img, (7, 7), 5)

        # Build Gaussian pyramids for both images
        pyramid_ref = [reference_img]
        pyramid_mov = [inspected_img]
        for _ in range(1, num_pyramid_levels):
            pyramid_ref.append(cv2.pyrDown(pyramid_ref[-1]))
            pyramid_mov.append(cv2.pyrDown(pyramid_mov[-1]))

        # Define ECC termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                   max_iterations, eps)

        # Refine the transform using an ECC pyramid
        warp_matrix_c = initial_warp.copy()
        failed = True
        for level in reversed(range(num_pyramid_levels)):
            try:
                _, warp_matrix_c = cv2.findTransformECC(
                    templateImage=pyramid_ref[level],
                    inputImage=pyramid_mov[level],
                    warpMatrix=warp_matrix_c,
                    motionType=warp_mode,
                    criteria=criteria
                )
                if level == 0:
                    failed = False
            except cv2.error:
                print(f"ECC refinement failed at level {level}, "
                      "continuing with current transform")

            if level > 0:
                warp_matrix_c[:, 2] *= 2.0  # Update translation before next level
        return initial_warp if failed else warp_matrix_c

    @staticmethod
    def compute_affine_transform_pyramid_ecc(
        img_ref: np.ndarray,
        img_mov: np.ndarray,
        num_levels: int = 3,
        warp_mode: int = cv2.MOTION_AFFINE,
        max_iterations: int = 4000,
        eps: float = 1e-8
    ) -> np.ndarray:
        """Compute affine transform by combining RANSAC initialization and ECC refinement.

        Args:
            img_ref: Reference image
            img_mov: Moving image to be aligned
            num_levels: Number of pyramid levels
            warp_mode: Type of transform for ECC
            max_iterations: Maximum ECC iterations
            eps: ECC convergence threshold

        Returns:
            Final affine transform matrix
        """
        # Get initial transform from RANSAC
        initial_warp, gray_ref, gray_mov = (
            CaseImagePairAligner.estimate_affine_transform_ransac(
                img_ref, img_mov, num_levels
            )
        )

        # Refine transform using ECC
        final_warp = CaseImagePairAligner.refine_affine_transform_ecc(
            gray_ref,
            gray_mov,
            initial_warp,
            num_pyramid_levels=num_levels,
            warp_mode=warp_mode,
            max_iterations=max_iterations,
            eps=eps
        )

        return final_warp

    @staticmethod
    def align_images_affine(
        reference_img: np.ndarray,
        inspected_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given two images where there is an affine transform between them,
        find the transform and perform the correction on the second image
        so that it is aligned with the first image.

        Args:
            reference_img: The reference image used as the target alignment
            inspected_img: The image to be aligned with the reference image

        Returns:
            A tuple containing:
                - The aligned version of inspected_img warped to match reference_img
                - A boolean mask indicating valid regions after warping
        """
        # Compute affine transform between the images
        transform_matrix = CaseImagePairAligner.compute_affine_transform_pyramid_ecc(
            reference_img,
            inspected_img
        )

        # Apply transform to align inspected image with reference
        aligned_image = cv2.warpAffine(
            reference_img,
            transform_matrix,
            (reference_img.shape[1], reference_img.shape[0]),
            borderValue=0
        )

        # Create mask to track valid transformed regions
        mask = np.ones_like(reference_img, dtype=np.uint8)
        valid_region = cv2.warpAffine(
            mask,
            transform_matrix,
            (reference_img.shape[1], reference_img.shape[0]),
            borderValue=0
        )
        valid_region = valid_region > 0

        return aligned_image, valid_region