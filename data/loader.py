import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


class CaseImagePairLoader:
    """A class for loading and managing pairs of inspected and reference images from defective
    and non-defective cases.

    This class handles loading and organizing image pairs from two directories - one containing
    defective cases and another containing non-defective cases. Each case consists of an
    inspected image and a reference image.
    """
    def __init__(self, defective_path: str, non_defective_path: str):
        """Initialize an ImagePairLoader instance with directories for defective and
        non-defective images.

        Args:
            defective_path (str): Directory path containing defective images
                (expected naming: "<case>_inspected_image.tif").
            non_defective_path (str): Directory path containing reference images
                (expected naming: "<case>_reference_image.tif").
        """
        self.defective_path, self.non_defective_path = map(
            Path, (defective_path, non_defective_path)
        )
        self.defective_cases = self._get_cases_from_files(
            os.listdir(self.defective_path)
        )
        self.non_defective_cases = self._get_cases_from_files(
            os.listdir(self.non_defective_path)
        )

    def _get_cases_from_files(self, files: List[str]) -> List[str]:
        """Extract unique case names from a list of image filenames.

        Args:
            files (List[str]): List of filenames to process

        Returns:
            List[str]: List of unique case names sorted alphabetically. Case names are
                extracted from filenames by taking the part before the first underscore.
        """
        return sorted(list(set([
            d.split('_')[0] for d in files if d.endswith('.tif')
        ])))

    def load_defective_image_pairs(
        self,
        mode: str
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Load image pairs from the specified directory based on the provided mode.

        Args:
            mode (str): Specifies which image pairs to load - use "defective" for defective
                cases and "non_defective" for non-defective cases.

        Returns:
            List[Tuple[str, np.ndarray, np.ndarray]]: A list of tuples, where each tuple
                consists of:
                - case name (str): The identifier for the case
                - inspected image (np.ndarray): The inspected image loaded as a numpy array
                - reference image (np.ndarray): The corresponding reference image loaded as
                    a numpy array

        Raises:
            ValueError: If the mode is neither "defective" nor "non_defective".
        """
        mode_config = {
            "defective": (self.defective_cases, self.defective_path),
            "non_defective": (self.non_defective_cases, self.non_defective_path)
        }

        if mode not in mode_config:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'defective' or 'non_defective'."
            )

        cases, directory = mode_config[mode]

        return [
            (
                case,
                cv2.imread(
                    os.path.join(directory, f"{case}_inspected_image.tif"),
                    cv2.IMREAD_UNCHANGED
                ),
                cv2.imread(
                    os.path.join(directory, f"{case}_reference_image.tif"),
                    cv2.IMREAD_UNCHANGED
                )
            )
            for case in cases
        ]
