import os
import cv2
import numpy as np

class DatasetLoader:
    def __init__(self, dataset_path, dataset_type="KITTI"):
        """
        Initialize the dataset loader.
        :param dataset_path: Path to the dataset directory.
        :param dataset_type: Type of the dataset ("KITTI" or "Malaga").
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type.lower()
        self.images = []
        self.calibration = None
        self.current_idx = 0

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """
        Load dataset images and calibration based on the dataset type.
        """
        if self.dataset_type == "kitti":
            self._load_kitti()
        elif self.dataset_type == "malaga":
            self._load_malaga()
        elif self.dataset_type == "parking":
            self._load_parking()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _load_kitti(self):
        """
        Load KITTI dataset.
        """
        image_path = os.path.join(self.dataset_path, "image_0")
        self.images = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith(".png")])

        # Load calibration
        calib_path = os.path.join(self.dataset_path, "calib.txt")
        with open(calib_path, "r") as f:
            for line in f:
                if line.startswith("P0:"):
                    calib_data = line.split()[1:]
                    self.calibration = np.array(calib_data, dtype=np.float32).reshape(3, 4)
                    break

    def _load_malaga(self):
        """
        Load Malaga dataset.
        """
        image_path = os.path.join(self.dataset_path, "Images")
        self.images = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith("left.jpg")])

        # Calibration (assuming fixed for Malaga)
        self.calibration = np.array([[621.18428, 0.0, 404.0076],
                                      [0.0, 621.18428, 309.05989],
                                      [0.0, 0.0, 1.0]], dtype=np.float32)

    def _load_parking(self):
        """
        Load Parking dataset.
        """
        image_path = os.path.join(self.dataset_path, "images")
        self.images = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith(".png")])

        # Load calibration
        calib_path = os.path.join(self.dataset_path, "K.txt")
        with open(calib_path, "r") as f:
            calib_data = []
            for line in f:
                calib_data.extend([x for x in line.strip().split(',') if x])
            self.calibration = np.array(calib_data, dtype=np.float32).reshape(3, 3)


    def get_next_image(self):
        """
        Get the next image in the sequence.
        :return: Grayscale image or None if end of sequence is reached.
        """
        if self.current_idx >= len(self.images):
            return None
        image_path = self.images[self.current_idx]
        self.current_idx += 1
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def get_image(self, idx):
        """
        Get the image at the specified index.
        :param idx: Index of the image to retrieve.
        :return: Grayscale image or None if index is out of range.
        """
        if idx < 0 or idx >= len(self.images):
            return None
        return cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)

    def get_calibration(self):
        """
        Get the camera intrinsic matrix.
        :return: 3x3 intrinsic matrix.
        """
        K = self.calibration
        if K.shape != (3, 3):  # Handle the case where K is not the correct size
            K = K[:3, :3]  # Extract the 3x3 intrinsic matrix
        K = K.astype(np.float32)
        self.calibration = K
        return K

    def get_num_images(self):
        """
        Get the total number of images in the dataset.
        :return: Number of images.
        """
        return len(self.images)

    def reset(self):
        """
        Reset the dataset loader to the first image.
        """
        self.current_idx = 0

