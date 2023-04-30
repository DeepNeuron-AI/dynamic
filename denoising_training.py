from typing import Iterable
import cv2
from torch.utils.data import Dataset
import torch 
import numpy as np
from torchvision import transforms
import scipy.ndimage
import echonet 
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()


def to_tensor(array: np.ndarray):
    return torch.from_numpy(array)


def to_float(tensor: torch.Tensor):
    return tensor.float()


def brighten_dark_areas(array: np.ndarray) -> np.ndarray:
    array = 255 * 0.5 * ((array/255)**2 + 1)
    array = array.astype(np.uint8)
    array = np.clip(array, 0, 255)
    return array
    
def add_contrast(array: np.ndarray) -> np.ndarray:
    alpha = 0.3
    beta = 50
    array = alpha * array + beta
    array = array.astype(np.uint8)
    array = np.clip(array, 0, 255)
    return array


class DenoisedDataset(Dataset):

    def __init__(self, directory: str, noise_factor: float = 0.05, kernel_size: int = 11):
        self.directory = Path(directory)
        self.video_directory = self.directory / "Videos"
        self.noise_factor = noise_factor
        
        # Compute mean and std
        # mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=str(self.directory), split="train"))
        # print(mean, std)
        self.mean = 32.823921 # [~50, ~50, ~50]
        self.std = 50.203746  # [~70, ~70, ~70]

        # Get filepaths for each video
        self.filepaths = list(self.video_directory.iterdir())

        self.transform = transforms.Compose([  # (color, frame, height, width)
            BGRToGray(), # (frame, height, width)
            Normalize(self.mean, self.std),
            to_tensor
        ])
        self.noisy_transform = transforms.Compose([  # (color, frame, height, width)
            BGRToGray(),  # (frame, height, width)
            # brighten_dark_areas,
            add_contrast,
            MedianBlur(kernel_size=(1, kernel_size, kernel_size)),
            # AverageBlur(kernel_size=(31, 1, 1)), # can average over frames if we want!
            Normalize(self.mean, self.std), 
            to_tensor, 
            # AddGaussianNoise(std=noise_factor)
        ])

    def __getitem__(self, index):
        raw_video = echonet.utils.loadvideo(str(self.filepaths[index])) # (color, frame, height, width)
        video = self.transform(raw_video)
        noisy_video = self.noisy_transform(raw_video)
        # video = to_tensor(video)
        # noisy_video = to_tensor(noisy_video)
        return noisy_video, video

    def __len__(self):
        return len(self.filepaths) 


class AddGaussianNoise(object):
    '''
    from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745 
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Transpose:
    def __init__(self, dims) -> None:
        self.dims = dims

    def __call__(self, array: np.ndarray):
        return array.transpose(self.dims)      


class BGRToGray:
    def __call__(self, array: np.ndarray):
        """
        Parameters
        ---------
        array: np.ndarray
            Dimensions should be (color, frame, height, width)
        """
        return (0.299*array[2] + 0.587*array[1] + 0.114*array[0]).astype(np.uint8)


class Normalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, array: np.ndarray):
        return (array - self.mean) / self.std


class MedianBlur:
    def __init__(self, kernel_size: Iterable[int]) -> None:
        for i, dim in enumerate(kernel_size):
            if dim % 2 == 0:
                raise ValueError(f"Kernel size should be odd integer but was {dim} along dimension {i}")

        self.kernel_size = kernel_size

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return scipy.ndimage.median_filter(array, size=self.kernel_size)
    

class AverageBlur:
    def __init__(self, kernel_size: Iterable[int]) -> None:
        for i, dim in enumerate(kernel_size):
            if dim % 2 == 0:
                raise ValueError(f"Kernel size should be odd integer but was {dim} along dimension {i}")

        self.ave_filter = np.ones(kernel_size) / np.prod(kernel_size)

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return scipy.ndimage.convolve(array, self.ave_filter)


if __name__ == "__main__":
    directory = os.environ.get("ECHONET_DIR", "a4c-video-dir")
    dataset = DenoisedDataset(directory, kernel_size=11, noise_factor=0.25)
    noisy_video, video = dataset[1000]
    noisy_video = noisy_video.numpy()
    video = video.numpy()

    mean, std = dataset.mean, dataset.std
    video = (video * std + mean).astype(np.uint8)

    noisy_video = noisy_video * std + mean
    noisy_video = np.clip(noisy_video, 0, 255)
    noisy_video = noisy_video.astype(np.uint8)
    
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Noisy", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)

    i = 0
    is_playing = True
    while True:
        if i >= len(video):
            i = 0

        frame, thresh = cv2.threshold(video[i],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow("Threshold", frame)

        cv2.imshow("Original", video[i])
        cv2.imshow("Noisy", noisy_video[i])

        keypress = cv2.waitKey(20) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord(' '):
            is_playing = not is_playing
        else:
            if is_playing:
                i += 1