import cv2
from torch.utils.data import Dataset
import torch 
import numpy as np
from torchvision import transforms
import os 
import echonet 
from pathlib import Path


def to_tensor(array: np.ndarray):
    return torch.from_numpy(array)
        
def to_float(tensor: torch.Tensor):
    return tensor.float()


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

        self.transform = transforms.Compose([
            ConvertColor(), # (frame, height, width)
            Normalize(self.mean, self.std),
            to_tensor
        ])
        self.noisy_transform = transforms.Compose([ # (color, frame, height, width)
            Transpose((1, 2, 3, 0)), # (frame,height, width, color)
            MedianBlur(kernel_size=kernel_size), # (frame,height, width, color)
            Transpose((3, 0, 1, 2)), # (color, frame, height, width)
            ConvertColor(), # (frame, height, width)
            Normalize(self.mean, self.std),
            to_tensor,
            AddGaussianNoise(std=noise_factor)
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


class ConvertColor:
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
    def __init__(self, kernel_size: int) -> None:
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size should be odd integer but was {kernel_size}!")

        self.kernel_size = kernel_size

    def __call__(self, array: np.ndarray) -> np.ndarray:
        result = np.zeros(array.shape, dtype=array.dtype)
        for i, frame in enumerate(array):
            print(i)
            result[i] = cv2.medianBlur(frame, self.kernel_size)
        return result
        



if __name__ == "__main__":
    directory = r"C:\Users\Allis\Documents\MDN\Ultrasound2023\dynamic\a4c-video-dir"
    dataset = DenoisedDataset(directory, noise_factor=0.1)
    noisy_video, video = dataset[1000]
    noisy_video = noisy_video.numpy()
    video = video.numpy()

    mean, std = dataset.mean, dataset.std
    print(((video * std + mean) < 0).sum())
    video = (video * std + mean).astype(np.uint8)
    # video = video.transpose((0, 3, 1, 2)) 

    print(((noisy_video * std + mean) < 0).sum())
    noisy_video = noisy_video * std + mean
    noisy_video = np.clip(noisy_video, 0, 255)
    noisy_video = noisy_video.astype(np.uint8)
    
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Noisy", cv2.WINDOW_NORMAL)

    i = 0
    is_playing = True
    while True:
        if i >= len(video):
            i = 0

        cv2.imshow("Original", video[i])
        cv2.imshow("Noisy", noisy_video[i])

        keypress = cv2.waitKey(50) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord(' '):
            is_playing = not is_playing
        else:
            if is_playing:
                i += 1

    """
    mean = 30, std= 50
    Starts: (0, 0, 0)
    Normalise: (-0.6, -0.6, -0.6)
    Normalise: (-0.05, 0, 0) -> (255, 0, 0)
    """