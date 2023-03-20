# What is this?
This is a fork of [echonet-dynamic](https://echonet.github.io/dynamic/). `echonet-dynamic` is a deep learning model that takes in echocardiogram ultrasound videos, and can perform the following:
1. semantic segmentation of the left ventricle
2. prediction of ejection fraction by entire video or subsampled clips, and
3. assessment of cardiomyopathy with reduced ejection fraction.

This fork is an extension of `echonet-dynamic` with a few broad goals in mind:

| Goal                                                                                                                      | Status  |
| ------------------------------------------------------------------------------------------------------------------------- | ------- |
| Apply `echonet-dynamic` to a different (possibly low-quality) dataset and test if the accuracy is significantly affected. | Pending |
| Ultimately *automatically* diagnose a patient based on their echocardiogram ultrasound.                                   | Pending |


Other goals are:

| Goal                                                                                                                      | Status                       |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| Implement automatic TAPSE measurement                                                                                     | Done (but could be improved) |
| Use a trained `echonet-dynamic` segmentation model to develop *right* ventricle segmentations in an unsupervised fashion. | In progress                  |
| Post-process to improve the quality of ultrasounds, simulating contrast images                                            | Pending                      |
| Deploy to a network of hospitals, probably using federated learning to manage patient privacy                             | Pending                      |

# Who is behind this?
This project is a collaboration between Monash DeepNeuron and Global MedEd Network.

## Monash DeepNeuron
[Monash DeepNeuron (MDN)](https://www.deepneuron.org/) is a student-led team from Monash University using AI to solve problems. MDN is broken up into branches and smaller teams, who each tackle different projects.

Specifically, this project's MDN members are:

| Name            | Status                | Season(s)                    |
| --------------- | --------------------- | ---------------------------- |
| Allister Lim    | Active (current lead) | 2023 Summer, 2023 Semester 1 |
| Lex Gallon      | Active                | 2023 Summer, 2023 Semester 1 |
| Colin La        | Active (former lead)  | 2023 Summer, 2023 Semester 1 |
| Mark Zheng      | Active                | 2023 Summer, 2023 Semester 1 |
| Elizabeth Perry | Active                | 2023 Semester 1              |
| Jack Bassat     | Previous member       | 2023 Summer                  |

## Global MedEd Network
[Global MedEd Network](https://www.globalmedednetwork.org/) is a not-for-profit aiming to make good-quality healthcare more globally accessible.

# How to use this code?
We'll assume you have some basic understanding of git repos, the command-line, and virtual environments.

## Cloning
In your command prompt, enter:

```
git clone https://github.com/DeepNeuron-AI/dynamic.git
cd dynamic
```

Optionally, you can change the name of the destination folder like so:

```
git clone https://github.com/DeepNeuron-AI/dynamic.git MDN-ultrasound
cd MDN-ultrasound
```

## Environment setup
It is recommended you always use a virtual environment when installing dependencies for a project. Assuming you are using [conda](https://docs.conda.io/en/latest/miniconda.html), you can create a new environment for this repo with:

```
conda env create -f environment.yml
```

## Dataset
You can download the original `echonet` dataset from [their page](https://echonet.github.io/dynamic/). For the original `echonet-dynamic` code, you'll need to add your data directory to a file called `echonet.cfg`, but it's recommended to just read [their original instructions](https://github.com/echonet/dynamic) for this setup.

## Configuration
Our code often relies on a `.env` file to store personal environment variables such as filepaths of particular videos or segmentation files, etc. You'll see what variables you need to define in this file based on whatever python script/notebook you are trying to run.

## Running segmentation on original dataset
To make things easier to debug on our end, we altered the original command-line interface for the `echonet-dynamic` code. To train the segmentation model *in the original code*, you would do something like:

```
echonet segmentation --save_video
```

However, *for this repo*, you would replace this with

```
python -m echonet/utils/segmentation.py --save-video
```

All arguments are otherwise provided in the same format (like `save-video`).

## Running ejection fraction model
Just do

```
python -m echonet/utils/video.py
```

# How do we achieve right ventricle segmentations without any data?
The original `echonet` dataset only contained *left* ventricle (LV) segmentation annotations due to time/effort constraints. For us to segment the *right* ventricle (RV), we tried to trick the already-trained LV segmentation model into finding the RV instead. The way we do this is by simply mirroring the input ultrasound from left to right, which gives the RV a similar shape and orientation to the original LV.

These initial RV segmentations are unsurprisingly imperfect, so we then refine these segmentations using relatively intuitive heuristics. These heuristics mainly work off of the (trained model's) LV segmentations, since we know those to be of relatively high quality. For example, we can estimate the septum's location based on the LV's, and so can remove any part of the septum that appears in the RV segmentations.

## Where do we do this?
Originally, we simply ran the pre-trained segmentation model on a flipped copy of the dataset. This only saves segmentations for the *test* split (~1,000 of the 10,000 videos). We could then read both our LV and RV segmentations from our saved files and go from there.

The main addition so far is [weak_labels.ipynb](./weak_labels.ipynb). This notebook was our playground for reusing the left-ventricle segmentation model in order to achieve *right* ventricle segmentations (without any ground truth right-ventricle data!). It is intended to load in a *single* video, along with its corresponding LV and initial RV segmentations. It then takes steps to refine the RV segmentations, ultimately displaying the final, hopefully improved, RV segmentation in a separate window.

## How do we verify the quality of our RV segmentations?
**We can't**, really. More precisely, we cannot *quantify* the quality of our RV segmentations (e.g. using DICE similarity scores) since we have *no* ground-truth RV data to compare with. Therefore, we were forced to qualitatively evaluate our RV segmentations by eye, adjusting our techniques according to whatever "looked" best. Getting access to annotated RV data would obviously make a huge difference here!

## What techniques did we use to refine the RV segmentations?
1. Extend the edges of the LV's bounding box. Crop out any RV segmentations that go beyond these edges (e.g. remove any pixels in the RV segmentation that are *below* the LV's lower edge, etc.).
2. Estimate the width of the septum (the wall between the LV and RV). This is done by finding the distance from the bottom-right corner of the RV segmentations (after cropping) to the LV segmentation. The average of this distance across the whole video is our estimated septum width. We use the bottom-right corner specifically because we found it to be the only corner of the RV segmentation that seemed to consistently be in the right area. Now we have the septum width, we can simply crop away any RV pixels that are within that width of the LV's left edge, effectively removing the septum.
3. Crop out ultrasound borders. The segmentations can actually leak outside of the ultrasound borders themselves, but we can easily fix this.
4. For some frames, the RV segmentation can practically vanish (usually because it's just difficult to even idenfity the RV as a human in that frame). We identify frames whose area appears "too small" to be reasonable (more than 1.5 standard deviations below the average area). We then search for the nearest frame whose area is reasonable, and simply copy the reasonable RV segmentation to the current frame. This has mediocre results.

Ultimately, we plan to then use these RV segmentations as annotations to train a separate RV segmentation model.

## Do these techniques guarantee good quality RV segmentations?
**No**. Some of the rougher RV segmentations simply cannot be saved, and only a few videos seem to yield truly "good" quality RV segmentations. For the purposes of training a new RV segmentation model, we should only work with these best RV segmentations, meaning we somehow need to filter out everything else. Doing this by eye is infeasible, so we instead apply a few metrics to automatically filter out the best:
1. RV area should correlate very well with LV area. If it doesn't, then we discard that video.
2. If the initial RV segmentation was already quite good, then our cropping steps would not remove many pixels overall. If a segmentation ends up having a large majority of its pixels cropped out, it's likely that the refined segmentation won't be a very nice shape (e.g. it will have very straight, unnatural edges), so we discard these videos.
3. Does the RV completely vanish in any frames? If so, then it's probably not worth trying to fix that segmentation, so we discard.
4. Other metrics are in the works...


# Original EchoNet-Dynamic README
Below is the README for the original `echonet-dynamic` repo.

EchoNet-Dynamic:<br/>Interpretable AI for beat-to-beat cardiac function assessment
------------------------------------------------------------------------------

EchoNet-Dynamic is a end-to-end beat-to-beat deep learning model for
  1) semantic segmentation of the left ventricle
  2) prediction of ejection fraction by entire video or subsampled clips, and
  3) assessment of cardiomyopathy with reduced ejection fraction.

For more details, see the accompanying paper,

> [**Video-based AI for beat-to-beat assessment of cardiac function**](https://www.nature.com/articles/s41586-020-2145-8)<br/>
  David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich, Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. <b>Nature</b>, March 25, 2020. https://doi.org/10.1038/s41586-020-2145-8

Dataset
-------
We share a deidentified set of 10,030 echocardiogram images which were used for training EchoNet-Dynamic.
Preprocessing of these images, including deidentification and conversion from DICOM format to AVI format videos, were performed with OpenCV and pydicom. Additional information is at https://echonet.github.io/dynamic/. These deidentified images are shared with a non-commerical data use agreement.

Examples
--------

We show examples of our semantic segmentation for nine distinct patients below.
Three patients have normal cardiac function, three have low ejection fractions, and three have arrhythmia.
No human tracings for these patients were used by EchoNet-Dynamic.

| Normal                                 | Low Ejection Fraction                  | Arrhythmia                             |
| ------                                 | ---------------------                  | ----------                             |
| ![](docs/media/0X10A28877E97DF540.gif) | ![](docs/media/0X129133A90A61A59D.gif) | ![](docs/media/0X132C1E8DBB715D1D.gif) |
| ![](docs/media/0X1167650B8BEFF863.gif) | ![](docs/media/0X13CE2039E2D706A.gif ) | ![](docs/media/0X18BA5512BE5D6FFA.gif) |
| ![](docs/media/0X148FFCBF4D0C398F.gif) | ![](docs/media/0X16FC9AA0AD5D8136.gif) | ![](docs/media/0X1E12EEE43FD913E5.gif) |

Installation
------------

First, clone this repository and enter the directory by running:

    git clone https://github.com/echonet/dynamic.git
    cd dynamic

EchoNet-Dynamic is implemented for Python 3, and depends on the following packages:
  - NumPy
  - PyTorch
  - Torchvision
  - OpenCV
  - skimage
  - sklearn
  - tqdm

Echonet-Dynamic and its dependencies can be installed by navigating to the cloned directory and running

    pip install --user .

Usage
-----
### Preprocessing DICOM Videos

The input of EchoNet-Dynamic is an apical-4-chamber view echocardiogram video of any length. The easiest way to run our code is to use videos from our dataset, but we also provide a Jupyter Notebook, `ConvertDICOMToAVI.ipynb`, to convert DICOM files to AVI files used for input to EchoNet-Dynamic. The Notebook deidentifies the video by cropping out information outside of the ultrasound sector, resizes the input video, and saves the video in AVI format. 

### Setting Path to Data

By default, EchoNet-Dynamic assumes that a copy of the data is saved in a folder named `a4c-video-dir/` in this directory.
This path can be changed by creating a configuration file named `echonet.cfg` (an example configuration file is `example.cfg`).

### Running Code

EchoNet-Dynamic has three main components: segmenting the left ventricle, predicting ejection fraction from subsampled clips, and assessing cardiomyopathy with beat-by-beat predictions.
Each of these components can be run with reasonable choices of hyperparameters with the scripts below.
We describe our full hyperparameter sweep in the next section.

#### Frame-by-frame Semantic Segmentation of the Left Ventricle

    echonet segmentation --save_video

This creates a directory named `output/segmentation/deeplabv3_resnet50_random/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - size.csv: estimated size of left ventricle for each frame and indicator for beginning of beat
  - videos: directory containing videos with segmentation overlay

#### Prediction of Ejection Fraction from Subsampled Clips

  echonet video

This creates a directory named `output/video/r2plus1d_18_32_2_pretrained/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - test_predictions.csv: ejection fraction prediction for subsampled clips

#### Beat-by-beat Prediction of Ejection Fraction from Full Video and Assesment of Cardiomyopathy

The final beat-by-beat prediction and analysis is performed with `scripts/beat_analysis.R`.
This script combines the results from segmentation output in `size.csv` and the clip-level ejection fraction prediction in `test_predictions.csv`. The beginning of each systolic phase is detected by using the peak detection algorithm from scipy (`scipy.signal.find_peaks`) and a video clip centered around the beat is used for beat-by-beat prediction.

### Hyperparameter Sweeps

The full set of hyperparameter sweeps from the paper can be run via `run_experiments.sh`.
In particular, we choose between pretrained and random initialization for the weights, the model (selected from `r2plus1d_18`, `r3d_18`, and `mc3_18`), the length of the video (1, 4, 8, 16, 32, 64, and 96 frames), and the sampling period (1, 2, 4, 6, and 8 frames).
