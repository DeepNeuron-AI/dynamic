import numpy as np
import ffmpeg
import pydicom
# import pydicom.pixel_data_handlers


def convert_dicom_color(dataset: pydicom.Dataset, desired_color: str):
    """Change color of DICOM pixel array *as well as color-space element*.

    This simply ensures that the DICOM's pixel array's color space is consistent
    with the "PhotometricInterpretation" of the DICOM dataset object.

    Parameters
    ----------
    dataset : pydicom.Dataset
        DICOM dataset object whose pixel data you wish to change the color space
        of.
    desired_color : str
        Desired colorspace. One of 'RGB', 'YBR_FULL', 'YBR_FULL_422'.

    Raises
    ------
    ValueError
        _description_
    """
    if "PhotometricInterpretation" not in dataset:
        raise ValueError("This dataset does not have a specified color space")
    # No need to change color in this case
    current_color = dataset["PhotometricInterpretation"].value
    if current_color == desired_color:
        return

    # ds.PixelData = pydicom.pixel_data_handlers.convert_color_space(dataset.pixel_array, current=current_color, desired=desired_color)
    # ds["PhotometricInterpretation"].value = desired_color
    return pydicom.pixel_data_handlers.convert_color_space(dataset.pixel_array, current=current_color, desired=desired_color)


def vidwrite(fn, images, framerate=60, vcodec='libx264'):
    print(fn)
    # Taken from https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-520200981
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()