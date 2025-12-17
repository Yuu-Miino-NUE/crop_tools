"""Tools to crop by OpenCV"""

__all__ = ["crop_face", "crop_face_dir"]

from cv2 import (
    imread,
    imshow,
    imwrite,
    waitKey,
    resize,
    cvtColor,
    # CascadeClassifier,
    # COLOR_BGR2GRAY,
    COLOR_GRAY2BGR,
    COLOR_BGRA2BGR,
    rectangle,
    Mat,
    FaceDetectorYN,
)

# from cv2.data import haarcascades
from numpy import array, maximum, minimum, ndarray

# from typing import Literal
import os

EXTENSIONS = ("jpg", "jpeg")
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "face_detection_yunet_2023mar.onnx",
)


class ExtentionError(Exception):
    def __str__(self) -> str:
        return (
            "filename must end with "
            + " or ".join([f"`{ext}`" for ext in EXTENSIONS])
            + "."
        )


def detect_faces_yunet(
    img_bgr: ndarray,
    minSize: tuple[int, int] = (50, 50),
    maxSize: tuple[int, int] | None = None,
):
    # 入力正規化（APIでは必須）
    if img_bgr.ndim == 2:
        img_bgr = cvtColor(img_bgr, COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cvtColor(img_bgr, COLOR_BGRA2BGR)

    H, W = img_bgr.shape[:2]
    IN_W, IN_H = 320, 320

    resized = resize(img_bgr, (IN_W, IN_H))

    det = FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(IN_W, IN_H),
        score_threshold=0.88,
        nms_threshold=0.3,
        top_k=5000,
    )
    det.setInputSize((IN_W, IN_H))

    faces = det.detect(resized)[1]
    if faces is None:
        return []

    sx = W / IN_W
    sy = H / IN_H

    out = []
    for f in faces:
        x, y, fw, fh = f[:4]
        x = int(x * sx)
        y = int(y * sy)
        fw = int(fw * sx)
        fh = int(fh * sy)

        if fw < minSize[0] or fh < minSize[1]:
            continue
        if maxSize is not None and (fw > maxSize[0] or fh > maxSize[1]):
            continue

        out.append((x, y, fw, fh))

    # out に候補が入った後
    out.sort(key=lambda b: b[2] * b[3], reverse=True)
    return out[:1]


def crop_face(
    input: str | Mat,
    width: int = 300,
    height: int = 300,
    margin: float = 1.75,
    show: bool = False,
    frame: bool = False,
    # classifier: Literal[
    #     "default",
    #     "alt",
    #     "alt2",
    # ] = "default",
    minSize: tuple[int, int] = (50, 50),
    maxSize: tuple[int, int] | None = None,
    # scaleFactor: float = 1.1,
    # minNeighbors: int = 4,
) -> Mat | None:
    """Generate face cropped jpg file

    Parameters
    ----------
    input : str
        Input filename.
    width : int, optional
        Width of the output jpg, by default ``300``.
    height : int, optional
        Height of the output jpg, by default ``300``.
    margin : float, optional
        Ratio of margin to the detected face, by default ``1.75``.
    show : bool, optional
        Flag to immediately show the output image, by default ``False``.
    frame : bool, optional
        Flag to show the frame of the detected face, by default ``False``.
    classifier : Literal[ 'default', 'alt', 'alt2', ], optional
        Cascade classifier, by default ``default``.
    minSize : Tuple[int, int], optional
        Minimum possible object size, by default ``(50, 50)``.
    maxSize : Tuple[int, int], optional
        Maximum possible object size, by default ``(100, 100)``.
    scaleFactor : float, optional
        Parameter specifying how much the image size is reduced at each image scale, by default ``1.1``.
    minNeihbors : int
        Parameter specifying how many neighbors each candidate rectangle should have to retain it, by default ``4``.

    Return
    ------
    output : numpy.ndarray
        Output binary

    Raises
    ------
    ExtentionError
        Imcompatible extensions specified.
    FileNotFoundError
        Input file not exist.
    """
    if isinstance(input, str):
        if not input.endswith(EXTENSIONS):
            raise ExtentionError
        else:
            if not os.path.isfile(input):
                raise FileNotFoundError(f"No such file or directory: '{input}'")
        img = imread(input)
    else:
        img = input

    # gray = cvtColor(img, COLOR_BGR2GRAY)
    aspect_ratio = height / width

    # cascade_path = os.path.join(
    #     haarcascades, "haarcascade_frontalface_" + classifier + ".xml"
    # )

    # face_cascade = CascadeClassifier(cascade_path)
    # faces = face_cascade.detectMultiScale(
    #     image=gray,
    #     scaleFactor=scaleFactor,
    #     minNeighbors=minNeighbors,
    #     minSize=minSize,
    #     maxSize=maxSize,
    # )
    faces = detect_faces_yunet(img, minSize=minSize, maxSize=maxSize)

    face = None
    for x, y, w, h in faces:
        center = array([y + h / 2, x + w / 2], dtype=int)
        if frame:
            rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        im_h, im_w, _ = img.shape
        if w * aspect_ratio > h:
            half_diag = array(margin * array([w / 2 * aspect_ratio, w / 2]), dtype=int)
            ymin, xmin = maximum(center - half_diag, array([0, 0]), dtype=int)
            ymax, xmax = minimum(center + half_diag, array([im_h, im_w]), dtype=int)
        else:
            half_diag = array(margin * array([h / 2, h / 2 / aspect_ratio]), dtype=int)
            ymin, xmin = maximum(center - half_diag, array([0, 0]), dtype=int)
            ymax, xmax = minimum(center + half_diag, array([im_h, im_w]), dtype=int)

        if ymin == 0 or ymax == im_h:
            xmax = int(center[1] + (ymax - ymin) / aspect_ratio / 2)
            xmin = int(center[1] - (ymax - ymin) / aspect_ratio / 2)
            print(
                "[Warning] underfull width due to the image height, margin ignored. ",
                end="",
            )
        elif xmin == 0 or xmax == im_w:
            ymax = int(center[0] + (xmax - xmin) * aspect_ratio / 2)
            ymin = int(center[0] - (xmax - xmin) * aspect_ratio / 2)
            print(
                "[Warning] underfull height due to the image width, margin ignored. ",
                end="",
            )
        else:
            pass

        face = resize(img[ymin : ymax + 1, xmin : xmax + 1], (width, height))
        if show:
            imshow("Type any key to quit", face)

    if show:
        waitKey()
    return face


def crop_face_to_file(
    input: str | Mat,
    output: str,
    width: int = 300,
    height: int = 300,
    margin: float = 1.75,
    show: bool = False,
    frame: bool = False,
    # classifier: Literal[
    #     "default",
    #     "alt",
    #     "alt2",
    # ] = "default",
    minSize: tuple[int, int] = (50, 50),
    maxSize: tuple[int, int] | None = None,
    # scaleFactor: float = 1.1,
    # minNeighbors: int = 4,
) -> bool:
    """Generate face cropped jpg file

    Parameters
    ----------
    input : str
        Input filename.
    output : str or None, optional
        Output filename, by default ``None``.
    width : int, optional
        Width of the output jpg, by default ``300``.
    height : int, optional
        Height of the output jpg, by default ``300``.
    margin : float, optional
        Ratio of margin to the detected face, by default ``1.75``.
    show : bool, optional
        Flag to immediately show the output image, by default ``False``.
    frame : bool, optional
        Flag to show the frame of the detected face, by default ``False``.
    classifier : Literal[ 'default', 'alt', 'alt2', ], optional
        Cascade classifier, by default ``default``.
    minSize : Tuple[int, int], optional
        Minimum possible object size, by default ``(50, 50)``.
    maxSize : Tuple[int, int], optional
        Maximum possible object size, by default ``(100, 100)``.
    scaleFactor : float, optional
        Parameter specifying how much the image size is reduced at each image scale, by default ``1.1``.
    minNeihbors : int
        Parameter specifying how many neighbors each candidate rectangle should have to retain it, by default ``4``.

    Raises
    ------
    ExtentionError
        Imcompatible extensions specified.
    FileNotFoundError
        Input file not exist.
    """
    if not output.endswith(EXTENSIONS):
        output += ".jpg"

    face = crop_face(
        input=input,
        width=width,
        height=height,
        margin=margin,
        show=show,
        frame=frame,
        # classifier=classifier,
        minSize=minSize,
        maxSize=maxSize,
        # scaleFactor=scaleFactor,
        # minNeighbors=minNeighbors,
    )
    if face is None:
        return False
    return imwrite(output, face)


def crop_face_dir(
    input_dir: str = "input",
    output_dir: str = "output",
    verbose: bool = False,
    **options,
):
    """Generate face cropped jpg files in directory

    Parameters
    ----------
    input_dir : str, optional
        Input directory including jpeg files, by default ``input``.
    output_dir : str, optional
        Output directory to put the generated jpeg files, by default ``output``.
    verbose : bool, optional
        Flag to turn on the verbose mode, by default ``False``
    **options:
        Options to pass to ``crop_face`` function.
    """
    dir_list = os.listdir(input_dir)
    jpg_list = [f for f in dir_list if f.endswith(EXTENSIONS)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        print(f"{len(jpg_list)} jpg files found.")

    for jpg in jpg_list:
        if verbose:
            print(f"Processing `{jpg}`... ", end="")
        ret = crop_face_to_file(
            input_dir + "/" + jpg, output_dir + "/" + jpg, **options
        )
        if verbose:
            print(f"Done." if ret else "No face detected.")
