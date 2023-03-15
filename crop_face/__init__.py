__all__ = [
    'crop_face', 'crop_face_dir'
]

from cv2 import imread, imshow, imwrite, waitKey, resize, cvtColor, CascadeClassifier, COLOR_BGR2GRAY, rectangle
from cv2.data import haarcascades
from numpy import array, maximum, minimum
from typing import Literal
import os

EXTENSIONS = ('jpg', 'jpeg')

class ExtentionError(Exception):
    def __str__(self) -> str:
        return ('filename must end with '+' or '.join([f'`{ext}`' for ext in EXTENSIONS])+'.')

def crop_face(
        input: str,
        output: str | None = None,
        width: int = 300,
        height: int = 300,
        margin: float = 1.75,
        show: bool = False,
        frame: bool = False,
        classifier: Literal[
            'default',
            'alt',
            'alt2',
        ] = 'default'
    ):
    if not input.endswith(EXTENSIONS):
        raise ExtentionError
    else:
        if not os.path.isfile(input):
            raise FileNotFoundError(f"No such file or directory: '{input}'")

    if output is None:
        output = ''.join(input.split('.')[:-1])+'-out.jpg'
    else:
        if not output.endswith(EXTENSIONS):
            output += '.jpg'

    img = imread(input)
    gray = cvtColor(img, COLOR_BGR2GRAY)
    aspect_ratio = height / width

    cascade_path = os.path.join(
        haarcascades, 'haarcascade_frontalface_'+classifier+'.xml'
    )

    face_cascade = CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        center = array([y+h/2, x+w/2], dtype=int)
        if frame: rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

        im_h, im_w, _ = img.shape
        if w * aspect_ratio > h:
            half_diag = array(margin * array([w/2*aspect_ratio, w/2]), dtype=int)
            ymin, xmin = maximum(center - half_diag, array([0, 0]), dtype=int)
            ymax, xmax = minimum(center + half_diag, array([im_h, im_w]), dtype=int)
        else:
            half_diag = array(margin * array([h/2, h/2/aspect_ratio]), dtype=int)
            ymin, xmin = maximum(center - half_diag, array([0, 0]), dtype=int)
            ymax, xmax = minimum(center + half_diag, array([im_h, im_w]), dtype=int)

        if ymin == 0 or ymax == im_h:
            xmax = int(center[1] + (ymax - ymin) / aspect_ratio / 2)
            xmin = int(center[1] - (ymax - ymin) / aspect_ratio / 2)
            print('[Warning] underfull width due to the image height, margin ignored. ', end='')
        elif xmin == 0 or xmax == im_w:
            ymax = int(center[0] + (xmax - xmin) * aspect_ratio / 2)
            ymin = int(center[0] - (xmax - xmin) * aspect_ratio / 2)
            print('[Warning] underfull height due to the image width, margin ignored. ', end='')
        else:
            pass

        face = resize(img[ymin:ymax+1, xmin:xmax+1], (width, height))
        if show: imshow("Type any key to quit", face)
        imwrite(output, face)

    if show: waitKey()

def crop_face_dir(
        input_dir: str = 'input',
        output_dir: str = 'output',
        verbose: bool = False,
        **options
    ):
    dir_list = os.listdir(input_dir)
    jpg_list = [f for f in dir_list if f.endswith(EXTENSIONS)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose: print(f'{len(jpg_list)} jpg files found.')

    for jpg in jpg_list:
        if verbose: print(f'Processing `{jpg}`... ', end='')
        crop_face(input_dir+'/'+jpg, output_dir+'/'+jpg, **options)
        if verbose: print('Done')