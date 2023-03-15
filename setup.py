import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="crop_tools",
    version="0.0.1",
    author="Yuu-Miino-NUE",
    description="Tools to crop by OpenCV",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Yuu-Miino-NUE/crop_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.10',
    install_requires = [
        'opencv-python'
    ]
)