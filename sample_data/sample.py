from crop_tools import crop_face_to_file, crop_face_dir

# For a single file
crop_face_to_file(
    input="input/test1.jpg",
    output="output/htu_test1.jpg",
    width=250,
    height=250,
    margin=2.0,
    frame=True,
    # minSize=(50, 50),
    # maxSize=(100, 100),
)

# For all files in a directory
crop_face_dir(
    input_dir="input",
    output_dir="output",
    verbose=True,
    width=375,
    height=450,
    margin=1.8,
    frame=True,
    # maxSize=(100, 100),
)
