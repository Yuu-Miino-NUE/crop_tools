from crop_tools import crop_face, crop_face_dir

# For a single file
crop_face(input='input/test1.jpg', output='output/htu_test1.jpg', width=250, height=250, margin = 1.8, frame = True, classifier='alt')

# For all files in a directory
crop_face_dir(input_dir='input', output_dir='output', verbose=True, width=200, height=200, margin=1.8, frame=True, classifier='default', maxSize=(100,100))
