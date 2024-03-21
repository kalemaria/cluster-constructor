# for loading/processing the images  
from keras.preprocessing.image import load_img

# for everything else
import os
from tqdm import tqdm
import pickle
import numpy as np

def dump_to_pickle_file(parts_feature, dump_path):
    '''
    Dump a dictionary to a pickle file.

    Parameters:
    parts_feature (dict): Dictionary to be saved.
    dump_path (str): Path to the pickle file.
    '''
    with open(dump_path, 'wb') as file:
        pickle.dump(parts_feature, file)

def load_from_pickle_file(pickle_path):
    '''
    Load a dictionary from a pickle file.

    Parameters:
    pickle_path (str): Path to the pickle file.

    Returns:
    dict: Loaded dictionary.
    '''
    with open(pickle_path, 'rb') as file: 
        return pickle.load(file) 

def map_parts_to_features(image_folder, model, preprocess_input, dump_path=None, target_size=None, grayscale=False):
    '''
    Generates a dictionary to store feature vectors of every part image in
    {part_image_filename: [feature_vector], ....} format. Processes images one by one.

    Parameters:
    - image_folder (str): Path where the images are located.
    - model (keras): Loaded pretrained CNN model without the top layer.
    - preprocess_input (function): preprocess_input function of the respective model.
    - dump_path (Optional[str]): Path where to save the feature vectors dictionary as .pkl file, default: None.
    - target_size (Optional[Tuple[int, int]]): Either None (default to original size) or tuple of ints (img_height, img_width), default: None.
    - grayscale (bool): Whether to convert the image to grayscale, default: False.

    Returns:
    Dict[str, np.ndarray]: Dictionary in {part_image_filename (with extension): [feature_vector], ....} format.
    '''
    parts_feature = {}
    for filename in tqdm(
        # Sort filesnames alphanumerically, like part IDs in the CSV file
        sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x)))),
        desc="Processing image"):
        if filename.endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            
            try:
                # resize and load the image as an array
                if grayscale:
                    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
                    img = np.array(img)  # convert from 'PIL.Image.Image' to numpy array
                    img = np.stack((img, )*3, axis=-1) # convert 1 channel to 3 channels
                else:
                    img = load_img(image_path, target_size=target_size)
                    img = np.array(img)
                # reshape the data for the model to (num_of_samples, dim 1, dim 2, channels)
                reshape_img = img.reshape(1, *img.shape)
                # prepare image for model
                preprocess_img = preprocess_input(reshape_img)
                # get the feature vector
                parts_feature[filename] = model.predict(preprocess_img, use_multiprocessing=True, verbose=0)
        
            except Exception as e:
                print(f"Error reading '{filename}': {str(e)}")
                
    # Save the feature vectors dictionary:
    if dump_path is not None:
        dump_to_pickle_file(parts_feature, dump_path)
        
    return parts_feature

def get_feature_matrix(parts_feature):
    '''
    Given part to feature dictionary returned by map_parts_to_features(), reshapes the values
    into a matrix with rows number equal to the number of images and columns number equal to the
    number of features.

    Parameters:
    - parts_feature (Dict[str, np.ndarray]): Feature dictionary as returned by map_parts_to_features().

    Returns:
    np.ndarray: 2D numpy array with the feature matrix.
    '''
    feature_matrix = np.array(
        [value.reshape(-1) for value in parts_feature.values()]
    )
    return feature_matrix