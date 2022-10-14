import os
import tensorflow as tf
import seaborn as sns

# pixel labels in the video frames
class_names = ['sky', 'building','column/pole', 'road', 'side walk',
               'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void']

BATCH_SIZE = 64

colors = sns.color_palette(None, len(class_names))


def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):
    '''
      Preprocesses the dataset by:
        * resizing the input image and label maps
        * normalizing the input image pixels
        * reshaping the label maps from (height, width, 1) to (height, width, 12)

      Args:
        t_filename (string) -- path to the raw input image
        a_filename (string) -- path to the raw annotation (label map) file
        height (int) -- height in pixels to resize to
          width (int) -- width in pixels to resize to

        Returns:
          image (tensor) -- preprocessed image
          annotation (tensor) -- preprocessed annotation
    '''

    # Convert image and mask files to tensors
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)

    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotation = tf.stack(stack_list, axis=2)

    # Normalize pixels in the input image
    image = image/127.5
    image -= 1

    return image, annotation


def get_dataset_slice_paths(image_dir, label_map_dir):
    '''
      generates the lists of image and label map paths

      Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory

      Returns:
        image_paths (list of strings) -- paths to each image file
        label_map_paths (list of strings) -- paths to each label map
    '''
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths


def get_training_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the training set.

    Args:
      image_paths (list of strings) -- paths to each image file in the train set
      label_map_paths (list of strings) -- paths to each label map in the train set

    Returns:
      tf Dataset containing the preprocessed train set
    '''
    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)

    return training_dataset


def get_validation_dataset(image_paths, label_map_paths):
    '''
    Prepares batches of the validation set.

    Args:
      image_paths (list of strings) -- paths to each image file in the val set
      label_map_paths (list of strings) -- paths to each label map in the val set

    Returns:
      tf Dataset containing the preprocessed validation set
    '''
    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()

    return validation_dataset
