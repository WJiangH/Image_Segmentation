import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import random, string

# Dataset Preprocessing Utilities
def random_flip(input_image, input_mask):
    '''does a random flip of the image and mask'''
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


def normalize(input_image, input_mask):
    '''
    normalizes the input image pixel values to be from [0,1].
    subtracts 1 from the mask labels to have a range from [0,2]
    '''
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    '''resizes, normalizes, and flips the training data'''
    input_image = tf.image.resize(datapoint['image'], (128, 128), method='nearest')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')
    input_image, input_mask = random_flip(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    '''resizes and normalizes the test data'''
    input_image = tf.image.resize(datapoint['image'], (128, 128), method='nearest')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# Image process utilities
# class list of the mask pixels
class_names = ['pet', 'background', 'outline']

def display_with_metrics(display_list, iou_list, dice_score_list, save_img=False):
    '''displays a list of images/masks and overlays a list of IOU and Dice Scores'''

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) \
                         in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]

    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) \
                           for idx, iou, dice_score in metrics_by_id]

    display_string = "\n\n".join(display_string_list)

    display(display_list, ["Image", "Predicted Mask", "True Mask"], display_string=display_string, save_img=save_img)



def display(display_list,titles=[], display_string=None, save_img=False):
    '''displays a list of images/masks'''

    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if display_string and i == 1:
            plt.xlabel(display_string, fontsize=12)

        img_arr = tf.keras.preprocessing.image.array_to_img(display_list[i])

        plt.imshow(img_arr)

    if save_img:
        name = ''.join(random.choices(string.ascii_lowercase, k= 5)) + '.jpg'
        plt.savefig(name, bbox_inches='tight')

    plt.show()


def show_image_from_dataset(dataset, save_img=False):
    '''displays the first image and its mask from a dataset'''

    for image, mask in dataset.take(1):
        sample_image, sample_mask = image, mask


    display([sample_image, sample_mask], titles=["Image", "True Mask"], save_img=save_img)


def plot_metrics(metric_name, title, ylim=5):
    '''plots a given metric from the model history'''
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(model_history.history[metric_name],color='blue',label=metric_name)
    plt.plot(model_history.history['val_' + metric_name],color='green',label='val_' + metric_name)


#compute the IOU and Dice Score.
def class_wise_metrics(y_true, y_pred):
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001
    for i in range(3):

        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)

        dice_score =  2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score
