# model, prediction, and pre-processing imports
import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

# cleverhans imports
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2

# general imports
import numpy as np
from PIL import Image
import os
import pandas as pd
import datetime


class TestPipeline():

    classifier_types = {"vgg16": VGG16,
                        "resnet50": ResNet50}

    attack_types = {"fgsm": FastGradientMethod,
                    "cwl2": CarliniWagnerL2}

    def __init__(self, base_classifier_type):
        keras.backend.set_image_dim_ordering('tf')
        self.sess = backend.get_session()
        self.classifier_type = base_classifier_type
        self.model = self.classifier_types[self.classifier_type](weights='imagenet')
        self.ch_model = KerasModelWrapper(self.model)
        self.attacks = []
        self.attack_params = []

        target_output = [[0] * 1000]
        target_output[0][870] = 1
        target_output = np.array(target_output)

        self.attacks.append(FastGradientMethod(self.ch_model, sess=self.sess))

        self.attack_params.append({'eps': 0.1 * (152 - (-124)),
                                   'clip_min': -124.,
                                   'clip_max': 152.,
                                   'y_target': target_output})

        self.attacks.append(CarliniWagnerL2(self.ch_model, sess=self.sess))

        self.attack_params.append({'batch_size': 1,
                                   'confidence': 0,
                                   'learning_rate': 0.1,
                                   'binary_search_steps': 5,
                                   'max_iterations': 1000,
                                   'abort_early': True,
                                   'initial_const': 100,
                                   'clip_min': -124,
                                   'clip_max': 152,
                                   'y_target': target_output})

    def load_input_images(self, images_path):
        _, _, image_files = next(os.walk(images_path))
        image_files.sort()

        input_images = []
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            im = image.load_img(image_path, target_size=(224, 224))
            im = image.img_to_array(im)
            im = np.expand_dims(im, axis=0)
            input_img = preprocess_input(np.copy(im))
            input_images.append(input_img)

        return input_images

    def conduct_attack(self, input_img, attacks_to_conduct, target_index=None):
        if target_index is not None:
            target_output = [[0] * 1000]
            target_output[0][target_index] = 1
            target_output = np.array(target_output)
        else:
            target_output = None

        adv_ims = [None]*len(attacks_to_conduct)
        if 'fgsm' in attacks_to_conduct:
            self.attack_params[0]['y_target'] = target_output
            adv_im_fgsm = self.attacks[0].generate_np(input_img, **self.attack_params[0])
            adv_ims[attacks_to_conduct.index('fgsm')] = adv_im_fgsm
        if 'cwl2' in attacks_to_conduct:
            self.attack_params[1]['y_target'] = target_output
            adv_im_cwl2 = self.attacks[1].generate_np(input_img, **self.attack_params[1])
            adv_ims[attacks_to_conduct.index('cwl2')] = adv_im_cwl2
        return adv_ims

    def model_input_to_image(self, model_input):
        img = np.squeeze(model_input.copy())
        mean = [103.939, 116.779, 123.68]

        img[..., 0] += mean[0]
        img[..., 1] += mean[1]
        img[..., 2] += mean[2]

        img = img[..., ::-1]

        img = img.astype(np.uint8)
        return img

    def save_np_image(self, np_image, filename):
        im_to_save = Image.fromarray(np_image)
        im_to_save.save(filename)

    def test(self, images_path, results_directory, targeted=False, save_adv_images=False, enable_print_to_console=False,
             attacks_to_conduct=['fgsm', 'cwl2']):

        if targeted is True:
            target_index = 870
        else:
            target_index = None

        time = datetime.datetime.now()
        results_directory = os.path.join(results_directory,
                                         "./test-results-{}-{}-{}-{}-{}/".format(
                                             time.month, time.day, time.year, time.hour, time.minute))
        os.mkdir(results_directory)
        csv_path = os.path.join(results_directory, "predictions.csv")

        results = {'original_prediction': [], 'target_prediction': [], 'attacked_prediction': [],
                   'attack_type': [], 'classifier': [], 'base_classifier': []}

        input_imgs = self.load_input_images(images_path=images_path)
        i = 0
        for input_img in input_imgs:
            adv_inputs = self.conduct_attack(input_img, attacks_to_conduct, target_index=target_index)
            j = 0
            for adv_input in adv_inputs:
                adv_image = self.model_input_to_image(adv_input)
                if save_adv_images:
                    attack_str = attacks_to_conduct[j]
                    adv_image_file = "adv-{}-{}-{}.jpeg".format(i, attack_str, "targeted" if targeted else "untargeted")
                    adv_image_file_path = os.path.join(results_directory, adv_image_file)
                    self.save_np_image(adv_image, adv_image_file_path)

                original_prediction = decode_predictions(self.model.predict(input_img))
                if targeted is True:
                    target_prediction = decode_predictions(self.attack_params[0]['y_target'])
                else:
                    target_prediction = None
                attacked_prediction = decode_predictions(self.model.predict(adv_input))

                if enable_print_to_console:
                    print("Original:", original_prediction)
                    if targeted:
                        print("Target:", target_prediction)
                    else:
                        print("Not Targeted")
                    print("Attacked:", attacked_prediction)

                results['original_prediction'].append(original_prediction)
                results['target_prediction'].append(target_prediction)
                results['attacked_prediction'].append(attacked_prediction)
                results['attack_type'].append("")
                results['base_classifier'].append(self.classifier_type)
                results['classifier'].append(self.classifier_type)

                j += 1
            i += 1

        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path)


if __name__ == '__main__':
    pipeline = TestPipeline(base_classifier_type='vgg16')
    pipeline.test("./../data/test-pipeline-data/test-images-tiny/",
                  "./",
                  targeted=False,
                  save_adv_images=True,
                  enable_print_to_console=True)
