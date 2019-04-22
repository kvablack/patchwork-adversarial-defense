# model, prediction, and pre-processing imports
import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

import tensorflow as tf

# cleverhans imports
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2

# defense imports
from testing import defenses

# general imports
import numpy as np
from PIL import Image
import os
import pandas as pd
import datetime
import json
import logging
from typing import Dict, Tuple


class TestPipeline:
    classifier_types = {"vgg16": VGG16,
                        "resnet50": ResNet50}

    attack_types = {"fgsm": FastGradientMethod,
                    "cwl2": CarliniWagnerL2}

    classifier_input_shapes: Dict[str, Tuple[int, int]] = {"vgg16": (224, 224),
                                                           "resnet50": (224, 224)}

    defense_types = {"dummy": (defenses.dummy_perturb, defenses.dummy_unperturb),
                     "gaussian_noise_denoise": (defenses.gaussian_noise, defenses.gaussian_denoise),
                     "blur_deblur": (defenses.blur, defenses.deblur_gan),
                     "interpolation_superresolution": (defenses.bicubic_interpolation, defenses.srgan),
                     "patchwork": (defenses.create_patchwork, defenses.patchwork_gan)}

    def __init__(self, base_classifier_type, additional_classifiers=None):
        keras.backend.set_image_dim_ordering('tf')
        self.sess = backend.get_session()

        self.classifier_type = base_classifier_type
        self.classifier = self.get_classifier(self.classifier_type)
        self.additional_classifiers = {}
        if additional_classifiers is not None:
            for additional_classifier in additional_classifiers:
                self.additional_classifiers[additional_classifier] = self.get_classifier(additional_classifier)
        self.ch_model = KerasModelWrapper(self.classifier)

        self.attacks = {}
        self.attack_params = {}

        target_output = None

        self.attacks['fgsm'] = FastGradientMethod(self.ch_model, sess=self.sess)

        self.attack_params['fgsm'] = {'eps': 0.003 * (152 - (-124)),
                                      'clip_min': -124.,
                                      'clip_max': 152.,
                                      'ord': np.inf,
                                      'y_target': target_output}

        self.attacks['cwl2'] = CarliniWagnerL2(self.ch_model, sess=self.sess)

        self.attack_params['cwl2'] = {'batch_size': 1,
                                      'confidence': 0,
                                      'learning_rate': 0.1,
                                      'binary_search_steps': 5,
                                      'max_iterations': 1000,
                                      'abort_early': True,
                                      'initial_const': 100,
                                      'clip_min': -124,
                                      'clip_max': 152,
                                      'y_target': target_output}

    @staticmethod
    def get_classifier(classifier_type):
        image_input = keras.layers.Input(shape=(None, None, 3))
        resized = keras.layers.Lambda(lambda image_input: tf.image.resize(image_input,
                                                                          size=TestPipeline.classifier_input_shapes[
                                                                              classifier_type],
                                                                          method=tf.image.ResizeMethod.BICUBIC,
                                                                          align_corners=True))(image_input)
        classifier = TestPipeline.classifier_types[classifier_type](weights='imagenet', input_tensor=resized)
        return classifier

    @staticmethod
    def load_input_images(images_path):
        _, _, image_files = next(os.walk(images_path))
        image_files.sort()

        input_images = []
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            im = image.load_img(image_path)
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

        adv_ims = {}
        for attack_type in attacks_to_conduct:
            self.attack_params[attack_type]['y_target'] = target_output
            adv_im = self.attacks[attack_type].generate_np(input_img, **self.attack_params[attack_type])
            adv_ims[attack_type] = adv_im
            self.attacks[attack_type] = TestPipeline.attack_types[attack_type](self.ch_model, sess=self.sess)

        return adv_ims

    @staticmethod
    def model_input_to_image_array(model_input):
        img = np.squeeze(model_input.copy())
        mean = [103.939, 116.779, 123.68]

        img[..., 0] += mean[0]
        img[..., 1] += mean[1]
        img[..., 2] += mean[2]

        img = img[..., ::-1]

        img = img.astype(np.uint8)
        return img

    @staticmethod
    def image_array_to_model_input(image):
        model_input = np.copy(image)
        model_input = np.expand_dims(model_input, axis=0)
        model_input = preprocess_input(model_input)
        return model_input

    @staticmethod
    def save_np_image(np_image, filename):
        im_to_save = Image.fromarray(np_image)
        im_to_save.save(filename)

    @staticmethod
    def save_params(attack_params, results_directory):
        for attack, params in attack_params.items():
            if params['y_target'] is not None:
                params['y_target'] = int(np.argmax(params['y_target']))
        TestPipeline.save_as_json(attack_params, results_directory, "attack-params.json")

    @staticmethod
    def save_as_json(to_save, results_directory, file_name):
        json_obj = json.dumps(to_save)
        json_file = open(os.path.join(results_directory, file_name), "w")
        json_file.write(json_obj)
        json_file.close()

    def test(self, images_path, results_directory, target_index=None, save_adv_images=False, enable_print_to_console=False,
             attacks_to_conduct=['fgsm', 'cwl2'], defenses_to_use=["dummy"]):

        targeted = target_index is not None

        time = datetime.datetime.now()
        results_directory = os.path.join(results_directory,
                                         "./test-results-{}-{}-{}-{}-{}-{}/".format(
                                             time.month, time.day, time.year, time.hour, time.minute, time.second))
        os.mkdir(results_directory)
        csv_path = os.path.join(results_directory, "predictions.csv")

        results = {'original_prediction': [], 'target_prediction': [],
                   'attacked_prediction': [], 'defended_prediction': [],
                   'attack_type': [], 'defense_name': [], 'classifier': [], 'attacked_classifier': [],
                   'top_1_accurate_undefended': [], 'top_5_accurate_undefended': [],
                   'top_1_accurate_defended': [], 'top_5_accurate_defended': [],
                   'image_size': [], 'defended_prediction_unattacked': [], 'defended_image_preserved': []}

        input_imgs = self.load_input_images(images_path=images_path)

        i = 0
        for input_img in input_imgs:
            adv_inputs = self.conduct_attack(input_img, attacks_to_conduct, target_index=target_index)
            j = 0
            for attack_type, adv_input in adv_inputs.items():
                adv_image = TestPipeline.model_input_to_image_array(adv_input)
                adv_input = TestPipeline.image_array_to_model_input(adv_image)
                if save_adv_images:
                    attack_str = attacks_to_conduct[j]
                    adv_image_file = "adv-{}-{}-{}.jpeg".format(i, attack_str, "targeted" if targeted else "untargeted")
                    adv_image_file_path = os.path.join(results_directory, adv_image_file)
                    self.save_np_image(adv_image, adv_image_file_path)

                for defense in defenses_to_use:
                    perturb, unperturb = TestPipeline.defense_types[defense]

                    result = self.get_result(input_img, adv_input, targeted, attack_type,
                                             perturb, unperturb, defense,
                                             self.classifier_type, self.classifier, enable_print_to_console)

                    for feature, value in result.items():
                        results[feature].append(value)

                    for classifier_type, classifier in self.additional_classifiers.items():
                        result = self.get_result(input_img, adv_input, targeted, attack_type,
                                                 perturb, unperturb,
                                                 classifier_type, classifier, enable_print_to_console)
                        for feature, value in result.items():
                            results[feature].append(value)

                j += 1
            i += 1

        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path)
        TestPipeline.analyze_results(results_df, results_directory)
        TestPipeline.save_params(self.attack_params, results_directory)

    @staticmethod
    def defend(perturb, unperturb, img, convert=True):
        im = np.copy(img)
        if convert:
            im = TestPipeline.model_input_to_image_array(im)
        defended_im = unperturb(perturb(im))
        if convert:
            defended_im = np.expand_dims(defended_im, axis=0)
            defended_im = preprocess_input(defended_im)
        return defended_im

    def get_result(self, input_img, adv_input, targeted, attack_type,
                   perturb, unperturb, defense_name,
                   classifier_type, classifier, enable_print_to_console):

        result = {}

        original_prediction = decode_predictions(classifier.predict(input_img))

        if targeted is True:
            target_prediction = decode_predictions(self.attack_params['fgsm']['y_target'])
        else:
            target_prediction = None

        attacked_prediction = decode_predictions(classifier.predict(adv_input))

        defended_image = TestPipeline.defend(perturb, unperturb, adv_input, convert=True)
        defended_prediction = decode_predictions(classifier.predict(defended_image))

        defended_image_unattacked = TestPipeline.defend(perturb, unperturb, input_img)
        defended_prediction_unattacked = decode_predictions(classifier.predict(defended_image_unattacked))

        result['original_prediction'] = original_prediction
        result['target_prediction'] = target_prediction
        result['attacked_prediction'] = attacked_prediction
        result['defended_prediction'] = defended_prediction
        result['attack_type'] = attack_type
        result['defense_name'] = defense_name
        result['classifier'] = classifier_type
        result['attacked_classifier'] = self.classifier_type
        result['top_1_accurate_undefended'] = original_prediction[0][0][0] == attacked_prediction[0][0][0]
        result['top_1_accurate_defended'] = original_prediction[0][0][0] == defended_prediction[0][0][0]
        result['top_5_accurate_undefended'] = \
            sum([original_prediction[0][0][0] == attacked_prediction[0][i][0]
                 for i in range(len(attacked_prediction[0]))]) > 0
        result['top_5_accurate_defended'] = \
            sum([original_prediction[0][0][0] == defended_prediction[0][i][0]
                 for i in range(len(attacked_prediction[0]))]) > 0
        result['image_size'] = np.shape(input_img)
        result['defended_prediction_unattacked'] = defended_prediction_unattacked
        result['defended_image_preserved'] = defended_prediction_unattacked[0][0][0] == original_prediction[0][0][0]

        if enable_print_to_console:
            print("Classifier:", classifier_type)
            print("Attack Type:", attack_type)
            print("Defense Name:", defense_name)
            print("Original:", original_prediction)
            if targeted:
                print("Target:", target_prediction)
            else:
                print("Not Targeted")
            print("Attacked:", attacked_prediction)
            print("Defended:", defended_prediction)
            print("Defended Unattacked:", defended_prediction_unattacked)

        return result

    @staticmethod
    def analyze_results(results_df, results_directory):
        attack_sucess_rates = TestPipeline.get_attack_success_rates(results_df)
        TestPipeline.save_as_json(attack_sucess_rates, results_directory, "attack-success-rates.json")

        defense_sucess_rates = TestPipeline.get_defense_success_rates(results_df)
        TestPipeline.save_as_json(defense_sucess_rates, results_directory, "defense-success-rates.json")

        defense_preservation_rates = TestPipeline.get_defense_preservation_rates(results_df)
        TestPipeline.save_as_json(defense_preservation_rates, results_directory, "defense-preservation-rates.json")

    @staticmethod
    def get_attack_success_rates(results_df):
        rates = {}

        if len(results_df['top_1_accurate_undefended']) == 0:
            rates['overall'] = "undefined"
            for attack in results_df['attack_type'].unique():
                rates[attack] = "undefined"
        else:
            rates['overall'] = len(results_df[results_df['top_1_accurate_undefended'] == False]) \
                               / len(results_df['top_1_accurate_undefended'])

            for attack in results_df['attack_type'].unique():
                results_subset = results_df[results_df['attack_type'] == attack]
                rates[attack] = len(results_subset[results_subset['top_1_accurate_undefended'] == False]) \
                               / len(results_subset['top_1_accurate_undefended'])

        return rates

    @staticmethod
    def get_defense_success_rates(results_df):
        rates = {}

        for defense in results_df['defense_name'].unique():
            defense_results_df_unfiltered = results_df[results_df['defense_name'] == defense]
            defense_results_df = defense_results_df_unfiltered[defense_results_df_unfiltered['top_1_accurate_undefended'] == False]
            sub_rates = {}
            if len(defense_results_df['top_1_accurate_defended']) == 0:
                sub_rates['overall'] = "undefined"
                for attack in results_df['attack_type'].unique():
                    sub_rates[attack] = "undefined"
            else:
                sub_rates['overall'] = len(defense_results_df[defense_results_df['top_1_accurate_defended'] == True]) \
                                   / len(defense_results_df['top_1_accurate_defended'])

                for attack in results_df['attack_type'].unique():
                    defense_results_subset = defense_results_df[defense_results_df['attack_type'] == attack]
                    sub_rates[attack] = len(defense_results_subset[defense_results_subset['top_1_accurate_defended'] == True]) \
                                    / len(defense_results_subset['top_1_accurate_defended'])
            rates[defense] = sub_rates

        return rates

    @staticmethod
    def get_defense_preservation_rates(results_df):
        rates = {}
        if len(results_df['defended_image_preserved']) == 0:
            rates['overall'] = "undefined"
            for defense in results_df['defense_name'].unique():
                rates[defense] = "undefined"
        else:
            rates['overall'] = len(results_df[results_df['defended_image_preserved'] == True]) \
                               / len(results_df['defended_image_preserved'])

            for defense in results_df['defense_name'].unique():
                results_subset = results_df[results_df['defense_name'] == defense]
                rates[defense] = len(results_subset[results_subset['defended_image_preserved'] == True]) \
                                / len(results_subset['defended_image_preserved'])

        return rates

if __name__ == '__main__':
    tf.get_logger().setLevel(logging.ERROR)
    pipeline = TestPipeline(base_classifier_type='vgg16', additional_classifiers=None)
    pipeline.test("./../data/test-pipeline-data/test-images-super-tiny/",
                  "./../data/test-pipeline-data/results/",
                  # target_index=870,
                  save_adv_images=True,
                  enable_print_to_console=True,
                  attacks_to_conduct=['fgsm', 'cwl2'],
                  defenses_to_use=['gaussian_noise_denoise'])
