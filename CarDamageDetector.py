import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mrcnn.model as modellib
import mrcnn.visualize as visualize
import cv2
import custom
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from keras import backend as K


@st.cache(show_spinner=False, allow_output_mutation=True)
def create_model(device, model_dir, weights):
    # Model configuration
    config = custom.CustomConfig()
    # TensorFlow model
    with tf.device(device):
        print('Inizializzazione modello')
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)
    print('Caricamento dei pesi')
    model.load_weights(weights, by_name=True)
    model.keras_model._make_predict_function()
    session = K.get_session()
    return model, session


@st.cache(show_spinner=False)
def predict(model, session, image_file, save_image=True, output_folder=''):
    nparr = np.fromstring(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run object detection
    K.set_session(session)
    results = model.detect([image], verbose=0)

    # Save labeled image
    if save_image:
        filename, ext = os.path.splitext(os.path.basename(image_file.name))
        output_filename = filename + '_PREDICT' + ext
        _save_predicted_image(image, results, os.path.join(output_folder, output_filename))
        return os.path.join(output_folder, output_filename)

    return results


def _save_predicted_image(image, results, output_path):
    r = results[0]
    f, ax = plt.subplots(1, figsize=(image.shape[0] / 100, image.shape[1] / 100), frameon=False, dpi=100)
    ax.axis('off')
    visualize.display_instances(image=image,
                                boxes=r['rois'],
                                masks=r['masks'],
                                class_ids=r['class_ids'],
                                class_names=['bg', 'Danno'],
                                scores=r['scores'],
                                ax=ax)
    f.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0)
