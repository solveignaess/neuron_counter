# script to set up streamlit app that can load jpg-image

import streamlit as st
from skimage import io
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgba
from stardist.models import StarDist2D
import numpy as np
from PIL import Image

# title at the top of the app
st.title('Count neurons')

# st.subheader('Upload image to count cells')
uploaded_img = st.file_uploader('Upload image', type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)


if uploaded_img:
    # show loaded image
    st.image(uploaded_img)

classify_button = st.button('Find neurons')

if classify_button:
    img = Image.open(uploaded_img)
    img = np.array(img)
    print(img.shape)
    img_gray = rgb2gray(img)
    # create a pretrained stardist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # find cells
    labels, _ = model.predict_instances(normalize(img_gray))
    rendered_labels = render_label(labels)

    # img_with_pred = np.ubyte(gray2rgba(img_gray) + render_label(labels))
    st.image(rendered_labels, clamp=True)

    # convert loaded image to np.ndarray


    # img = io.imread(uploaded_img)
    # fig = plt.figure()
    # plt.subplot(1,1,1)
    # plt.imshow(img)
    # plt.show()

    # img_tau_gray = rgb2gray(img_tau)
    # labels_tau, _ = model.predict_instances(normalize(img_tau_gray))
    # zoom_area = [400, 400, 800, 800]
    # plot_pred_overlay(img_tau_gray, labels_tau, zoom=zoom_area)