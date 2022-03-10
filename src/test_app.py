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
from skimage import morphology, measure

min_diam = 20
min_rad = min_diam/2
min_area = np.pi*min_rad**2

max_ab_ratio = 2

# title at the top of the app
st.title('Count neurons')

# st.subheader('Upload image to count cells')
uploaded_img = st.file_uploader('Upload image', type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)


def remove_long_cells(segments, max_ab_ratio):
    out = segments.copy()

    reg_props = measure.regionprops(segments)
    a_s = np.array([lab_props.axis_major_length for lab_props in reg_props])
    b_s = np.array([lab_props.axis_minor_length for lab_props in reg_props])
    r_s_ = a_s / b_s

    r_s = np.zeros(r_s_.shape[0] + 1)
    r_s[1:] = r_s_

    too_long = r_s > max_ab_ratio
    too_long_mask = too_long[out]
    out[too_long_mask] = 0

    return out

classify_button = st.button('Find neurons')

if uploaded_img:
    # show loaded image
    st.image(uploaded_img)


if classify_button:
    img = Image.open(uploaded_img)
    img = np.array(img)
    print(img.shape)
    img_gray = rgb2gray(img)
    # create a pretrained stardist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # find cells
    labels, _ = model.predict_instances(normalize(img_gray))
    labels_no_long = remove_long_cells(labels, max_ab_ratio)
    labels_no_long_no_small = morphology.remove_small_objects(labels_no_long, min_size=min_area)
    rendered_labels = render_label(labels_no_long_no_small)

    # show cells
    st.image(rendered_labels, clamp=True)
    num_cells = len(measure.regionprops(labels_no_long_no_small))
    # display cell count
    'Number of cells:', num_cells
