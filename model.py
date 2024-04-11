import keras
import cv2
import os
import tensorflow as tf
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import streamlit as st
import urllib.request



MAX_FILE_SIZE = 5 * 1024 * 1024  # This is the max size of image that use can upload


@st.cache_resource
def load_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/srogoobeer/C964/blob/master/model.h5', 'model.h5')
    return keras.models.load_model('model.h5')


with open('binarizer.pkl', 'rb') as f:    # Loading the binarizer file to inverse tranform the labels to convert them back into string
    label_binarizer = pickle.load(f)

class_labels = list(label_binarizer.classes_)


def get_prediction(img):        #Function will prepare the image according to the model and feed it to the model and it will also return the prediction after inverse traforming it from the binarizer
    if img is None:
        return "No Image Uploaded"
    else:
        image = cv2.resize(img, (264,264))   
        image_array = img_to_array(image)
        image_array = np.array(image_array/255.0)
        image_array = np.expand_dims(image_array, axis=0)
        model = load_model()
        predictions = model.predict(image_array)
        predicted_class_label = label_binarizer.inverse_transform(predictions)
        return predicted_class_label






st.set_page_config(layout="wide")

#Main Content
st.title('Leaf Identification App:leaves:')
st.markdown("\n")
st.write("Upload an image of a leaf to identify its species using our advanced machine learning model. This model can predict 24 different plant leaves species with an accuracy over 90%.")
st.markdown("\n")


#Side Bar Content
st.sidebar.image("crop_logo.png")
st.sidebar.title("Upload an Image :camera:")
col1, col2 = st.columns(2)
option = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


#Added a drop down list of the species that our model can predict
with st.sidebar.expander("Species Included :four_leaf_clover:"):
    species_list = [
        "Apple", "Arjun", "Banana", "Basil", "Cherry", "Chinar", "Chinkapin oak",
        "Coffee", "Corn", "Cucumber", "Japanese Maple", "Java Plum", "Lemon", "Peach",
        "Pomegranate", "Potato", "Red Buckeye", "Rice", "Soybean", "Strawberry",
        "Sugi", "Tea", "Tomato", "Wheat"
    ]
    for species in species_list:
        st.write(f"- {species}")


# Apply custom styles
st.markdown(
    """
    <style>
    [data-testid="stSidebarUserContent"]{
        margin:0px;
        padding:30px;
    }
    [data-testid="stVerticalBlock"]{
        width:10%;
    }
    [data-testid="stVerticalBlock"]{
        width:100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)


sticky_note_style = """
    <style>
        .sticky-note {
            background-color: rgba(144, 238, 144, 0.3);
            padding: 15px;
            border-radius: 10px;
        }
    </style>
"""

sticky_note_style_col = """
    <style>
        .sticky-note-col {
            background-color: rgba(144, 238, 144, 0.3);
            padding: 15px;
            border-radius: 10px;
        }
    </style>
"""

col2.markdown(sticky_note_style_col, unsafe_allow_html=True)



if option is not None:
    if option.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else: 
        image = cv2.imdecode(np.fromstring(option.read(), np.uint8), 1)
        prediction = get_prediction(image)
        col1.image(option,width =400)
        col2.markdown("\n")
        col2.markdown("\n")
        col2.markdown("\n")
        col2.markdown("\n")
        col2.markdown('<div class="sticky-note-col"><h1>This is a {} Image</h1></div>'.format(prediction[0]), unsafe_allow_html=True)
        #st.image(option,width = 100)
        #st.write(prediction[0])
else:
    col1.image("peach.jpg",width=400)
    col2.markdown("\n")
    col2.markdown("\n")
    col2.markdown("\n")
    col2.markdown("\n")
    col2.markdown('<div class="sticky-note-col"><h1>This is a Peach Image</h1></div>', unsafe_allow_html=True)

    #st.image("/Users/apple/Desktop/Capstone/ML Model/Peach_Test.jpg")
    #st.write("This is a peach Image")

    


st.sidebar.markdown(sticky_note_style, unsafe_allow_html=True)
st.sidebar.markdown('<div class="sticky-note">'
            '<h3>Note</h3>'  
            '<ul>'
            '<li>Upload pictures of size less than 5MB</li>'
            '<li>Pictures should covering full leaf</li>'
            '<li>Take pictures in bright light</li>'
            '<li>Avoid uploading damaged leaf</li>'
            '<li>Only upload pictures of mentioned species</li>'
            '</ul>'
            '</div>', unsafe_allow_html=True)
