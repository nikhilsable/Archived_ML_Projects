import streamlit as st
import make_prediction
from PIL import Image
import os

def save_uploaded_file(uploadedfile):
  with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return st.success("Saved file :{} in tempDir".format(uploadedfile.name))

### Excluding Imports ###
st.title("Central Cone Stain Checker")

uploaded_file = st.file_uploader("Upload Relative Collector Image..", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    st.write("")
    st.write("Saving File Locally for Classification ...")
    save_uploaded_file(uploaded_file)

    st.write("Classifying...")

    label = make_prediction.main(f"tempDir//{uploaded_file.name}")

    st.subheader(f"Is there central cone stain? : {label[0]}")
    # st.write('%s (%.2f%%)' % (label[1], label[2]*100))

    