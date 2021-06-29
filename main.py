import streamlit as st
from CarDamageDetector import *

# Costruisco il modello
m, sess = create_model(device="/cpu:0",
                       model_dir=r'mrcnn/logs',
                       weights=r'mrcnn/weights/mask_rcnn_damage_0100.h5')

# -----Sidebar----- #
st.sidebar.image(r'figures/MicrosoftTeams-image.png', width=225)
st.sidebar.title('Car Damage Detection')

uploaded_file = st.sidebar.file_uploader("Carica l'immagine", type=['jpg', 'png'])

st.sidebar.markdown(
    "<h5 style='text-align: center; color: black;'>si consiglia refresh del browser ad ogni nuovo file testato ("
    "pulizia cache)</h4>",
    unsafe_allow_html=True)

# -----Page----- #

if uploaded_file is None:
    st.subheader('Carica l\'immagine di un\'auto danneggiata')

else:
    col_sx, col_dx = st.beta_columns(2)
    col_sx.image(uploaded_file, caption='immagine di input')
    output_file = predict(model=m,
                          session=sess,
                          image_file=uploaded_file,
                          output_folder=r'output_images')
    col_dx.image(output_file, caption='immagine di output')
