import tensorflow as tf
import numpy as np

import streamlit as st


from PIL import Image, ImageOps

from tensorflow.keras.preprocessing.image import ImageDataGenerator

@st.cache(allow_output_mutation=True)
def load_data():
	#PATH_LOADMODEL = "C:/Users/Admin/Downloads/transfer_learning_model"
	loaded_model = tf.keras.models.load_model("my_model2.h5")

	return loaded_model



ld_model = load_data()



st.header('***CAN WE DETECT COVID-19 INFECTION BY CHEST RADIOGRAPH?***')
st.write('Tun Thunchawin')
st.write('   Of course, Applying deep learning is one of the possible approaches to detect covid-19 infection. Image transfer learning is the technique that I used to create the model, because the number of images we have is somewhat limited; thus, the technique whereby a neural network model is first trained on a problem similar to the problem that is being solved is suitable for our issue.')
st.write('')
st.write('')

st.image('transfer2.png')
st.write('Source:https://www.mdpi.com/2072-4292/12/11/1780')

st.header('***EFFORTLESS TO USE JUST INSERT AN IMAGE DOWN BELOW 👇👇👇***')






#st.sidebar.write('Hello')



st.image("corona.jpg")

st.title("")
st.header("")

st.markdown("**UPLOAD YOUR RADIOGRAPH HERE !!!:sunglasses:**")
st.markdown("**THEN JUST WAIT PATIENTLY FOR THE RESULT!!! 📲**")





upload_file = st.file_uploader("Choose an image",type=["png","jpg","jpeg"])





def classifying3(x,y):

	if upload_file is not None:
		img= Image.open(x).convert('RGB')

		st.subheader("The photo you've selected")
		st.image(img, caption='Uploaded Chest X-ray', use_column_width=True)
		st.write("")
		loading = st.text("Classified...✔️")

		data = np.ndarray(shape=(1,256,256,3),dtype=np.float32)
		image=img
		size = (256,256)


		image = image.resize(size)
		image_array = np.array(image)

		img_tf = tf.convert_to_tensor(image_array)
		img_tf = tf.image.resize(img_tf, [256, 256])
		img_tf = tf.reshape(img_tf, [1, img_tf.shape[0], img_tf.shape[1], img_tf.shape[2]])

		pred = y.predict(img_tf)


		prediction = tf.nn.sigmoid(pred)

		x = np.where(prediction < 0.5, 0, 1)

		if x ==0:
			st.write('**Result: Detected COVID-19...🏨😟**')
			st.write('**Noted: the model has 89 percent accuracy, you are required to recieve further diagnosis.**')
			st.write(abs(1-prediction[0][0])*100)
			st.image('diagnosis.png')
			st.write('**Noted: The model was trained at balanced data.**')


		else:
			st.write('**Result: Undetected COVID-19...🏘️🤗**')			
			st.write('**Noted: the model has 89 percent accuracy, you are required to recieve further diagnosis.**')
			st.write(prediction[0][0])
			st.image('diagnosis.png')
			st.write('**Noted: The model was trained at balanced data.**')



classifying3(upload_file,ld_model)





	








