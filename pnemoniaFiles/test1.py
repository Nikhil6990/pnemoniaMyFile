import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
img = tf.keras.preprocessing.image.load_img('chest_xray\\test\\PNEUMONIA\person10_virus_35.jpeg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_data = tf.keras.applications.vgg16.preprocess_input(img_array)
img_data = img_data.reshape((1, 224, 224, 3))

prediction = model.predict(img_data)
if prediction[0][0]>prediction[0][1]:  #Printing the prediction of model.
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
print(f'Predictions: {prediction}')
