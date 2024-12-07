from tensorflow.keras.models import load_model
model = load_model("E:/Rice crop Health/rice_disease_model2.h5")
print(model.summary())
