from captcha.image import ImageCaptcha
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import random
import string


# 驗證碼包含字母和數字，共 36 種可能性
# 數字 string.digits
# 小寫字母 string.ascii_lowercase
# 大寫字母 string.ascii_uppercase
# 所有大小寫字母 string.ascii_letters
# 驗證碼長度為 4

characters = string.digits + string.ascii_letters
num_classes = len(characters)
captcha_length = 6
batch_size = 32
num_batches = 100


# 使用 ImageCaptcha 生成驗證碼圖片
def generate_captcha():
    captcha_text = ''.join(random.choices(characters, k=captcha_length))
    image_captcha = ImageCaptcha(width=160, height=60)
    captcha_image = image_captcha.generate_image(captcha_text)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def draw_pict(captcha_text,captcha_image):
    plt.imshow(captcha_image)
    plt.title(captcha_text)
    show=plt.show()
    return show

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

# 將圖片轉換成灰階
def preprocess_image(captcha_image):
    captcha_image = tf.image.rgb_to_grayscale(captcha_image)
    captcha_image = captcha_image / 255
    return captcha_image

# 將驗證碼轉換成 one-hot 編碼
def one_hot_encode(text):
    return np.array([np.eye(num_classes)[characters.index(c)] for c in text],dtype=np.float32)

def generate_data(batch_size):
    CC=[]
    captcha_texts = []
    images_captcha = []
    for i in range(batch_size):
        captcha_text, captcha_image = generate_captcha()
        CC.append(captcha_text)
        captcha_image=preprocess_image(captcha_image)
        captcha_text=one_hot_encode(captcha_text)
        images_captcha.append(captcha_image)
        captcha_texts.append(captcha_text)
    return CC,np.array(captcha_texts), np.array(images_captcha)



#
# 產生一個驗證碼圖片並顯示
train_CC,train_captcha,train_images = generate_data(1000)
validation_CC,validation_captcha,validation_images = generate_data(50)
test_CC,test_captcha,test_images = generate_data(50)


# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(60,160,1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(captcha_length * num_classes, activation='softmax'),
    tf.keras.layers.Reshape((captcha_length, num_classes))
])

# 優化器
# model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])

# 訓練模型
history = model.fit(
    train_images,
    train_captcha,
    epochs= 200,
    batch_size = 100,
    # validation_data=(validation_images, validation_captcha),
    validation_split=0.2,
    # verbose=2,
    validation_steps=10,
)

# 使用模型預測驗證碼
# prediction = model.predict(np.array([images_captcha][0]))[0]
# predicted_captcha = ''.join([characters[np.argmax(prediction[i])] for i in range(captcha_length)])
# print(f"Model: Actual captcha: {CC[0]}, predicted captcha: {predicted_captcha}")


def predict_captcha(model, image):
    prediction = model.predict(np.array([image]))[0]
    predicted_captcha = ''.join([characters[np.argmax(prediction[i])] for i in range(captcha_length)])
    return predicted_captcha

for i in range(len(train_images)):
    predicted_captcha = predict_captcha(model, train_images[i])
    actual_captcha = ''.join([characters[np.argmax(train_captcha[i][j])] for j in range(captcha_length)])
    print(f"Actual captcha: {actual_captcha}, predicted captcha: {predicted_captcha}")

scores = model.evaluate(train_images, train_captcha)
print('accuracy',scores[1])   
plot_train_history(history,'accuracy','val_accuracy')
plot_train_history(history,'loss','val_loss')
# for i in range(len(test_images)):
#     predicted_captcha = predict_captcha(model, test_images[i])
#     actual_captcha = ''.join([characters[np.argmax(test_captcha[i][j])] for j in range(captcha_length)])
#     print(f"Actual captcha: {actual_captcha}, predicted captcha: {predicted_captcha}")

#######################################################################################################
# captcha_text, captcha_image = generate_captcha()
# captcha_image=preprocess_image(captcha_image)
# captcha_text_one_hot = one_hot_encode(captcha_text)

# model1 = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='elu', input_shape=(60, 160, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),

#     tf.keras.layers.Conv2D(64, (3, 3), activation='elu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),

#     tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),

#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='elu'),
#     tf.keras.layers.Dropout(0.8),
#     tf.keras.layers.Dense(captcha_length * num_classes, activation='softmax'),
#     tf.keras.layers.Reshape((captcha_length, num_classes))
# ])

# model1.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])

# # 訓練模型
# history1 = model1.fit(
#     np.array([captcha_image]),
#     np.array([captcha_text_one_hot]),
#     epochs=200
# )

# # 使用模型預測驗證碼
# prediction1 = model1.predict(np.array([captcha_image]))[0]
# predicted_captcha1 = ''.join([characters[np.argmax(prediction1[i])] for i in range(captcha_length)])
# print(f"Model1: Actual captcha: {captcha_text}, predicted captcha: {predicted_captcha1}")
# %%
