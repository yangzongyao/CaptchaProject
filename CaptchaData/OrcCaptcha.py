# %%
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import choice
 
# %%
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = choice(char_set)
        captcha_text.append(c)
    return captcha_text
 

def gen_captcha_text_and_image():
    image = ImageCaptcha()
 
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
 
    captcha = image.generate(captcha_text)
    #image.write(captcha_text, captcha_text + '.jpg')  # 寫入文件
 
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    captcha_image = captcha_image.astype(np.float32)
    return captcha_text, captcha_image
 
def show_gen_image(text, image):
    image = image.astype(np.uint8)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

text, image = gen_captcha_text_and_image()

# %%
