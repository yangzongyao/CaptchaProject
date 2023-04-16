from captcha.image import ImageCaptcha
from captcha.audio import AudioCaptcha
from PIL import Image
import random
import string

path=r'D:/ImageCaptcha/'
# audio = AudioCaptcha(voicedir='/path/to/voices')
# image = ImageCaptcha(fonts=[path+'A.ttf', path+'B.ttf'])
characters = string.digits + string.ascii_letters
num_classes = len(characters)
captcha_length = 6

for i in range(10):
    text = ''.join(random.choices(characters, k=captcha_length))

# Create an ImageCaptcha object with the generated text
    image_ready = ImageCaptcha()
    # image_ready = ImageCaptcha(width = 280, height = 90)
    image = image_ready.generate(text)
    pil_image = Image.open(image)

# Save the image to a file
    file_name='picture'
    file_num=str(i)
    file_png='.png'
    file = file_name+file_num+file_png
    try:
        with open(file, 'wb') as f:
            pil_image.save(f, format='png')
        print(f'Successfully saved image to {file}')
    except IOError as e:
        print(f'Error saving image: {e}')