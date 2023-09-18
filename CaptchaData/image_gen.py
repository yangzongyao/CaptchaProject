# %%
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import pathlib as Path
import numpy as np
ROOT = Path.Path(__file__).parent
DIR_IMAGE = ROOT / 'image'
DIR_IMAGE.mkdir(parents=True, exist_ok=True)

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
classes_l = number + alphabet + ALPHABET
classes_dict = dict(enumerate(classes_l))
classes_dict = dict((v , k) for k, v in classes_dict.items())

Color_List = ["rgb(166,206,227)",
              '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
              '#FECB52', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F',
              '#BCBD22', '#17BECF', '#3366CC', '#DC3912', '#FF9900', '#109618', '#990099', '#0099C6', '#DD4477',
              '#66AA00', '#B82E2E', '#316395', '#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#EECA3B',
              '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC', '#AA0DFE', '#3283FE', '#85660D', '#782AB6', '#565656',
              '#1C8356', '#16FF32', '#F7E1A0', '#E2E2E2', '#1CBE4F', '#C4451C', '#DEA0FD', '#FE00FA', '#325A9B',
              '#FEAF16', '#F8A19F', '#90AD1C', '#F6222E', '#1CFFCE', '#2ED9FF', '#B10DA1', '#C075A6', '#FC1CBF',
              '#B00068', '#FBE426', '#FA0087', '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
              '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D', '#6C7C32',
              '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038',
              '#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55',
              '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7',
              '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']

####################################################################
#                               plot                               #
####################################################################
def show_gen_image(text, image):
    image = np.array(image)
    image = image.astype(np.uint8)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

def draw_imgae_rect(captcha_image, rect_l, width, height):
    drawobj = ImageDraw.Draw(captcha_image)
    for i, rect in enumerate(rect_l):
        rect[0] = rect[0]*width # real  x
        rect[1] = rect[1]*height # real y
        rect[2] = rect[2]*width # w
        rect[3] = rect[3]*height # h
        drawobj.rectangle(rect, fill=None, outline=Color_List[i])
    image_rect = captcha_image
    return image_rect

####################################################################
#                           preprocessing                          #
####################################################################
def random_captcha_text(captcha_size, char_set=number+alphabet+ALPHABET):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 處理框圖 y座標超過圖片大小問題
def adjust_oversize(y_init, h, height):
    if y_init + h > height:
        h = height - y_init
    return h

def gen_captcha_text_and_image(captcha_size=4, width=160, height=60):
    image = ImageCaptcha(width, height)
 
    captcha_text = random_captcha_text(captcha_size)
    captcha_text = ''.join(captcha_text)
 
    captcha, chars_l, coord_list, width_in_old = image.generate(captcha_text)
    #image.write(captcha_text, captcha_text + '.jpg')  # 寫入文件
 
    captcha_image = Image.open(captcha)
    
    # image 被resize過時, 嘗試將座標跟寬度都乘上比例
    # x y w h are belong to [0,1] ratio, and x y are center points
    if width_in_old > width:
        x_ratio = (width / width_in_old) / width
        y_ratio = 1 / height
        rect_l = []
        for coord in coord_list:
            coord[1] = adjust_oversize(y_init = coord[3], h = coord[1], height = height)

            coord[0] = coord[0]*x_ratio # w
            coord[1] = coord[1]*y_ratio # h
            coord[2] = coord[2]*x_ratio # left-corner  x
            coord[3] = coord[3]*y_ratio # upper-corner y
            print(f'x_init:{coord[2]}, y_init:{coord[3]}, w:{coord[0]}, h:{coord[1]} \n')
            # calculate the (left, upper, right, bottom)
            rect_l.append([coord[2], coord[3], coord[2]+coord[0], coord[3]+coord[1]])
    else:
        x_ratio = 1 / width
        y_ratio = 1 / height
        rect_l = []
        for coord in coord_list:
            coord[1] = adjust_oversize(y_init = coord[3], h = coord[1], height = height)

            coord[0] = coord[0]*x_ratio # w
            coord[1] = coord[1]*y_ratio # h
            coord[2] = coord[2]*x_ratio # left-corner  x
            coord[3] = coord[3]*y_ratio # upper-corner y
            print(f'x_init:{coord[2]}, y_init:{coord[3]}, w:{coord[0]}, h:{coord[1]} \n')
            # calculate the (left, upper, right, bottom)
            rect_l.append([coord[2], coord[3], coord[2]+coord[0], coord[3]+coord[1]])

    return captcha_text, captcha_image, chars_l, rect_l
 
####################################################################
#                           main & save                            #
####################################################################
def image_generate(N = 1000, width = 160, height = 60):
    for idx in range(N):
        text, captcha_image, chars_l, rect_l = gen_captcha_text_and_image()
        # show_gen_image(text, captcha_image)

        # 畫框框
        # image_rect = draw_imgae_rect(captcha_image, rect_l, width, height)
        # show_gen_image(text, image_rect)

        # 刪去空格
        rect_l_tmp = rect_l.copy()
        for i , value in enumerate(chars_l):
            if value == " ":
                rect_l.remove(rect_l_tmp[i])

        while " " in chars_l:
            chars_l.remove(" ")

        # Each txt for one jpg image
        # Every classes in txt has one row  classID x y w h 
        # x y w h are belong to [0,1] ratio, and x y are center points
        Info_x_y_w_h = [[(rect[0]+rect[2])/2, (rect[1]+rect[3])/2 , (rect[2]-rect[0]), (rect[3]-rect[1])] for rect in rect_l]
        str_Info = [f'{Info[0]} {Info[1]} {Info[2]} {Info[3]}' for Info in Info_x_y_w_h]
        txt = [f"{classes_dict[chars_l[i]]} " + str_Info[i] + "\n" for i in range(len(chars_l))]

        # Output jpg and txt (Same filename between txt and jpg)
        captcha_image.save(DIR_IMAGE / f'image{idx}.jpg')

        with open(DIR_IMAGE / f'image{idx}.txt', 'w') as f:
            f.writelines(txt)
            # f.close()

def classes_generate():
    classes = [tt + '\n' for tt in classes_l]
    with open(ROOT / f'classes.txt', 'w') as f:
        f.writelines(classes)
        # f.close()
def shuffle_train_valid_generate(N=1000):
    train_idx = np.random.randint(0, N, int(N*0.8))
    train_idx.sort()
    valid_idx = set(range(N)) - set(train_idx)
    str_train_image = [str(DIR_IMAGE.absolute() / f'image{idx}.jpg').replace('\\', '/')+"\n" for idx in list(train_idx)]
    str_valid_image = [str(DIR_IMAGE.absolute() / f'image{idx}.jpg').replace('\\', '/')+"\n" for idx in list(valid_idx)]

    with open(DIR_IMAGE / f'train.txt', 'w') as f:
        f.writelines(str_train_image)
    with open(DIR_IMAGE / f'valid.txt', 'w') as f:
        f.writelines(str_valid_image)
# %%
if __name__ == '__main__':    
    classes_generate()
    image_generate()
    shuffle_train_valid_generate()
