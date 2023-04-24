# %%
import numpy as np
from torch.utils.data import DataLoader, Dataset
from CaptchaData.OrcCaptcha import gen_captcha_text_and_image, show_gen_image
import torch

def one_hot_encode(text, num_classes, characters):
    return np.array([np.eye(num_classes)[characters.index(c)] for c in text],dtype=np.float32)

class captchaData(Dataset):
    def __init__(self, data_num=101, device="cpu", gray=False):
        number = ['0','1','2','3','4','5','6','7','8','9']
        alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        space = ['-']

        self.characters = space + number + alphabet + ALPHABET
        num_classes = len(self.characters)
        for _ in range(data_num):
            txt, img = gen_captcha_text_and_image()
            if(gray):
                img = np.dot(img, [0.2989, 0.5870, 0.1140])
                img = img.reshape(1, 1, 60, 160).astype(np.float32)
            else:
                img = img.reshape(1, 3, 60, 160)
            # 因應torch的CrossEntropyLoss，將y改為label的值
            # one_hot_y = one_hot_encode(txt, num_classes, self.characters).reshape(1, 4, 62)
            txt = [self.characters.index(i) for i in txt]
            if _ == 0:
                x = img
                # 因應torch的CrossEntropyLoss，將y改為label的值
                # y = one_hot_y
                y = [txt]
            else:
                x = np.append(x, img, axis=0)
                # 因應torch的CrossEntropyLoss，將y改為label的值
                # y = np.append(y, one_hot_y, axis=0)
                y = np.append(y, [txt], axis=0)
        self.x = torch.from_numpy(x).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.n_samples = x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
# %%

# data = captchaData()
# DataLoader(dataset=data, batch_size=50, shuffle=True)
# %%