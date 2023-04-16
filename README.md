# CaptchaProject 

## 專案介紹

這是一個用於解決 captcha 辨識問題的專案。captcha 是一種常見的網站驗證機制，通常是由一張包含數字、字母或符號的圖片組成，目的是防止機器人和自動化程式的攻擊。本專案旨在開發一個能夠自動識別 captcha 的模型，以提高網站的安全性和使用體驗。

------------------------------

## 資料
```
|--- ChaptchaData/  
       |---CreateCaptchaData_forImg.py     # Conver to image file
       |---CreateCaptchaData_forTorch.py   # Convert to Pytorch Dataset
       |---OrcCaptcha.py                   # 字元定義、產生Captcha圖片程式碼
```

可以用OrcCaptcha.gen_captcha_text_and_image() function產生Captcha圖在跟文字
```
text, img = gen_captcha_text_and_image()
```
------------------------------

## 模型

```
|--- TorchFunction/  
       |---TorchMethod.py  #  定義Torch方法
       |---TorchModel.py   #  定義Torch模型
```
