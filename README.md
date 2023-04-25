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
----------------------------
## 目前進度
1. ~~修改torch模型，改用CNN~~ (訓練20k圖片、500epoch發現無法general於Test Set)
2. 修正loss function寫法
3. ~~使用CRNN (OCR任務常用)~~
4. 已建立CRNN，且新增CTC loss，尚須理解CTC loss作法；以及增加訓練集，增加epoch測試看看模型效果
---------------------------
## Ref.
1. [文本辨識網頁分享](https://aiacademy.tw/ai-optical-character-recognition/)
2. [TrOCR](./Ref/TrOCR.pdf)
3. [CRNN paper](./Ref/TextRec.pdf)
4. [CRNN 網頁](https://cinnamonaitaiwan.medium.com/ocr-驗證碼識別理論與實作-a97273a5657d)

---------------------------
## PLH_ADD