import io
import requests

import streamlit as st
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


st.title('顔認証アプリ')

subscription_key = 'a48b35dc7b3e41ebaa413cf1a9ee7035'
endpoint = 'https://20210505miya.cognitiveservices.azure.com/'
face_api_url = endpoint + 'face/v1.0/detect'
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}
params = {
    'returnFaceId': 'true',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion',
}

# 検出した顔に描く長方形の座標を取得


def get_rectangle(faceDictionary):
    rect = faceDictionary['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']

    return ((left, top), (right, bottom))


# 描画するテキストを取得
def get_draw_text(faceDictionary):
    rect = faceDictionary['faceRectangle']
    faceAttr = faceDictionary['faceAttributes']
    age = int(faceAttr['age'])
    gender = faceAttr['gender']

    text = f'{gender} {age}'

    # 枠に合わせてフォントサイズを調整
    font_size = max(16, int(rect['width'] / len(text)))
    font = ImageFont.truetype(r'C:\windows\fonts\meiryo.ttc', font_size)

    return (text, font)


# 認識された顔の上にテキストを描く座標を取得
def get_text_rectangle(faceDictionary, text, font):
    rect = faceDictionary['faceRectangle']
    text_width, text_height = font.getsize(text)
    left = rect['left'] + rect['width'] / 2 - text_width / 2
    top = rect['top'] - text_height - 1

    return (left, top)


# テキストを描画
def draw_text(faceDictionary):
    text, font = get_draw_text(faceDictionary)
    text_rect = get_text_rectangle(faceDictionary, text, font)
    draw.text(text_rect, text, align='center', font=font, fill='red')


uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
if uploaded_file:
    img = Image.open(uploaded_file)

    with io.BytesIO() as output:
        img.save(output, format='JPEG')
        binary_img = output.getvalue()

    res = requests.post(face_api_url, params=params,
                        headers=headers, data=binary_img)
    results = res.json()

    if not results:
        raise Exception('画像から顔を検出できませんでした。')

    for result in results:
        draw = ImageDraw.Draw(img)
        draw.rectangle(get_rectangle(result), outline='green', width=3)

        draw_text(result)

    st.image(img, caption='Uploaded Image.', use_column_width=True)
