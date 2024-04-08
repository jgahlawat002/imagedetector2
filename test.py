import base64
from io import BytesIO
from PIL import Image

with open("dogs.jpg", "rb") as f:
    im_b64 = base64.b64encode(f.read())
    print(im_b64)
    with open("text.txt", "w") as fl:
        fl.write(str(im_b64))


im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
im_file = BytesIO(im_bytes)  # convert image to file-like object
img = Image.open(im_file)   # img is now PIL Image object
