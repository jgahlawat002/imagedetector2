from flask import Flask, request, jsonify
import numpy as np
import base64
from imagedetector import ImageDetector
import io
from PIL import Image

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload():
    data = request.json

    # validation
    if 'id' not in data or 'image' not in data:
        return jsonify({"error": "Both 'id' and 'image' are required."}), 400

    # Decode base64 image
    encoded_image = data['image']
    im_bytes = base64.b64decode(encoded_image)  
    im_file = io.BytesIO(im_bytes)  
    img = Image.open(im_file)
    img = np.array(img)
    imageDetector = ImageDetector()
    
    objects = imageDetector.detectImage(img)

    # Return detected objects
    return jsonify({"id": data['id'], "objects": objects}), 200

if __name__ == '__main__':
    app.run(port=1024, debug=True)
