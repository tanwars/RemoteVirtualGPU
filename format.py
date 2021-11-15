from PIL import Image
import io
import os

if __name__ == "__main__":
    for d in os.listdir('flower_photos_formatted'):
        for f in os.listdir(os.path.join('flower_photos_formatted', d)):
            img = Image.open(os.path.join(os.path.join('flower_photos_formatted', d), f))
            img = img.resize((224, 224))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            print(d, f)

            with open(os.path.join(os.path.join('flower_photos_formatted', d), f), 'wb') as fp:
                fp.write(img_byte_arr.getbuffer())
