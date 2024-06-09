import os
import cv2

# Görüntüleri 128x128 alt görüntüler haline getirip belirtilen klasöre kaydetme fonksiyonu
def create_subimages(input_dir, output_dir):
    # Çıkış klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Giriş klasöründeki her bir görüntüyü işle
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı olarak görüntüyü oku

            if image is None:
                continue

            # Görüntüyü 128x128 alt görüntüler haline getir
            subimages = []
            height, width = image.shape
            for y in range(0, height, 192):
                for x in range(0, width, 192):
                    subimage = image[y:y+192, x:x+192]
                    if subimage.shape[:2] == (192, 192):  # Alt görüntü boyutu 128x128 olmalı
                        subimages.append(subimage)

            # Alt görüntüleri kaydet
            for i, subimage in enumerate(subimages):
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{i}.png")
                cv2.imwrite(output_path, subimage)

                print(f"Saved subimage to: {output_path}")

# Giriş ve çıkış klasörlerini belirt
input_dir = 'clean_db'
output_dir = 'clean_sub_images_192x192'

# Alt görüntülerin oluşturulması
create_subimages(input_dir, output_dir)
