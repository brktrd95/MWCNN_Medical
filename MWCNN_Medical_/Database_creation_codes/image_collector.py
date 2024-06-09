import os
from PIL import Image, ImageDraw, ImageFont

# Klasör yolları
folder1 = '_test_set_clean'
folder2 = '_test_set_noisy'
folder3 = '_test_set_mwcnn'
output_folder = 'clean_noisy_mwcnnResult'

# Klasör yollarını bir listeye ekleyin
folders = [folder1, folder2, folder3]

# Başlıklar
titles = ['Clean', 'Noisy', 'MWCNN']

# Çıkış klasörünü oluştur
os.makedirs(output_folder, exist_ok=True)

# Yazı tipi ayarları
font_size = 72  # Daha büyük bir yazı boyutu kullan
font = ImageFont.truetype("arial.ttf", font_size)  # Arial yazı tipini kullan

# Klasörlerden aynı isimdeki dosyaları al ve birleştir
for filename in os.listdir(folder1):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        images = []
        
        # Her klasördeki resmi oku ve listeye ekle
        for folder in folders:
            img_path = os.path.join(folder, filename)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)
            else:
                print(f"Image {filename} not found in {folder}")

        # Resimleri yatay olarak birleştir
        if len(images) == 3:
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights) + font_size + 10  # Başlık için ekstra boşluk

            new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(new_img)
            
            x_offset = 0
            for i, img in enumerate(images):
                new_img.paste(img, (x_offset, font_size + 10))
                text_width, text_height = draw.textsize(titles[i], font=font)
                draw.text((x_offset + (img.width - text_width) // 2, 5), titles[i], font=font, fill=(0, 0, 0))
                x_offset += img.width
            
            # Birleştirilmiş resmi kaydet
            output_path = os.path.join(output_folder, filename)
            new_img.save(output_path)
            print(f"Saved combined image to {output_path}")
        else:
            print(f"Skipping {filename}, as it does not exist in all folders")
