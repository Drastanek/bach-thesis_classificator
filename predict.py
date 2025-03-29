import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFile

# some images may be truncated, this avoid earrors while loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = YOLO("yolo11n/50epoch/no_rotifera/exp1/weights/best.pt")

image_folder = "data_to_classify/"
results_folder = "predict_results/"

os.makedirs(results_folder, exist_ok=True)

# Get a list of image files in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
class_counts = {}

# process images
batch_size = 50
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i + batch_size]
    
    # convert images to grayscale
    grayscale_images = []
    color_images = []
    for file in batch_files:
        try:
            color_img = Image.open(file)
            grayscale_img = color_img.convert('L')
            grayscale_images.append(grayscale_img)
            color_images.append(color_img)
        except OSError as e:
            print(f"Skipping file {file} because error: {e}")
            continue
    
    results = model(grayscale_images)  # return a list of Results objects

    for j, result in enumerate(results):
        draw = ImageDraw.Draw(color_images[j])
        font = ImageFont.load_default()
        font = font.font_variant(size=50)
        
        # bounding boxes with class names and confidence on the color image
        for box in result.boxes:
            class_index = int(box.cls.item())
            class_name = result.names[class_index]
            confidence = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            
            text = f"{confidence:.2f}|{class_name}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # text position to avoid overlapping with edges
            text_x = x1
            text_y = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y2 + 10
            
            draw.text((text_x, text_y), text, fill="blue", font=font)
            
            # update counter
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        # save result image
        output_filename = os.path.join(results_folder, os.path.basename(batch_files[j]))
        color_images[j].save(output_filename)

    print(f"Processed %d of %d images" % (i + len(batch_files), len(image_files)))

# counts to count.txt
with open(os.path.join(results_folder, "count.txt"), "w") as count_file:
    for class_name, count in class_counts.items():
        count_file.write(f"{class_name}: {count}\n")