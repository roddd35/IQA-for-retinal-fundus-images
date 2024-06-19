import os
import shutil
import pandas as pd 

path = '/home/rodrigocm/scratch/datasets/brset/fundus_photos'
labels = pd.read_csv('~/scratch/datasets/brset/labels.csv', sep=',')

labels['quality_label'] = [0] * len(labels['quality'])

labels['quality_label'] = labels['quality'].apply(lambda x: 1 if x == 'Adequate' else 0)

os.makedirs('0', exist_ok=True)
os.makedirs('1', exist_ok=True)

# important_columns = ['image_id', 'patient_id', 'focus', 'iluminaton', 'image_field', 'artifacts', 'quality', 'quality_label']
# labels[important_columns].iloc[6800:6820]

for index, row in labels.iterrows():
    image_name = f"{row['image_id']}.jpg"
    quality = row['quality_label']

    original_image_path = os.path.join(path, image_name)

    if os.path.exists(original_image_path):
        destination_path = os.path.join(path, str(quality), image_name)

        shutil.move(original_image_path, destination_path)
    else:
        print(f"A imagem {image_name} não foi encontrada.")

print("Processo de movimentação concluído.")