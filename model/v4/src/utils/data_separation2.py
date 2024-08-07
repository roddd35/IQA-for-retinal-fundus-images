import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# contar o balanceamento das imagens
def contBalance(df):
    contAdequate = len(df[df['quality_label'] == 1])
    contInadequate = len(df[df['quality_label'] == 0])
    print("Adequados: {} / Inadequados: {} / Balanceamento adequadas: {:.2f}".format(
        contAdequate, contInadequate, contAdequate / (contAdequate + contInadequate)))

# criar diretorios caso nao existam
def create_directories(base_path):
    os.makedirs(os.path.join(base_path, '0'), exist_ok=True)
    os.makedirs(os.path.join(base_path, '1'), exist_ok=True)

# mover as imagens pro diretorio
def move_images(img_path, dest_path, df):
    create_directories(dest_path)
    for index, row in df.iterrows():
        image_name = f"{row['image_id']}.jpg"
        quality = row['quality_label']
        original_image_path = os.path.join(img_path, image_name)
        destination_path = os.path.join(dest_path, str(quality), image_name)

        if os.path.exists(original_image_path):
            shutil.move(original_image_path, destination_path)
        else:
            print(f"A imagem {image_name} não foi encontrada.")

    print("Processo de movimentação concluído.")

def main():
    image_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos'
    train_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/train'
    test_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/test'
    val_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/validation'

    labels = pd.read_csv('/home/rodrigocm/scratch/datasets/brset/labels.csv', sep=',')

    # Corrigir a definição do quality_label
    labels['quality_label'] = labels['quality'].apply(lambda x: 1 if x == 'Adequate' else 0)

    # Dividir os dados em treino, teste e validação
    train_df, test_df = train_test_split(labels, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42)

    # Balancear as classes no conjunto de treino
    train_df_balanced = pd.concat([
        resample(train_df[train_df['quality_label'] == 1], replace=True, n_samples=len(train_df[train_df['quality_label'] == 0]), random_state=42),
        train_df[train_df['quality_label'] == 0]
    ])

    # Balancear as classes no conjunto de validação
    val_df_balanced = pd.concat([
        resample(val_df[val_df['quality_label'] == 1], replace=True, n_samples=len(val_df[val_df['quality_label'] == 0]), random_state=42),
        val_df[val_df['quality_label'] == 0]
    ])

    print("Treino balanceado:")
    contBalance(train_df_balanced)
    print("Validação balanceada:")
    contBalance(val_df_balanced)
    print("Teste:")
    contBalance(test_df)

    # Movimentar as imagens
    move_images(img_path=image_path, dest_path=train_path, df=train_df_balanced)
    move_images(img_path=image_path, dest_path=test_path, df=test_df)
    move_images(img_path=image_path, dest_path=val_path, df=val_df_balanced)

main()