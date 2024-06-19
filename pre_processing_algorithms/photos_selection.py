import os
import random
import pandas as pd

# 12279 imagens a serem excluidas
# deletar as fotos nao escolhidas
def delete_photos_func(delete_photos):
    cont = 0
    path = '/home/rodrigocm/scratch/datasets/brset/selected_photos'
    
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext == '.jpg' and name in delete_photos:
            os.remove(os.path.join(path, filename))
            cont += 1
    print("Excluidos: {}".format(cont))


# selecionar 2000 fotos que sejam adequadas aleatoriamente
def select_photos():
    selected_photos = []
    all_photos = []
    labels_df = pd.read_csv('/home/rodrigocm/scratch/datasets/brset/labels.csv', sep=',')

    for index, row in labels_df.iterrows():
        if row['quality'] == 'Adequate':
            selected_photos.append(row['image_id'])

    random.shuffle(selected_photos)

    delete_photos = selected_photos[2000:]
    selected_photos = selected_photos[0:2000]

    return selected_photos, delete_photos, labels_df

def update_csv(labels_df, selected_photos, output_csv_path):
    updated_df = labels_df[(labels_df['quality'] == 'Inadequate') | (labels_df['image_id'].isin(selected_photos))]
    updated_df.to_csv(output_csv_path, index=False)
    print("Novo CSV salvo em: {}".format(output_csv_path))

def main():
    selected_photos, delete_photos, labels_df = select_photos()

    for i in selected_photos:
        i = i + '.jpg'
        print(i)
    print("Quantidade de imagens adequadas mantidas: {}".format(len(selected_photos)))
    print("Quantidade de imagens adequadas excluídas: {}".format(len(delete_photos)))

    delete_photos_func(delete_photos)
    update_csv(labels_df, selected_photos, "/home/rodrigocm/scratch/datasets/brset/labelsNovas.csv")
main()
