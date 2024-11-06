import tensorflow as tf
import numpy as np

#Defenição dos diretóroios onde se encontram as imagens
DATASET_TRAIN_PATH = 'cats_and_dogs/train'
DATASET_VALIDATION_PATH = 'cats_and_dogs/validation'

#Defenição das variáveis para a criação dos datasets
BATCH_SIZE = 500   #Foi escolhido um BATCH_SIZE de 500, pois é o que divide melhor as imagens para Validação e Teste
IMG_HEIGHT = 160
IMG_WIDTH = 160
SEED = 2323

def determineDogsNCats(dataset):
    label_0_count = 0
    label_1_count = 0

    for _, labels in dataset:
        # Conta as ocorrências de 0 e 1 no lote atual
        label_0_count += tf.reduce_sum(tf.cast(labels == 0, tf.int32)).numpy()
        label_1_count += tf.reduce_sum(tf.cast(labels == 1, tf.int32)).numpy()

    print(f"Total de imagens com gatos: {label_0_count}")
    print(f"Total de imagens com cães: {label_1_count}")

#Criação dos datasets de treino e validação original (Que será dividido no dreal de validação e treino)
#O modo das labels está em binário, sendo a label=0 referente a gatos e a label=1 referente a cães
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_TRAIN_PATH,
    labels = 'inferred',
    label_mode = 'binary',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

validation_dataset_original = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_VALIDATION_PATH,
    labels = 'inferred',
    label_mode = 'binary',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

print("Numero de gatos e cães no dataSet de treino")
determineDogsNCats(train_dataset)
print("\n")


print("Numero de gatos e cães no dataSet de validação original")
determineDogsNCats(validation_dataset_original)
print("\n")


#Criação dos datasets de validação e treino, cada um com um total de 500 imagens
#Com este BATCH_SIZE e SEED o validation_dataset possui 249 imagens de gatos e 251 de cães
#E o test_dataset tem 251 de gatos e 249 de cães
validation_dataset, test_dataset = tf.keras.utils.split_dataset(
    validation_dataset_original,
    left_size = 0.5,
    seed = SEED
)

print("Numero de gatos e cães no dataSet de validação")
determineDogsNCats(validation_dataset)
print("\n")

print("Numero de gatos e cães no dataSet de teste")
determineDogsNCats(test_dataset)
print("\n")


