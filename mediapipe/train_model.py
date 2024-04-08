import os
import tensorflow as tf
from mediapipe_model_maker.python.vision import image_classifier
from mediapipe_model_maker.python.vision import Dataset

# 画像分類モデルをトレーニングする関数
def train_image_classifier(image_path, export_dir='model'):
    data = Dataset.from_folder(image_path)
    train_data, remaining_data = data.split(0.8)
    test_data, validation_data = remaining_data.split(0.5)

    spec = image_classifier.ModelSpec(uri='https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1')
    model = image_classifier.create(train_data, model_spec=spec, validation_data=validation_data, epochs=5)

    loss, accuracy = model.evaluate(test_data)
    print(f'Test accuracy: {accuracy}')

    model.export(export_dir=export_dir)

    return model

if __name__ == '__main__':
    image_path = tf.keras.utils.get_file(
        'flower_photos.tgz',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        extract=True)
    image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

    train_image_classifier(image_path)
