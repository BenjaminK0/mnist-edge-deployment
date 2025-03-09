import numpy as np
import tensorflow as tf
import pandas as pd
import argparse, time

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-p', '--path', required=True, help='Path of input CSV file')
    parser.add_argument(
        '-d', '--digit', type=int, default=0, help='Digit from 0-9 to test')
    return parser.parse_args()


def load_data(path):
    test_dataframe = pd.read_csv(path)
    test_np = test_dataframe.to_numpy().astype(np.float32)
    test_labels = test_np[:, 0]
    test_data = test_np[:, 1:]
    test_data = test_data.reshape((len(test_data),28,28))
    test_images = np.expand_dims(test_data, axis=-1)

    return test_images, test_labels


# https://ai.google.dev/edge/litert/inference#load_and_run_a_model_in_python
if __name__ == '__main__':
    args = parse_args()

    # select chosen input image
    data = load_data(args.path)
    image = np.expand_dims(data[0][args.digit], axis=0)
    label = data[1][args.digit]

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    confidence = np.max(output_data)
    prediction = np.where(output_data[0] == confidence)[0][0]

    print('-------RESULTS--------')
    print(f"Predicted Label: {prediction}, Confidence: {confidence}")
    print(f'Actual Label: {int(label)}')
