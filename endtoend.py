import torch
from dataloaders.tokenizer import tokenize
from dataloaders.constants import SENTENCE_MAX_LEN, END_OF_SENTENCE, END_OF_WORD
from model import Dhvani

def inference(model, input_data):
    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(input_data)

    # Process the output to obtain the predicted phoneme sequence
    predicted_sequence = process_output(output)

    return predicted_sequence

def process_output(output):
    # Check if the output is a tuple or dictionary
    if isinstance(output, tuple):
        # Assume the predicted sequence is the first element of the tuple
        predicted_sequence = output[0]
    elif isinstance(output, dict):
        # Assume the predicted sequence is the value corresponding to the 'predicted_sequence' key
        predicted_sequence = output['predicted_sequence']
    else:
        # Assume the output is the predicted sequence
        predicted_sequence = output

    # Convert the predicted sequence to a readable format (e.g., list of tokens)
    predicted_sequence = convert_to_readable_format(predicted_sequence)

    return predicted_sequence

def convert_to_readable_format(tensor):
    # Implement the logic to convert the tensor to a readable format
    # (e.g., a list of tokens or characters)
    pass

def compare_sentences(original, predicted):
    print("Original sentence:", original)
    print("Predicted sequence:", predicted)

def prepare_sentence_for_inference(sentence):
    # Tokenize the sentence
    tokens = tokenize(sentence)

    # Convert tokens to tensor
    sentence_tensor = torch.tensor(tokens)

    # Pad or truncate the tensor to the desired length (e.g., SENTENCE_MAX_LEN)
    padded_tensor = pad_or_truncate_tensor(sentence_tensor, SENTENCE_MAX_LEN)

    return padded_tensor

def pad_or_truncate_tensor(tensor, max_len):
    # Truncate if the tensor is longer than max_len
    if tensor.size(0) > max_len:
        tensor = tensor[:max_len]
    # Pad if the tensor is shorter than max_len
    elif tensor.size(0) < max_len:
        padding = torch.zeros(max_len - tensor.size(0), dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)

    return tensor

def load_model_from_checkpoint(checkpoint_path):
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract the model state dictionary from the checkpoint
    state_dict = checkpoint['model']

    # Create an instance of the Dhvani model
    dhvani = Dhvani()
    dhvani.eval()

    # Load the state dictionary into the model
    dhvani.load_state_dict(checkpoint['model'])

    return dhvani

if __name__ == "__main__":
    checkpoint_path = '/home/satvik/Downloads/PRN_epoch-100_fold-2.pth'
    dhvani = load_model_from_checkpoint(checkpoint_path)
    sentence = "यह एक उदाहरण वाक्य है।"
    input_data = prepare_sentence_for_inference(sentence)
    predicted_sequence = inference(dhvani, input_data)
    compare_sentences(sentence, predicted_sequence)