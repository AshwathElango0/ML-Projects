import re
import csv
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

def extract_donut_outputs_to_csv():
    # Load the locally saved CORD-v2 dataset
    dataset = load_dataset("naver-clova-ix/cord-v2")

    # Load the Donut processor and model
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    # Move the model to the appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Function to decode predictions into JSON
    def decode_prediction(pixel_values, task_prompt="<s_cord-v2>"):
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Decode the model's output
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

        return processor.token2json(sequence)

    # Open a CSV file to write predictions
    with open("donut_predictions.csv", mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(["Image_ID", "Predicted_JSON"])

        # Loop through the validation set and write outputs
        for idx, example in enumerate(tqdm(dataset['validation'], desc="Processing images")):
            image = example['image']

            # Process image and get model prediction
            pixel_values = processor(image, return_tensors="pt").pixel_values
            predicted_json = decode_prediction(pixel_values)

            # Write the image ID and predicted JSON to the CSV file
            writer.writerow([idx, predicted_json])

    print("Predictions have been written to 'donut_predictions.csv'.")

if __name__ == "__main__":
    extract_donut_outputs_to_csv()
