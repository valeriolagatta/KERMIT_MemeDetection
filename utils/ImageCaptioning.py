import torch
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel


loc = "ydshieh/vit-gpt2-coco-en"

#Build Model
feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)

#If GPU Available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

model.eval()

#Funzione che data un'immagine ritorna la sua descrizione
def Captioning(path_img):
    image = Image.open(path_img).convert('RGB')
    image = image.resize([256, 256])

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    pixel_values = pixel_values.to(device=device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds[0]
