from src.Replacer import Replacer
import cv2
from PIL import Image




replacer = Replacer()
input_image = Image.open("/home/evobits/vyrodrive/rzamarefat/Replacer/test_images/01.jpg")

grounding_caption = "replace the car with a monkey"
replacer.replace(input_image=input_image, grounding_caption=grounding_caption)