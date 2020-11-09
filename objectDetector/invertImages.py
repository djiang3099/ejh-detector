from PIL import Image
import PIL.ImageOps    
import glob


for filepath in glob.glob('test2/*.png'):
    
    image = Image.open(filepath)

    inverted_image = PIL.ImageOps.invert(image)

    inverted_image.save(filepath)