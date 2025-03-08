def resize_image(image, target_size):
    from PIL import Image
    return image.resize(target_size, Image.ANTIALIAS)

def filter_image(image, filter_type):
    from PIL import ImageFilter
    if filter_type == 'blur':
        return image.filter(ImageFilter.BLUR)
    elif filter_type == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)
    else:
        return image

def convert_image_to_text(image):
    import pytesseract
    return pytesseract.image_to_string(image)

def save_image(image, filepath):
    image.save(filepath)