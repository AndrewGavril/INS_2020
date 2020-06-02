from PIL import Image
import os

def scale_image(input_image_path,
                output_image_path,
                width=None,
                height=None
                ):
    original_image = Image.open(input_image_path)
    w, h = original_image.size
    '''print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h)) '''

    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')

    original_image.thumbnail(max_size, Image.ANTIALIAS)
    original_image.save(output_image_path)

    scaled_image = Image.open(output_image_path)
    width, height = scaled_image.size
    ''' print('The scaled image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height)) '''


if __name__ == '__main__':
    tree = os.listdir('train_images')
    if os.path.exists('./new_train_images'):
        print("already exists!")
    else:
        print("making new_train_images")
        os.mkdir('new_train_images')
        for i in tree:
            scale_image(input_image_path = 'train_images/'+i,
                output_image_path='new_train_images/'+i,
                width=800,height=128)