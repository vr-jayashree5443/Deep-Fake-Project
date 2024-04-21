# Deep-Fake-Project

## Description
This project utilizes the First Order Motion Model to generate deep fake videos. It employs facial keypoint detection and image manipulation techniques to animate a source image using the movements from a driving video.

## Setup
To run this project, follow these steps:

1. Clone the First Order Model repository:
    ```
    !git clone https://github.com/AliaksandrSiarohin/first-order-model.git
    ```

2. Navigate to the cloned repository:
    ```
    cd first-order-model/
    ```

3. Import necessary libraries:
    ```
    import imageio
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from skimage.transform import resize
    from IPython.display import HTML
    import warnings
    ```

4. Load source image and driving video:
    ```
    source = imageio.imread('/content/drive/MyDrive/1.png')
    driver = imageio.get_reader('/content/drive/MyDrive/test.mp4')
    ```

5. Define helper functions for visualization:
    ```
    def display(source, driving, generated=None):
        ...
    ```

6. Load pre-trained checkpoints:
    ```
    from demo import load_checkpoints

    generator, kp_detector = load_checkpoints(config_path='/content/first-order-model/config/vox-256.yaml',
                                              checkpoint_path='/content/drive/MyDrive/Datasets/First_Order_Model_Data/vox-adv-cpk.pth.tar')
    ```

7. Generate deep fake animation:
    ```
    from demo import make_animation
    from skimage import img_as_ubyte

    prediction = make_animation(source_image, driver_video, generator, kp_detector)
    ```

8. Display and save the generated video:
    ```
    HTML(display(source_image, driver_video, prediction).to_html5_video())
    imageio.mimsave('../new_video.mp4', [img_as_ubyte(i) for i in prediction])
