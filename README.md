# ImageColorization

The aim of this project was to experiment with colorizing grayscale images. As my model I implemented [ChromaGAN](https://openaccess.thecvf.com/content_WACV_2020/html/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.html) in TensorFlow. As another model to compare to I used [ColorfulColorization](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40) which is a segmentation architecture. For the dataset I used new [Natural-Color Dataset](https://github.com/saeed-anwar/ColorSurvey) which has simple images with white background where is no doubt what the color should be.

## Folder Structure

- `data` - contains data for the task
- `notebooks` - contains notebooks: SOTA in image colorization, exploration of dataset, and training with evaluation
- `report` - contains simple project report
- `src` - contains source code used in notebooks
- `LICENSE` - standard MIT license
