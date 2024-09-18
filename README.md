# ResNet-50 and Image Classification

<p align="justify">
<strong>ResNet-50</strong> is a deep convolutional neural network (CNN) from the ResNet (Residual Network) family, introduced by Microsoft in 2015. With 50 layers, it is designed to tackle the challenges of training very deep networks, such as the vanishing gradient problem, by using residual connections or shortcuts. These connections allow the network to learn more efficiently and enable the training of much deeper architectures. ResNet-50 is known for its strong performance on image classification tasks and has become a popular backbone model for various computer vision applications, offering a balance between depth, accuracy, and computational efficiency.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_ResNet50_ImageClassification_ibuyesha.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_resnet50_train_exydumnh.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_lgqllayc/aicandy_model_pth_ydvnemld.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_resnet50_test_ycmlontg.py --image_path ../image_test.jpg --model_path aicandy_model_out_lgqllayc/aicandy_model_pth_ydvnemld.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_resnet50_convert_onnx_nuuipble.py --model_path aicandy_model_out_lgqllayc/aicandy_model_pth_ydvnemld.pth --onnx_path aicandy_model_out_lgqllayc/aicandy_model_onnx_eiggdxhh.onnx --num_classes 2
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-resnet-50-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




