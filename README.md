# Gender classifier
## [Requerements](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/requirements.txt)
## Training model
For traning model use [train_classifier.ipynp](https://github.com/AllexFrolov/Gender-classifier/blob/master/train_classifier.ipynb)<br>
Download dataset [here](https://drive.google.com/file/d/1-HUNDjcmSqdtMCvEkVlI0q43qlkcXBdK/view)<br>
By default, the model looks for files in the "Data/internship_data" folder<br>
For training in google colaboratory put the dataset archive "internship_data.tar.gz" in "My Drive/Colab/Gender_classifier/"<br>
## Data preprocessing
Image size 224x224<br>
Normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])<br>
## Testing model
For test model run
[use_model.py](https://github.com/AllexFrolov/Gender-classifier/blob/master/use_model.py)  
For example: 

```python
python use_model.py Data/test  
```

Where __Data/test__ is a directory with test images<br>
The result of use_model.py is the file _predictions.json_ with predictions in the form:<br>
```python
{"000001.jpg": "female", "000004.jpg": "male", "000009.jpg": "female", "000010.jpg": "female"}
```

## Model
Default model architecture:
| name | in_ch | out_ch | k_size | batchnorm | SE | non_linear | stride| input_size |
|------|:-----:|:------:|:------:|:---------:|:--:|:-------:|:-----:|:----------:|
| Conv | image_ch | 8  | 3 | True | False | RE | 2 | 224 |
| Conv | 8  | 16 | 3 | True | True  | RE | 2 | 112 |
| Conv | 16 | 32 | 3 | True | True  | HS | 2 | 56  |
| Conv | 32 | 32 | 3 | True | False | RE | 1 | 28  |
| Conv | 32 | 64 | 3 | True | False | RE | 2 | 28  |
| Conv | 64 | 128 | 3 | True | True | HS | 2 | 14  |
| AdaptiveAvgPool | out_size - 1 | - | - | - | - | - | - | 7 |
| Conv | 128 | 256 | 1 | False | False | HS | 1 | 1 |
| Dropout | p = 0.8 | - | - | - | - | - | - | 1 |
| Conv | 256 | classes_count | 1 | False | False | None | 1 | 1 |
|Flatten|

Where SE - [Squeeze and Excitation block](https://arxiv.org/abs/1709.01507). RE - ReLU, HS - [HardSwish](https://arxiv.org/abs/1905.02244)<br>
[model.py](https://github.com/AllexFrolov/Gender-classifier/blob/master/model.py) - model<br>
pretrained_weights/model.pkl - pretrained model with default architecture and training parameters. Accuracy 0.95
## Other
[data_functions.py](https://github.com/AllexFrolov/Gender-classifier/blob/master/data_functions.py) - functions to work with data  
[train_functions.py](https://github.com/AllexFrolov/Gender-classifier/blob/master/train_functions.py) - functions for train model  
