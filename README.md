# Testing task for NtechLab
## [Solution of Task 1](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/Task_1.py)
## Task 2
### Training model
For traning model use [Task2_draft.ipynp](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/Task_2_draft.ipynb)<br>
Download dataset [here](https://drive.google.com/file/d/1-HUNDjcmSqdtMCvEkVlI0q43qlkcXBdK/view)<br>
By default, the model looks for files in the "Data/internship_data" folder<br>
For train in google colaboratory you can put the dataset archive "internship_data.tar.gz" in "My Drive/Colab/NtechLab/"<br>
### Data Preprocessing
Image size 224x244<br>
Normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])<br>
### Testing model
<br>
For test model use [use_model.py](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/use_model.py)<br>
For example: 

```python
python use_model.py Data/test  
```

Where __Data\test__ is a directory with test images<br>
The result of use_model.py is the _predictions.json_ file<br>

### Model
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

Where SE - [Squeeze and Excited block](https://arxiv.org/abs/1709.01507). RE - ReLU, HS - [HardSwish](https://arxiv.org/abs/1905.02244)<br>
[model.py](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/model.py) - model<br>
model.pkl - pretreined model with default architecture and training parameters. Accuracy 0.95
### Other
[data_functions.py](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/data_functions.py) - functions to work with data  
[train_functions.py](https://github.com/AllexFrolov/NtechLab-testing_task/blob/master/train_functions.py) - functions for train model  
