import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch

from data_functions import MyDataLoader, Dataset
from model import AFModel


def predict(afmodel: AFModel, dataloader: MyDataLoader) -> dict:
    """
    do prediction
    :param afmodel: (AFModel) model
    :param dataloader: (MyDataLoader) Data iterator
    """
    afmodel.eval()
    predicts = []
    f_names = []
    with torch.no_grad():
        for X, names in dataloader:
            predict_proba = afmodel(torch.as_tensor(X)).squeeze(dim=-1)
            predicts += torch.argmax(predict_proba, dim=-1).data
            f_names += names

    classes = []
    for index in predicts:
        classes.append(afmodel.labels[index])
    result = dict(zip(f_names, classes))
    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_FOLDER = Path(sys.argv[1])
    else:
        DATA_FOLDER = Path(input('Путь к папке с изображениями: '))
    IM_SIZE = (224, 244)
    NORMALIZE = ([0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225])

    transformer = A.Compose([
        A.Resize(*IM_SIZE),
        A.Normalize(*NORMALIZE)
    ])

    test_transformer = lambda x: np.moveaxis(transformer(image=x)['image'], -1, 0)

    dataset = Dataset(DATA_FOLDER)
    loader = MyDataLoader(dataset, 100, None, False, test_transformer)
    model = AFModel()
    model.load_model()
    prediction = predict(model, loader)
    with open('predictions.json', 'w') as f:
        f.write(json.dumps(prediction))
