import os


def save_model(model):
    if not os.path.exists("weights/"):
        os.mkdir("weights/")
    model.save_weights("weights/checkpoint.h5")
    assert os.path.exists("weights/checkpoint.h5")


def load_model(model, path="weights/checkpoint.h5"):
    model.load_weights(path)
    return model
