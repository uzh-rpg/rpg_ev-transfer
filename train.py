"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train.py --settings_file "config/settings.yaml"
"""
import argparse

from config.settings import Settings
from training.classification_trainer import ClassificationModel
from training.object_det_trainer import ObjectDetModel


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    if settings.model_name == 'classification_model':
        trainer = ClassificationModel(settings)
    elif settings.model_name == 'object_det_model':
        trainer = ObjectDetModel(settings)

    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)

    trainer.train()


if __name__ == "__main__":
    main()
