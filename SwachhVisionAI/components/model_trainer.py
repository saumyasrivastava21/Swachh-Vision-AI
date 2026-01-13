import os
import sys
import yaml
import shutil
from SwachhVisionAI.utils.mains_utils import read_yaml_file
from SwachhVisionAI.logger import logging
from SwachhVisionAI.exception import AppException
from SwachhVisionAI.entity.config_entity import ModelTrainerConfig
from SwachhVisionAI.entity.artifact_entity import ModelTrainerArtifact
from zipfile import ZipFile


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # ---------------- Extract data.zip ----------------
            zip_file_path = "data.zip"
            if not os.path.exists(zip_file_path):
                raise AppException(f"{zip_file_path} does not exist", sys)

            logging.info("Extracting data.zip")
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_file_path)

            # ---------------- Read number of classes ----------------
            yaml_file_path = "data.yaml"
            if not os.path.exists(yaml_file_path):
                raise AppException(f"{yaml_file_path} does not exist", sys)

            with open(yaml_file_path, 'r') as stream:
                num_classes = int(yaml.safe_load(stream)['nc'])
            logging.info(f"Number of classes: {num_classes}")

            # ---------------- Prepare custom model config ----------------
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            logging.info(f"Model config file: {model_config_file_name}")

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            config['nc'] = num_classes

            custom_model_path = f'yolov5/models/custom_{model_config_file_name}.yaml'
            with open(custom_model_path, 'w') as f:
                yaml.dump(config, f)
            logging.info(f"Custom config saved: {custom_model_path}")

            # ---------------- Train YOLOv5 model ----------------
            logging.info("Starting YOLOv5 training")
            train_command = (
                f"cd yolov5 && python train.py "
                f"--img 416 "
                f"--batch {self.model_trainer_config.batch_size} "
                f"--epochs {self.model_trainer_config.no_epochs} "
                f"--data ../data.yaml "
                f"--cfg ./models/custom_{model_config_file_name}.yaml "
                f"--weights {self.model_trainer_config.weight_name} "
                f"--name yolov5s_results "
                f"--cache"
            )
            os.system(train_command)

            # ---------------- Save trained model ----------------
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            best_model_src = "yolov5/runs/train/yolov5s_results/weights/best.pt"
            best_model_dest = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            shutil.copy(best_model_src, best_model_dest)
            shutil.copy(best_model_src, "yolov5/best.pt")

            # ---------------- Cleanup ----------------
            shutil.rmtree("yolov5/runs", ignore_errors=True)
            shutil.rmtree("train", ignore_errors=True)
            shutil.rmtree("valid", ignore_errors=True)
            if os.path.exists("data.yaml"):
                os.remove("data.yaml")

            # ---------------- Return artifact ----------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=best_model_dest
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
