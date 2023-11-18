import sys
sys.path.append("..")

import torch
from utils import ResidualBlock, DownBlock, UpBlock, UNet, DiffusionModel
import pytest
import os
import shutil
from typing import Generator


@pytest.fixture
def conf_train():
    return DiffusionModel.Conf(
        num_epochs=2,
        output_dir="TMP_output",
        image_folder_test="images_test",
        image_folder_train="images_train",
        )


@pytest.fixture
def conf_test():
    return DiffusionModel.Conf(
        output_dir="TMP_output",
        image_folder_test="images_test",
        image_folder_train="images_train",
        initialize=DiffusionModel.Conf.Initialize.FROM_BEST_CHECKPOINT
        )


class TestDiff:


    def _clear(self, conf: DiffusionModel.Conf):
        if os.path.exists(conf.output_dir):
            shutil.rmtree(conf.output_dir)


    def _make_diff(self, conf: DiffusionModel.Conf) -> DiffusionModel:
        return DiffusionModel(conf)
    

    def test_train(self, conf_train: DiffusionModel.Conf):
        self._clear(conf_train)
        diff_train = self._make_diff(conf_train)
        diff_train.train()

        # Check that checkpoints have been saved
        assert os.path.exists(diff_train.conf.checkpoint_best)
        assert os.path.exists(diff_train.conf.checkpoint_latest)
        assert os.path.exists(diff_train.conf.checkpoint_init)

        # Clean up
        self._clear(conf_train)
    

    def test_generate(self, conf_train: DiffusionModel.Conf, conf_test: DiffusionModel.Conf):
        self._clear(conf_train)
        diff_train = self._make_diff(conf_train)
        diff_train.train()

        diff_test = self._make_diff(conf_test)
        images = diff_test.generate(
            num_images=1,
            diffusion_steps=20,
            initial_noise=None
            )
        assert len(images) == 1
        assert images[0].size == (conf_test.image_size, conf_test.image_size)

        # Clean up
        self._clear(conf_train)
