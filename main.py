import argparse
import yaml
import os
from loguru import logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or generate images from a diffusion model.")
    parser.add_argument("--command", type=str, choices=["plot-schedules", "plot-sinusoidal", "train", "generate", "generate-init", "loss"], required=True, help="Command to run. train: train a model. generate: generate images from a trained model. generate-init: generate images from a model initialized from scratch. plot-schedules: plot diffusion schedules. plot-sinusoidal: plot sinusoidal embeddings. loss: plot loss curve.")
    parser.add_argument("--conf", type=str, required=False, help="Path to config YAML file.")
    parser.add_argument("--mnt-dir", type=str, required=False, help="Path to mount directory. Dataset is assumed to be relative to this directory.")
    parser.add_argument("--show", action="store_true", required=False, help="Show plots.")
    args = parser.parse_args()

    if args.command == "plot-schedules":
        from utils import plot_diffusion_schedules, create_all_diffusion_schedules

        ds = create_all_diffusion_schedules(T=1000)
        plot_diffusion_schedules(ds)
    
    elif args.command == "plot-sinusoidal":
        from utils import plot_sinusoidal_embeddings

        plot_sinusoidal_embeddings(noise_embedding_size=32)

    elif args.command == "loss":
        from utils import plot_loss, DiffusionModel

        assert args.conf is not None, "Must provide a config YAML file with --conf"
        with open(args.conf, "r") as f:
            conf = DiffusionModel.Conf.from_dict(yaml.safe_load(f))
            conf.update_paths(args.mnt_dir)

        model = DiffusionModel(conf)
        metadata = model.load_training_metadata_or_new()
        fig = plot_loss(metadata)

        # Write
        fname = os.path.join(conf.output_dir, "loss.png")
        fig.write_image(fname)
        logger.info(f"Wrote loss plot to {fname}")

        if args.show:
            fig.show()

    elif args.command == "train":
        from utils import DiffusionModel

        assert args.conf is not None, "Must provide a config YAML file with --conf"
        with open(args.conf, "r") as f:
            conf = DiffusionModel.Conf.from_dict(yaml.safe_load(f))
            conf.update_paths(args.mnt_dir)

        model = DiffusionModel(conf)
        model.train()

    elif args.command in ["generate", "generate-init"]:
        from utils import DiffusionModel

        assert args.conf is not None, "Must provide a config YAML file with --conf"
        with open(args.conf, "r") as f:
            conf = DiffusionModel.Conf.from_dict(yaml.safe_load(f))
            conf.update_paths(args.mnt_dir)

        conf.initialize = DiffusionModel.Conf.Initialize.FROM_BEST_CHECKPOINT
        if args.command == "generate-init":
            conf.initialize = DiffusionModel.Conf.Initialize.FROM_SCRATCH

        model = DiffusionModel(conf)
        images = model.generate()
        image_dir = conf.img_output_dir
        os.makedirs(image_dir, exist_ok=True)
        for i, image in enumerate(images):
            fname = os.path.join(image_dir, f"%03d.png" % i)
            image.save(fname)
        logger.info(f"Saved {len(images)} images to {image_dir}")

    else:
        raise ValueError(f"Unknown command {args.command}")