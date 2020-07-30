"""
main.py

Written by Sindre Stenen Blakseth, 2020.

Main entry point for ESRGAN.
"""

#----------------------------------------------------------------------------
# Package imports

import argparse
import os

#----------------------------------------------------------------------------
# File imports


import config
import datasets
import download_data
import evaluate
import train

from models.esrgan import ESRGAN

#----------------------------------------------------------------------------

def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Set purpose of run.")
    parser.add_argument("--download", default=False, action="store_true", help="Download raw data.")
    parser.add_argument("--dataset",  default=False, action="store_true", help="Create new datasets from raw data.")
    parser.add_argument("--train",    default=False, action="store_true", help="Train ESRGAN.")
    parser.add_argument("--test",     default=False, action="store_true", help="Test pre-trained ESRGAN.")
    parser.add_argument("--use",      default=False, action="store_true", help="Use pre-trained ESRGAN on LR data.")
    args = parser.parse_args()

    # Check validity of arguments.
    if args.train != config.is_train or args.test != config.is_test or config.print_train_loss_period % config.d_g_train_ratio != 0:
        raise ValueError("Invalid configuration.")

    # -------------------------------
    # Ensure directories exist.
    os.makedirs(config.datasets_dir, exist_ok=True)
    os.makedirs(config.raw_data_dir, exist_ok=True)
    os.makedirs(config.results_dir,  exist_ok=True)
    os.makedirs(config.run_dir,      exist_ok=True)
    os.makedirs(config.tb_dir,       exist_ok=True)
    if config.is_train:
        os.makedirs(config.tb_run_dir,   exist_ok=False)
    os.makedirs(config.cp_load_dir,  exist_ok=True)
    os.makedirs(config.cp_save_dir,  exist_ok=True)
    os.makedirs(config.eval_im_dir,  exist_ok=True)
    os.makedirs(config.metrics_dir,  exist_ok=True)

    # -------------------------------
    # Download raw data.
    if args.download:
        print("Initiating download.")
        download_data.main()
        print("Completed download.")

    # -------------------------------
    # Create dataset from raw data.
    if args.dataset:
        print("Initiating dataset creation.")
        datasets.main()
        print("Completed dataset creation.")

    # -------------------------------
    # Train ESRGAN.
    if args.train:
        print("Initiating training.")
        train.train()
        print("Completed training.")

    # -------------------------------
    if args.test:
        print("Initiating testing.")

        # Build networks.
        model = ESRGAN()

        # Load from save.
        if config.load_model_from_save:
            print(f"Loading model from from saves. G: {config.generator_load_path}, D: {config.discriminator_load_path}")
            _, _ = model.load_model(
                generator_load_path     = config.generator_load_path,
                discriminator_load_path = config.discriminator_load_path,
                state_load_path         = None
            )
            model.G.eval()
            model.D.eval()
            print(f"Loading model state. State: {config.state_load_path}")
            _, _ = model.load_model(
                generator_load_path     = None,
                discriminator_load_path = None,
                state_load_path         = config.state_load_path
            )
            print("Completed loading model.")

        # Perform evaluation.
        evaluate.evaluate(
            model       = model,
            do_val      = False,
            do_test     = True,
            num_visuals = -1, # TODO: Explain this choice of value.
            comparisons = config.test_comparisons,
            metrics     = config.test_metrics,
            call_tag    = 'test',
            tb_writer   = None
        )

        print("Completed testing.")

    # -------------------------------
    # Use ESRGAN to make predictions.
    if args.use:
        print("Prediction is currently not implemented.") # TODO: Implement prediction in 'predict.py'

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------