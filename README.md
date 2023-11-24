# Diffusion in PyTorch

Implementation of diffusion in pure PyTorch.

![](readme_figures/examples/000.png)
![](readme_figures/examples/001.png)
![](readme_figures/examples/002.png)
![](readme_figures/examples/003.png)
![](readme_figures/examples/004.png)
![](readme_figures/examples/005.png)
![](readme_figures/examples/006.png)
![](readme_figures/examples/007.png)
![](readme_figures/examples/008.png)
![](readme_figures/examples/009.png)
![](readme_figures/examples/010.png)
![](readme_figures/examples/011.png)
![](readme_figures/examples/012.png)
![](readme_figures/examples/013.png)
![](readme_figures/examples/014.png)
![](readme_figures/examples/015.png)
![](readme_figures/examples/016.png)
![](readme_figures/examples/017.png)
![](readme_figures/examples/018.png)
![](readme_figures/examples/019.png)

Adapted from: [https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/b91507a769bc40d87f1428f3eabba11dda6ea8c0/notebooks/08_diffusion/01_ddm/ddm.ipynb#L104](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/b91507a769bc40d87f1428f3eabba11dda6ea8c0/notebooks/08_diffusion/01_ddm/ddm.ipynb#L104)

## Running

* Data - download from: [https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset). Let `/path/to/data` be the path to the folder that contains the `train` and `test` folders.

* Training:

    ```bash
    python main.py --conf conf.yml --command train --mnt-dir /path/to/data
    ```

* Generate samples:

    ```bash
    python main.py --conf conf.yml --command generate
    ```

* Plot loss:

    ```bash
    python main.py --conf conf.yml --command loss --show
    ```

![](readme_figures/loss.png)
