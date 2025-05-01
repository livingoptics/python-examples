# Working with Living Optics Datasets

Examples showcasing how to use and develop on top of datasets produced by the Living Optics Data Exploration Tool or downloadable from the Living Optics cloud.

These examples use the [Living Optics SDK](https://cloud.livingoptics.com/shared-resources?file=software/lo_sdk-1.6.0-dist.tgz) and the [Living Optics dataset reader](https://github.com/livingoptics/datareader).

## Getting Started

- New to Living Optics? [Sign up here](https://cloud.livingoptics.com/register) for a free Basic cloud account.
- Download the [Living Optics SDK](https://cloud.livingoptics.com/shared-resources?file=software/lo_sdk/free_tier).
- Download the [a hyperspectral dataset](https://cloud.livingoptics.com/shared-resources?file=annotated-datasets) for training and testing models.

> 📢 **Note:** Access to the dataset and SDK requires at least a Basic cloud tier subscription.


## Install guide:

For SDK installation help, please refer to the [installation guide](https://cloud.livingoptics.com/shared-resources?file=docs/ebooks/install-sdk.pdf).

To setup the dataset reader run:

```bash
source venv/bin/activate
git clone git@github.com:livingoptics/datareader.git
pip install -r ./datareader/requirements.txt
pip install -r ./requirements.txt
export PYTHONPATH="/path/to/your/folder:$PYTHONPATH"
```


## Run notebooks:

```bash
source venv/bin/activate
jupyter lab --notebook-dir=.
```

## Example Notebooks

| Notebook                                                           | Description                                                          |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| [Training a Classifier](./notebook_example_classification.ipynb)   | Train a basic spectral classifier using the example dataset.         |
| [Training a Regression Model](./notebook_example_regression.ipynb) | Train a model to estimate sugar content directly from spectral data. |

## Training a Classifier

This tutorial demonstrates how to train a binary classifier on an annotated dataset to detect grapes.

Further to this, KMeans clustering is used to group the grapes by their sugar content,
and a further classifier is trained to demonstrate how hyperspectral data can be used to classify the sugar content of grapes.

## Predicting sugar content of grapes using hyperspectral data

This tutorial demonstrates
how hyperspectral data annotated with the [Living optics Data Exploration Tool](https://docs.livingoptics.com/product/data-exploration-tool) can be used
to train a regression model to predict the sugar content of grapes.

---

## Helpful Resources

- [Develfroper Portal](https://developer.livingoptics.com/)
- [Product Documentation](https://docs.livingoptics.com/) (requires Basic cloud tier access)

## Support

Need help?  
Get in touch with [Living Optics Support](https://www.livingoptics.com/support).
## Contribution Guidelines
We welcome contributions to enhance this project. Please follow these steps to contribute:

**Fork the Repository**: Create a fork of this repository to your GitHub account.

**Create a Branch**: Create a new branch for your changes.
**Make Changes**: Make your changes and commit them with a clear and descriptive message.

**Create a Pull Request**: Submit a pull request with a description of your changes.

## Support

For any questions, contact us at [Living Optics support](https://www.livingoptics.com/support).