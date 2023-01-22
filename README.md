# Analysis and figure generation for XSwap project

This repository hosts notebooks, images, and data for the XSwap project.
The results of these analyses are described in the manuscript titled:

[**The probability of edge existence due to node degree: a baseline for network-based predictions**](https://github.com/greenelab/xswap-manuscript/)<br>

Along with the manuscript, this project produced an open-source implementation of our network permutation algorithm.
Available on PyPI as [`xswap`](https://pypi.org/project/xswap/) and source on GitHub in [hetio/xswap](https://github.com/hetio/xswap/).

## Layout

The analyses for this repository are performed by sequentially numbered Jupyter notebooks in the [`nb`](nb) directory.
Data is written to the [`data`](data) directory and figures are exported to the [`img`](img) directory.

The analyses depend on the [Hetionet HetMat dataset](https://github.com/hetio/hetionet/tree/master/hetnet/matrix),
which can be downloaded by running the following Python command from this repo's root directory:

```py
from hetmatpy.hetmat.archive import load_archive
load_archive(
    archive_path="https://github.com/hetio/hetionet/raw/b467b8b41087288390b41fdb796577ada9f03bda/hetnet/matrix/hetionet-v1.0.hetmat.zip",
    destination_dir="data/task1/hetionet-v1.0.hetmat",
)
```

## Environment

This repository uses [conda](http://conda.pydata.org/docs/) to manage its environment as specified in [`environment.yml`](environment.yml).
Install the environment with:

```sh
# install new xswap-analysis environment
conda env create --file=environment.yml

# update existing xswap-analysis environment
conda env update --file=environment.yml
```

Then use `conda activate xswap-analysis` and `conda deactivate` to activate or deactivate the environment.

## License

The entire repository is released under a [BSD 3-Clause License](LICENSE).
Furthermore:

- the contents of the [`data`](data) directory are released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication).
- the contents of the [`img`](img) directory are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (public domain dedication).
  For images that are used in the [xswap-manuscript](https://github.com/greenelab/xswap-manuscript/),
  please attribute this manuscript as the source.
