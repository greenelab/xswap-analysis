# Analysis and figure generation for XSwap project

This repository hosts notebooks, images, and data for the XSwap project.
The results of these analyses are described in the manuscript titled:

[**The probability of edge existence due to node degree: a baseline for network-based predictions**](https://github.com/greenelab/xswap-manuscript/)<br>

Along with the manuscript, this project produced an open-source implementation of our network permutation algorithm.
Available on PyPI as [`xswap`](https://pypi.org/project/xswap/) and source on GitHub in [hetio/xswap](https://github.com/hetio/xswap/).

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
