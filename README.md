## Python implementation of DeepGL

About
-----
* Python3 implemetation of DeepGL.
  * This package is implemented for:
    Fujiwara et al., Network Comparison with Interpretable Contrastive Network Representation Learning, JDSSV, 2022.
  * Related repository: Contrastive Network Representation Learning (cNRL), https://github.com/takanori-fujiwara/cnrl
  * Original DeepGL paper: Rossi et al., Deep Inductive Graph Representation Learning, IEEE TKDE, 2018.

* Current implementation supports a major portion of DeepGL. However, for example, local graphlet count-based features are not supported. These functionality will be tentatively implemented in the future.

******

Requirements
-----
* Python3
* graph-tool (https://graph-tool.skewed.de/)
* OS: macOS or Linux
  * Note: Tested on macOS BigSur and Ubuntu 20.0.4 LTS.
  * Windows is not supported because graph-tool is not available for Windows.
******

Setup
-----
* Install graph-tool (https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions)
  * For example, macOS X with Homebrew,

    `brew install graph-tool`

* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

******

Usage
-----
* Import installed modules from python (e.g., `from deepgl import DeepGL`). See sample.py for examples.
* For detailed documentations, please see doc/index.html or directly see comments in deepgl.py.

******

How to Cite
----
* If you use this implementation of DeepGL, please consider to cite: Fujiwara et al., Network Comparison with Interpretable Contrastive Network Representation Learning, JDSSV, 2022.
