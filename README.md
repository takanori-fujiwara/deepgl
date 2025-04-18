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
  * Note: Tested on macOS Sonoma and Ubuntu 20.0.4 LTS.
  * Windows is not supported because graph-tool is not available for Windows.
******

Setup
-----
* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

* Install graph-tool (https://graph-tool.skewed.de/installation.html)
  * For example, macOS with Homebrew (when not using virtual environment),

    `brew install graph-tool`
  
  * When using virtual environment, there are two options:
  
    - Option 1. Follow the graph-tool instruction (need a lot of time for compiling graph-tool).

      - Check a section of "Installing in a virtualenv".

      - graph-tool's instruction doesn't support Python3.12. For Python3.12, before the configure step (i.e., ./configure --prefix=$HOME/.local), run commands below:

        `pip3 install setuptools pycairo`

    - Option 2. Use virtual environment with "include-system-site-packages = true"
      
      - For example, edit "pyenv.cfg" to be `include-system-site-packages = true`

      - then
        
        `brew install graph-tool`

******

Usage
-----
* Import installed modules from python (e.g., `from deepgl import DeepGL`). See sample.py for examples.
* For detailed documentations, please see doc/index.html or directly see comments in deepgl.py.

******

How to Cite
----
* If you use this implementation of DeepGL, please consider to cite: Fujiwara et al., Network Comparison with Interpretable Contrastive Network Representation Learning, JDSSV, 2022.
