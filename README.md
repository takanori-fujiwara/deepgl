## Python implementation of DeepGL

About
-----
* Python3 implemetation of DeepGL.
  * Rossi et al., Deep Inductive Graph Representation Learning, IEEE TKDE, 2018.

* Current implementation supports a major portion of DeepGL. However, for example, local graphlet count-based features are not supported. These functionality will be tentatively implemented in the future.

******

Requirements
-----
* Python3
* graph-tool (https://graph-tool.skewed.de/)
  * Currently, graph-tool's server is down and cannot install.
* Note: Tested on macOS Catalina and Ubuntu 20.0.4 LTS.
******

Setup
-----
* Install graph-tool (https://graph-tool.skewed.de/)
  * Currently, graph-tool's server is down and cannot install.

* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

******

Usage
-----
* Import installed modules from python (e.g., `from deepgl import DeepGL`). See sample.py for examples.
* For detailed documentations, please see doc/index.html or directly see comments in deepgl.py.
