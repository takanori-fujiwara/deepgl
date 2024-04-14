from setuptools import setup

setup(
    name="deepgl",
    version=0.02,
    packages=[""],
    package_dir={"": "."},
    install_requires=["numpy", "scikit-learn"],
    py_modules=["deepgl", "deepgl_utils"],
)
