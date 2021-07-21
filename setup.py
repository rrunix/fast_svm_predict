import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="fast_svm_predict",
    version="0.0.1",
    description="Fast SVM predictions",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rrunix/fast_svm_predict",
    author="Ruben Rodriguez",
    author_email="ruben.rrf93@gmail.com",
    license="MIT",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["fast_svm_predict"],
    include_package_data=True,
    install_requires=["jax", "jaxlib", "numpy"],
)
