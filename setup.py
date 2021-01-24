import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchlibrosa",
    version="0.0.6",
    author="Qiuqiang Kong",
    author_email="qiuqiangkong@gmail.com",
    description="PyTorch implemention of part of librosa functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiuqiangkong/torchlibrosa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
