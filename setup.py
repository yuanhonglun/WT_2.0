from setuptools import setup, find_packages

setup(
    name="WT_2",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matchms",
        "huggingface-hub",
        "torch"
    ],
    author="liuzhenhuan",
    author_email="867245824@qq.com",
    description="WT2 Mass spectrometry data processing library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
