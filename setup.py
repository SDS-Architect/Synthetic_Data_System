import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SDS",  # Replace with your own username
    version="0.1a",
    author="Euan Gardner",
    author_email="euan.gardner@nhs.net",
    description="Designed to create synthetic medical data and other data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "jupyter",
        "catboost",
        "pyyaml",
        "psutil",
    ],
    entry_points={
        "console_scripts": [
            "synthesise = SDS.src.Synth_It_So:synthesis_activation"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: [Linux, Windows]",
    ],
    python_requires=">=3.5",
)
