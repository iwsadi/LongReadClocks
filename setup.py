from setuptools import setup, find_packages

setup(
    name="longreadclock",
    version="1.0.0",
    description="Cell-type-resolved epigenetic aging clocks from bulk long-read methylation data",
    author="Yinjie Wu, Mahdi Moqri, Vadim N. Gladyshev",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "gcs": ["google-cloud-storage>=2.0"],
        "bigquery": ["pandas-gbq>=0.19"],
    },
)
