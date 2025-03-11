from setuptools import setup, find_packages

setup(
    name="xai_dashboard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas==1.5.0",
        "numpy==1.24.0",
        "scikit-learn==1.2.1",
        "joblib==1.2.0",
        "shap==0.41.0",
        "streamlit==1.14.0",
        "pytest==7.2.0",
        "pytest-cov==4.0.0",
    ],
    author="Your Name",
    author_email="[email protected]",
    description="An Explainable AI Dashboard for Classification Models",
    url="https://github.com/<YOUR-ORG>/XAI_Dashboard",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
)
