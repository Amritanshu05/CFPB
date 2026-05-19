from setuptools import setup, find_packages

setup(
    name="cfpb_assistant",
    version="0.1.0",
    description="CFPB Complaint-to-Resolution Assistant — research pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
