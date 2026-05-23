from setuptools import setup, find_packages

setup(
    name="rediris",
    version="1.0.0",
    description="Red Iris candidate runtime foundation",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "sqlalchemy>=2.0.23",
        "psycopg2-binary>=2.9.9",
    ],
    python_requires=">=3.10",
)
