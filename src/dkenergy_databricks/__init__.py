"""Databricks deployment of the DK1/DK2 day-ahead price forecaster.

The modules here are plain pandas code so they can be tested locally. Only
``spark_io`` touches Spark, and the job entry points under ``databricks/jobs``
are thin wrappers that read and write Delta tables around these functions.
"""
