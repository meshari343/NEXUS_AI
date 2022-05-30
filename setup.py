from setuptools import setup

setup(
   name='nexus_ai',
   version='1.0',
   description='AI backend for nexus analytics website',
   author='meshari aldossari',
   author_email='meshari34343@gmail.com',
   packages=[
      'nexus_ai',
      'nexus_ai.ABSA',
      'nexus_ai.sentence_sentiment_analysis',
      'nexus_ai.time_series_forecasting',
      'nexus_ai.utilities',
      'nexus_ai.sentence_sentiment_analysis.english',
      'nexus_ai.sentence_sentiment_analysis.arabic',
      'nexus_ai.ABSA.english',
      'nexus_ai.ABSA.arabic',
   ]
)