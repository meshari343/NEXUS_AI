import uvicorn
from nexus_ai.sentence_sentiment_analysis.BERT import BertClassifier

if __name__ == '__main__':
    uvicorn.run('app:app',reload=True, host='0.0.0.0')
