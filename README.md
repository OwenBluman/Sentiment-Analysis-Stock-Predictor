# Sentiment Analysis Stock Predictor
A program to extract the latest news data from a variety of sources, and then predict stock price based
upon an algorithm weighting FinBERT's sentiment analysis and market volatility.

## Demo
A sample prediction for Google, Tesla, Nvidia and S&P 500 stock price, based upon each news source,
can be found in the image files ending in Predictions.png.

## Instructions
First run collect_data.py to gather a database of news headlines and Reddit post titles. Then alter
the stock prices and dates in main.py to accurately reflect the prediction period. Lastly, run main.py
to get the predicted pricing data. Currently, the program is configured to backtest on a prediction period
in the past, and thus it outputs a variety of statistics relating to prediction accuracy across news sources.

## Paper
A paper detailing the data collection process, prediction algorithm, analysis of findings and societal implications
can be found in the file The Impact of Varied News Sources on FinBERT-based Stock Predictions.pdf.