import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from datetime import datetime
import matplotlib.pyplot as plt


#Set up model and params
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

filename_to_source_tesla = {
    'cnbc_Tesla_news.csv': 'CNBC',
    'cnn_Tesla_news.csv': 'CNN',
    'forbes_Tesla_news.csv': 'Forbes',
    'wallstreetbets_tesla_tsla.csv': 'WallStreetBets',
    'investing_tesla_tsla.csv': 'Investing',
    'stocks_tesla_tsla.csv': 'Stocks'
}

filename_to_source_nvidia = {
    'cnbc_Nvidia_news.csv': 'CNBC',
    'cnn_Nvidia_news.csv': 'CNN',
    'forbes_Nvidia_news.csv': 'Forbes',
    'investing_nvidia_nvda.csv': 'Investing',
    'stocks_nvidia_nvda.csv': 'Stocks',
    'wallstreetbets_nvidia_nvda.csv': 'WallStreetBets'
}

filename_to_source_sp500 = {
    'cnbc_market_news.csv': 'CNBC',
    'cnn_market_news.csv': 'CNN',
    'forbes_market_news.csv': 'Forbes',
    'investing_market_spy.csv': 'Investing',
    'stocks_market_spy.csv': 'Stocks',
    'wallstreetbets_market_spy.csv': 'WallStreetBets'
}

filename_to_source_google = {
    'cnbc_Google_news.csv': 'CNBC',
    'cnn_Google_news.csv': 'CNN',
    'forbes_Google_news.csv': 'Forbes',
    'wallstreetbets_google_goog.csv': 'WallStreetBets',
    'investing_google_goog.csv': 'Investing',
    'stocks_google_goog.csv': 'Stocks'
}

color_map = {
    'CNBC': 'blue',
    'CNN': 'orange',
    'Forbes': 'green',
    'WallStreetBets': 'red',
    'Investing': 'purple',
    'Stocks': 'brown'
}

def calculate_sentiment(dataset_path, filename_to_source):
    net_sentiment_scores = {}

    for filename in os.listdir(dataset_path):
        if filename.endswith('.csv') and filename in filename_to_source:
            file_path = os.path.join(dataset_path, filename)
            df = pd.read_csv(file_path, header=None, names=['headline'])

            print(f"\nProcessing {filename}...")

            sentiment_totals = {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 0.0}
            valid_count = 0

            for line in df['headline'].dropna():
                try:
                    result = nlp(line[:512])[0]
                    label = result['label']
                    score = result['score']
                    sentiment_totals[label] += score
                    valid_count += 1
                except Exception as e:
                    print(f"Error processing line: {line}\n{e}")

            if valid_count > 0:
                net_score = (sentiment_totals['Positive'] - sentiment_totals['Negative']) / valid_count
            else:
                net_score = 0.0

            source_name = filename_to_source[filename]
            net_sentiment_scores[source_name] = net_score
            print(f"Net sentiment score for {source_name}: {net_score:.4f}")

    return net_sentiment_scores


#Calculate sentiment and volatility
dataset_path = 'dataset'
net_sentiment_scores_tesla = calculate_sentiment(dataset_path, filename_to_source_tesla)
net_sentiment_scores_nvidia = calculate_sentiment(dataset_path, filename_to_source_nvidia)
net_sentiment_scores_sp500 = calculate_sentiment(dataset_path, filename_to_source_sp500)
net_sentiment_scores_google = calculate_sentiment(dataset_path, filename_to_source_google)

tesla_dates_april = ["Apr 14, 2025", "Apr 15, 2025", "Apr 16, 2025", "Apr 17, 2025", "Apr 21, 2025"]
tesla_prices_april = [252.35, 254.11, 241.55, 241.37, 227.50]
tesla_datetime_april = [datetime.strptime(date, "%b %d, %Y") for date in tesla_dates_april]

nvidia_dates_april = ["Apr 14, 2025", "Apr 15, 2025", "Apr 16, 2025", "Apr 17, 2025", "Apr 21, 2025"]
nvidia_prices_april = [110.71, 112.20, 104.49, 101.49, 96.91]
nvidia_datetime_april = [datetime.strptime(date, "%b %d, %Y") for date in nvidia_dates_april]

sp500_dates_april = ["Apr 14, 2025", "Apr 15, 2025", "Apr 16, 2025", "Apr 17, 2025", "Apr 21, 2025"]
sp500_prices_april = [5405.97, 5396.63, 5275.70, 5282.70, 5158.20]
sp500_datetime_april = [datetime.strptime(date, "%b %d, %Y") for date in sp500_dates_april]

google_dates_april = ["Apr 14, 2025", "Apr 15, 2025", "Apr 16, 2025", "Apr 17, 2025", "Apr 21, 2025"]
google_prices_april = [161.47, 158.68, 155.50, 153.36, 149.86]
google_datetime_april = [datetime.strptime(date, "%b %d, %Y") for date in google_dates_april]

price_changes_tesla = [abs(tesla_prices_april[i + 1] - tesla_prices_april[i]) for i in range(3)]
avg_abs_change_tesla = sum(price_changes_tesla) / len(price_changes_tesla)

price_changes_nvidia = [abs(nvidia_prices_april[i + 1] - nvidia_prices_april[i]) for i in range(3)]
avg_abs_change_nvidia = sum(price_changes_nvidia) / len(price_changes_nvidia)

price_changes_sp500 = [abs(sp500_prices_april[i + 1] - sp500_prices_april[i]) for i in range(3)]
avg_abs_change_sp500 = sum(price_changes_sp500) / len(price_changes_sp500)

price_changes_google = [abs(google_prices_april[i + 1] - google_prices_april[i]) for i in range(3)]
avg_abs_change_google = sum(price_changes_google) / len(price_changes_google)


#Predict stock prices
apr_17_price_tesla = tesla_prices_april[3]
predicted_prices_tesla = {
    source: apr_17_price_tesla + (sentiment * avg_abs_change_tesla * 10)
    for source, sentiment in net_sentiment_scores_tesla.items()
}

apr_17_price_nvidia = nvidia_prices_april[3]
predicted_prices_nvidia = {
    source: apr_17_price_nvidia + (sentiment * avg_abs_change_nvidia * 10)
    for source, sentiment in net_sentiment_scores_nvidia.items()
}

apr_17_price_sp500 = sp500_prices_april[3]
predicted_prices_sp500 = {
    source: apr_17_price_sp500 + (sentiment * avg_abs_change_sp500 * 20)
    for source, sentiment in net_sentiment_scores_sp500.items()
}

apr_17_price_google = google_prices_april[3]
predicted_prices_google = {
    source: apr_17_price_google + (sentiment * avg_abs_change_google * 10)
    for source, sentiment in net_sentiment_scores_google.items()
}

#Plot figures
plt.figure(figsize=(10, 6))
plt.plot(tesla_datetime_april, tesla_prices_april, 'b-', linewidth=2, label='True Tesla Stock Price')
apr_21_date_tesla = tesla_datetime_april[-1]
for source, pred_price in predicted_prices_tesla.items():
    plt.scatter(apr_21_date_tesla, pred_price, color=color_map.get(source, 'gray'), label=f'{source} Sentiment Pred. (Tesla)')
plt.title('Tesla Stock Price with Sentiment-Based Predictions (April 2025)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(False)
plt.tick_params(axis='both', which='both', length=0)
plt.xticks(tesla_datetime_april, [d.strftime("%b %d") for d in tesla_datetime_april], rotation=0)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("Tesla_Stock_April_with_Sentiment_Predictions.png")

plt.figure(figsize=(10, 6))
plt.plot(nvidia_datetime_april, nvidia_prices_april, 'g-', linewidth=2, label='True Nvidia Stock Price')
apr_21_date_nvidia = nvidia_datetime_april[-1]
for source, pred_price in predicted_prices_nvidia.items():
    plt.scatter(apr_21_date_nvidia, pred_price, color=color_map.get(source, 'gray'), label=f'{source} Sentiment Pred. (Nvidia)')
plt.title('Nvidia Stock Price with Sentiment-Based Predictions (April 2025)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(False)
plt.tick_params(axis='both', which='both', length=0)
plt.xticks(nvidia_datetime_april, [d.strftime("%b %d") for d in nvidia_datetime_april], rotation=0)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("Nvidia_Stock_April_with_Sentiment_Predictions.png")

plt.figure(figsize=(10, 6))
plt.plot(sp500_datetime_april, sp500_prices_april, 'r-', linewidth=2, label='True S&P 500 Stock Price')
apr_21_date_sp500 = sp500_datetime_april[-1]
for source, pred_price in predicted_prices_sp500.items():
    plt.scatter(apr_21_date_sp500, pred_price, color=color_map.get(source, 'gray'), label=f'{source} Sentiment Pred. (S&P 500)')
plt.title('S&P 500 Stock Price with Sentiment-Based Predictions (April 2025)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(False)
plt.tick_params(axis='both', which='both', length=0)
plt.xticks(sp500_datetime_april, [d.strftime("%b %d") for d in sp500_datetime_april], rotation=0)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("SP500_Stock_April_with_Sentiment_Predictions.png")

plt.figure(figsize=(10, 6))
plt.plot(google_datetime_april, google_prices_april, 'm-', linewidth=2, label='True Google Stock Price')
apr_21_date_google = google_datetime_april[-1]
for source, pred_price in predicted_prices_google.items():
    plt.scatter(apr_21_date_google, pred_price, color=color_map.get(source, 'gray'), label=f'{source} Sentiment Pred. (Google)')
plt.title('Google Stock Price with Sentiment-Based Predictions (April 2025)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(False)
plt.tick_params(axis='both', which='both', length=0)
plt.xticks(google_datetime_april, [d.strftime("%b %d") for d in google_datetime_april], rotation=0)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("Google_Stock_April_with_Sentiment_Predictions.png")

def save_predictions_to_txt(stock_name, true_price, predicted_prices_dict):
    filename = f"{stock_name}_Predictions.txt"

    traditional_sources = ['CNBC', 'CNN', 'Forbes']
    reddit_sources = ['WallStreetBets', 'Investing', 'Stocks']

    traditional_diffs = []
    reddit_diffs = []

    with open(filename, "w") as f:
        f.write(f"{stock_name} Stock Price Predictions - April 21, 2025\n")
        f.write(f"True Price: ${true_price:.2f}\n\n")
        f.write("Predicted Prices by Source:\n")

        for source, pred_price in predicted_prices_dict.items():
            percent_diff = ((pred_price - true_price) / true_price) * 100
            abs_percent_diff = abs(percent_diff)
            f.write(f"{source}: ${pred_price:.2f} ({percent_diff:+.2f}%)\n")

            if source in traditional_sources:
                traditional_diffs.append(abs_percent_diff)
            elif source in reddit_sources:
                reddit_diffs.append(abs_percent_diff)

        avg_traditional = sum(traditional_diffs) / len(traditional_diffs) if traditional_diffs else 0.0
        avg_reddit = sum(reddit_diffs) / len(reddit_diffs) if reddit_diffs else 0.0

        f.write("\n")
        f.write(f"Average Absolute % Difference (Traditional Sources): {avg_traditional:.2f}%\n")
        f.write(f"Average Absolute % Difference (Reddit Sources): {avg_reddit:.2f}%\n")

    print(f"Saved predictions to {filename}")

#Save predictions
save_predictions_to_txt("Tesla", tesla_prices_april[-1], predicted_prices_tesla)
save_predictions_to_txt("Nvidia", nvidia_prices_april[-1], predicted_prices_nvidia)
save_predictions_to_txt("SP500", sp500_prices_april[-1], predicted_prices_sp500)
save_predictions_to_txt("Google", google_prices_april[-1], predicted_prices_google)
