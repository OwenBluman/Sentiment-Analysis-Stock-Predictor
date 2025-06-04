import praw
import csv
import os
import requests

REDDIT_CLIENT_ID = 'redacted'
REDDIT_CLIENT_SECRET = 'redacted'
REDDIT_USERNAME = 'redacted'
SUBREDDIT_NAMES = ['wallstreetbets', 'investing', 'stocks']
REDDIT_POST_LIMIT = 1000
REDDIT_CATEGORIES = {
    "market_spy": ["market", "spy"],
    "tesla_tsla": ["tesla", "tsla"],
    "nvidia_nvda": ["nvidia", "nvda"],
    "google_goog": ["google", "goog"]
}

NEWS_API_KEY = 'redacted'
NEWS_DOMAINS_TO_SEARCH = ['cnbc.com','cnn.com','forbes.com']
NEWS_SEARCH_TERMS = ["market", "Tesla", "Nvidia", "Google"]

STOCK_VETTING_KEYWORDS = {
    "Tesla": ["tesla", "tsla"],
    "Nvidia": ["nvidia", "nvda"],
    "Google": ["google", "goog", "alphabet"]
}

OUTPUT_DIR = "dataset"


def connect_to_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=f"script by u/{REDDIT_USERNAME}"
    )
    reddit.read_only = True
    _ = reddit.config.user_agent
    return reddit

def get_recent_post_titles(subreddit_names, post_limit=10):
    reddit = connect_to_reddit()
    subreddit_titles = {}
    for name in subreddit_names:
        print(f"Scraping r/{name}")
        subreddit = reddit.subreddit(name)
        recent_posts = list(subreddit.new(limit=post_limit))
        titles = [post.title for post in recent_posts]
        subreddit_titles[name] = titles
    return subreddit_titles

def categorize_and_save_reddit_titles(subreddit_title_map, categories):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for subreddit_name, titles in subreddit_title_map.items():
        filenames = {key: f"{subreddit_name}_{key}.csv" for key in categories}
        files = {}
        writers = {}
        full_paths = {key: os.path.join(OUTPUT_DIR, fname) for key, fname in filenames.items()}

        for key, full_path in full_paths.items():
            files[key] = open(full_path, mode='w', newline='', encoding='utf-8')
            writers[key] = csv.writer(files[key])

        for title in titles:
            title_lower = title.lower()
            for category_key, keywords in categories.items():
                if any(keyword.lower() in title_lower for keyword in keywords):
                    writers[category_key].writerow([title])

        for key in files:
             files[key].close()


def fetch_and_save_news(api_key, search_query, domain):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    domain_prefix = domain.split('.')[0]
    safe_search_query = "".join(c if c.isalnum() else "_" for c in search_query)
    csv_filename = f"{domain_prefix}_{safe_search_query}_news.csv"
    full_path = os.path.join(OUTPUT_DIR, csv_filename)

    base_url = "https://newsapi.org/v2/everything"
    params = {
        'q': search_query,
        'domains': domain,
        'sortBy': 'publishedAt',
        'apiKey': api_key,
        'language': 'en',
        'pageSize': 100
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    articles = data['articles']

    articles_saved_count = 0
    with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for article in articles:
            title = article['title']
            title_lower = title.lower()
            needs_vetting = search_query in STOCK_VETTING_KEYWORDS
            should_save = False

            if not needs_vetting:
                should_save = True
            else:
                required_keywords = STOCK_VETTING_KEYWORDS[search_query]
                if any(keyword.lower() in title_lower for keyword in required_keywords):
                    should_save = True

            if should_save:
                writer.writerow([title])
                articles_saved_count += 1


if __name__ == "__main__":

    fetched_reddit_data = get_recent_post_titles(SUBREDDIT_NAMES, REDDIT_POST_LIMIT)
    categorize_and_save_reddit_titles(fetched_reddit_data, REDDIT_CATEGORIES)

    for domain in NEWS_DOMAINS_TO_SEARCH:
         print(f"Scraping {domain}")
         for term in NEWS_SEARCH_TERMS:
             fetch_and_save_news(NEWS_API_KEY, term, domain)

    print("Complete")