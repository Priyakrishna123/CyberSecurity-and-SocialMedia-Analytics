from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import csv

# Take inputs from the user first
username = input("Enter your username: ")
profile_url = input("Enter the profile URL of the account: ")

# Set up WebDriver and navigate to Twitter login page after taking inputs
driver = webdriver.Chrome()
driver.get("https://twitter.com/login")

# Log in manually or with automated input

def monitor_tweets(username, profile_url, scroll_limit=5):
    # Navigate to the specified Twitter profile
    driver.get(profile_url)
    
    # Create a CSV file to store tweets
    with open(f"{username}_tweets.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tweet"])

        # Continuously scroll and check for new tweets
        for _ in range(scroll_limit):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(3)

            # Extract tweet content
            tweets = driver.find_elements(By.CSS_SELECTOR, "div[lang]")  # Updated selector syntax
            for tweet in tweets:
                tweet_text = tweet.text
                writer.writerow([tweet_text])  # Write tweet text to CSV
                print(tweet_text)  # Optional: Print the tweet text
            sleep(30)  # Wait between polls to avoid rate limits

monitor_tweets(username, profile_url)
