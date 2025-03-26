import time
import os
import mysql.connector
import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Retrieve credentials from the .env file
DB_HOST = os.getenv("host")
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_NAME = os.getenv("database")
INVESTING_EMAIL = os.getenv("investing_email")
INVESTING_PASSWORD = os.getenv("investing_password")

# Connect to MySQL and set up the database connection
def login_and_scrape_gold_news():
    # Try to connect to the MySQL database (scraped_data)
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
    except mysql.connector.Error as e:
        print("Error connecting to MySQL:", e)
        return

    # Launch an undetected Chrome browser instance (so sites don't block us)
    driver = uc.Chrome(use_subprocess=True)

    try:
        driver.get("https://www.investing.com/")
        time.sleep(5)

        # Click the cookie consent button
        try:
            cookie_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_btn.click()
            time.sleep(1)
        except:
            print("Cookie banner not there or already handled.")

        # Close the sign-up popup
        try:
            close_popup_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "svg[data-test='sign-up-close']"))
            )
            close_popup_btn.click()
            time.sleep(1)
        except Exception as e:
            print("Could not close the pop-up, maybe it never showed up:", e)

        # Click "Sign In" button
        try:
            login_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-test='login-btn']"))
            )
            login_button.click()
        except Exception as e:
            print("Unable to click the initial Sign In button:", e)
            driver.quit()
            return

        # Click on "Sign in with Email"
        time.sleep(2)
        try:
            sign_in_with_email_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Sign in with Email')]"))
            )
            sign_in_with_email_btn.click()
        except Exception as e:
            print("Failed to click 'Sign in with Email':", e)
            driver.quit()
            return

        # Enter the login credentials: email and password
        time.sleep(2)
        try:
            email_input = driver.find_element(By.NAME, "email")
            password_input = driver.find_element(By.NAME, "password")
            email_input.send_keys(INVESTING_EMAIL)
            password_input.send_keys(INVESTING_PASSWORD)
        except Exception as e:
            print("Could not fill in login details:", e)
            driver.quit()
            return

        # Submit the login form by clicking the submit button
        try:
            submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_button.click()
        except Exception as e:
            print("Failed to click final Sign In:", e)
            driver.quit()
            return

        time.sleep(5)
        print("Login attempt complete.")

        # Revisit the Gold News page
        driver.get("https://www.investing.com/commodities/gold-news")
        time.sleep(5)

        # Scrape all articles on the current page and insert them into the DB
        def scrape_current_page_articles():
            # Find all article links on the page
            try:
                article_links = driver.find_elements(
                    By.CSS_SELECTOR, "a[data-test='article-title-link']"
                )
            except Exception as e:
                print("Error finding article links:", e)
                article_links = []

            links = []
            for link_el in article_links:
                href = link_el.get_attribute("href")
                if href and href not in links:
                    links.append(href)

            print(f"Found {len(links)} article links on this page.")

            for link in links:
                print(f"\nScraping article: {link}")
                driver.get(link)
                time.sleep(3)

                # Grab title
                try:
                    title_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "h1"))
                    )
                    title = title_element.text.strip()
                except:
                    title = "Title Not Found"

                # Capture the publish date from the page
                publish_date = "Not Found"
                try:
                    publish_elem = driver.find_element(By.XPATH, "//span[contains(text(), 'Published')]")
                    publish_date = publish_elem.text.strip()
                except:
                    pass

                # Determine the content by checking candidate containers
                candidate_selectors = [
                    "div.article_WYSIWYG__0uwh.article_articlePage__UMz3q",
                    "div.article_container"
                ]
                found_container = None
                for sel in candidate_selectors:
                    try:
                        container = driver.find_element(By.CSS_SELECTOR, sel)
                        if container:
                            found_container = container
                            break
                    except:
                        pass
                if not found_container:
                    found_container = driver.find_element(By.TAG_NAME, "body")

                # Extract all paragraphs from the container
                paragraphs = found_container.find_elements(By.TAG_NAME, "p")
                content_list = [p.text.strip() for p in paragraphs if p.text.strip()]
                content = "\n\n".join(content_list) if content_list else "No content found"

                print("TITLE:", title)
                print("PUBLISH DATE:", publish_date)
                print("CONTENT (snippet):", content[:100], "...")

                # Insert the scraped data into the articles table in MySQL
                insert_query = """
                    INSERT INTO articles (title, link, content, published_date)
                    VALUES (%s, %s, %s, %s)
                """
                try:
                    cursor.execute(insert_query, (title, link, content, publish_date))
                    conn.commit()
                    print("Inserted into DB.")
                except mysql.connector.Error as e:
                    print("DB Insert Error:", e)
                    conn.rollback()

        # Scrape the main Gold News page (page 1)
        print("Scraping the main Gold News page (page 1)...")
        scrape_current_page_articles()

        # Loop through pages 2 to 11 and scrape each page
        for page_num in range(2, 12):
            next_page_url = f"https://www.investing.com/commodities/gold-news/{page_num}"
            print(f"\nVisiting page {page_num}: {next_page_url}")
            driver.get(next_page_url)
            time.sleep(4)
            scrape_current_page_articles()

    finally:
        # Clean up: close the DB cursor, connection, and the browser
        cursor.close()
        conn.close()
        driver.quit()

if __name__ == "__main__":
    login_and_scrape_gold_news()
