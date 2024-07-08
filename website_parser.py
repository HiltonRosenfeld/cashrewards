import logging
import requests
from bs4 import BeautifulSoup
import time

import re
import unicodedata


#
# Selenium
#

from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialise Chrome driver
driver = webdriver.Chrome()
#driver.implicitly_wait(10)

def interceptor(request):
    del request.headers['User-Agent']  # Delete the header first
    request.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
# Set the interceptor on the driver
driver.request_interceptor = interceptor



def clean_text(text: str):
    """
    Function to clean text from web pages
    """
    
    # Normalize line breaks to \n\n (two new lines)
    text = text.replace("\r\n", "\n\n")
    text = text.replace("\r", "\n\n")

    # Replace two or more spaces with a single space
    text = re.sub(" {2,}", " ", text)

    # Remove leading spaces before removing trailing spaces
    text = re.sub("^[ \t]+", "", text, flags=re.MULTILINE)

    # Remove trailing spaces before removing empty lines
    text = re.sub("[ \t]+$", "", text, flags=re.MULTILINE)

    # Remove empty lines
    text = re.sub("^\s+", "", text, flags=re.MULTILINE)

    # remove unicode Non Breaking Space
    text = unicodedata.normalize('NFKC', text)

    return text




    
def website_parser(job_url: str):

    """
    This function loads a URL and parses the content using Beautiful Soup. 
    The parsing is specific to the website and based on the CSS of the specific elements we are interested in.
    """

    # Initialise attributes
    item = {}
    item_title = ""
    item_price = ""
    item_img = ""
    key_features = ""
    tech_specs = ""
    item_description = ""
    product_features = ""

    # set HTTP Header User Agent
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

    try:
        # Fetch page content using Request
        """
        page = requests.get(job_url, headers=headers)
        if page.status_code != 200:
            print(f"Failed to retrieve the job posting at {job_url}. Status code: {page.status_code}")
        # Parse the HTML content of the job posting using BeautifulSoup
        soup = BeautifulSoup(page.text, 'html.parser')
        """
        
        # Fetch page content using Selenium
        driver.get(job_url)
        time.sleep(3)
        driver.execute_script('window.scrollBy(0, 1500)')
        time.sleep(3)



        # Parse the HTML content of the job posting using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')


        # Title
        # class="pdp__main-title text-light text-graydark-3 text-center"
        item_title = soup.find('h1', {'class': 'pdp__main-title'})
        if item_title is not None:
            item_title = item_title.text.strip()
        else:
            item_title = ""


        # Brand
        if item_title is not None:
            # set brand to the first word of the title
            brand = item_title.split()[0]


        # Description
        # span class="moreContent"
        item_description = soup.find('span', {'class': 'moreContent'})
        if item_description is not None:
            item_description = item_description.text.strip()
            # strip '...Less' and trailing spaces from the end of the description
            item_description = item_description.replace("...Less", "").strip()
        else:
            # span class="firstContent"
            item_description = soup.find('span', {'class': 'firstContent'})
            if item_description is not None:
                item_description = item_description.text.strip()
                # strip '...Less' and trailing spaces from the end of the description
                item_description = item_description.replace("...Less", "").strip()
            else:
                item_description = ""

        # Price
        # div id="PriceDisplayWidget"
        PriceDisplayWidget = soup.find('div', {'id': 'PriceDisplayWidget'})
        if PriceDisplayWidget is not None:
            item_price = PriceDisplayWidget.find('span')
            if item_price is not None:
                item_price = item_price.text.strip()
                # get the leading $ sign
                item_currency = item_price[0]
                # remove the leading $ sign
                item_price = item_price[1:]
            else:
                item_price = ""
        else:
            item_price = ""


        # Image
        #product_gallery = soup.find('section', {'class': re.compile('product-gallery')})
        slickCarousel = soup.find('section', {'id': 'slickCarousel'})
        if slickCarousel is not None:
            # <img alt='TCL 32" S5400 FHD Android Smart TV 23  32S5400AF' class="product-image-main" data-lazy="//thegoodguys.sirv.com/products/50085261/50085261_866872.PNG?scale.height=505&amp;scale.width=773&amp;canvas.height=505&amp;canvas.width=773&amp;canvas.opacity=0&amp;q=90"></img>
            image = slickCarousel.find('img')
            if image is not None:
                try:
                    item_img = image['data-lazy']
                except:
                    item_img = None

                if not item_img:
                    try: 
                        item_img = image['src']
                    except:
                        item_img = None
                # split item_img on "?" character
                item_img = item_img.split("?")[0]



        # Key Features
        key_features_dict = {}
        key_features_text = ""

        # section id="keyftr"
        keyftr = soup.find('section', {'id': 'keyftr'})
        if keyftr is not None:
            brand_logo_keyftrs = keyftr.find('img', {'class': 'brand_logo_keyftrs'})['src']
            #key_features_dict.update({"brand_logo": brand_logo_keyftrs})
            #key_features_text += f"Brand Logo: {brand_logo_keyftrs}\n"

            featurelist = keyftr.find('ul', {'class': 'featurelist'})
            if featurelist is not None:
                # get the unordered list items
                featurelist = featurelist.find_all('li')
                if featurelist is not None:
                    # get the text of each list item
                    for item in featurelist:
                        # get the h5 item
                        h5 = item.find('h5').text.strip()
                        # get the small item
                        small = item.find('small').text.strip()
                        key_features_dict.update({small: h5})
                        key_features_text += f"{small}: {h5}\n"



        # Technical Specs
        tech_specs_dict = {}
        tech_specs_text = ""

        # class="speci_area"
        speci_area = soup.find('table', {'class': 'speci_area'})
        if speci_area is not None:
            # get the table rows
            rows = speci_area.find_all('tr')
            if rows is not None:
                # get each column of the row
                for row in rows:
                    # get the label and value of each row
                    th = row.find('th').text.strip()
                    # strip trailing ':' if it exists
                    th = th.rstrip(":")
                    td = row.find('td').text.strip()
                    tech_specs_dict.update({th: td})
                    tech_specs_text += f"{th}: {td}\n"


        # Product Features
        try:
            product_features = driver.find_element(By.ID, "AllInOne")

            if product_features:
                product_features = product_features.text.strip()

                # remove line with 'previous'
                product_features = product_features.replace("Previous", "")
                product_features = product_features.replace("Next", "")

                # remove line containing regregular expression ''
                product_features = re.sub("\d\d*\/.*", " ", product_features)

        except:
            product_features = ""



    except Exception as e:
        logging.error(f"Could not get the description from the URL: {job_url}")
        logging.error(e)
        exit()

    item['title'] = item_title
    item['brand'] = brand
    item['description'] = item_description
    item['price'] = item_price
    item['img'] = item_img
    item['key_features'] = key_features_text
    item['tech_specs'] = tech_specs_text
    item['product_features'] = product_features
    

    return item