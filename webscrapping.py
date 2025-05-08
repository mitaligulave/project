import pandas as pd
import requests
from bs4 import BeautifulSoup
Product_name=[]
Prices=[]
Description=[]
Reviews=[]


for i in range(2 ,92):
    url="https://www.flipkart.com/search?q=laptop&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page="+str(i)
    r=requests.get(url)


    soup=BeautifulSoup(r.text,"lxml")

    box=soup.find("div",class_="DOjaWF gdgoEp")

    names = box.find_all("div", class_="KzDlHZ")
    for i in names:
        name=i.text
        Product_name.append(name)


    prices=box.find_all("div", class_="Nx9bqj _4b5DiR")
    for i in prices:
        name=i.text
        Prices.append(name)


    desc=box.find_all("div", class_="_6NESgJ")
    for i in desc:
        name = i.text
        Description.append(name)


    reviews=box.find_all("div", class_="XQDdHH")
    for i in reviews:
        name = i.text
        Reviews.append(name)


min_len = min(len(Product_name), len(Prices), len(Description), len(Reviews))
Product_name = Product_name[:min_len]
Prices = Prices[:min_len]
Description = Description[:min_len]
Reviews = Reviews[:min_len]

df = pd.DataFrame({"Product_name": Product_name, "Prices": Prices, "Description": Description, "Reviews": Reviews})
df.to_csv("flipkart_laptop.csv")