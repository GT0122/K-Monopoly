import pandas as pd
from fastapi import FastAPI, Form, Request, File
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import urllib.request
from bs4 import BeautifulSoup

import model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def get_news():
    rssUrl = "http://www.renews.co.kr/rss/clickTop.xml"
    html = urllib.request.urlopen(rssUrl).read()
    soup = BeautifulSoup(html, 'html.parser')
    html = []
    for item in soup.find_all('item')[:2]:
        title = item.find('title').text
        description = item.find('description').get_text()
        body = BeautifulSoup(description, 'html.parser')
        body_text = body.find('p').text
        try:
            image = "http://www.renews.co.kr" + body.find('img').attrs['src']
        except:
            image = ''
        index = item.text.find('http://')
        length = item.text.find('\n', index)
        link = item.text[index:length]
        html.append([title, body_text, link, image])
    return pd.DataFrame(html, columns=['title', 'body', 'link', 'image'])

df_html = get_news()

@app.get("/")
async def home(req: Request):
    info = {"price": 50000, "spc": 15, "worker": "서울시청", "floor": "고"}
    users = pd.DataFrame([['일련번호', '아파트명', '가격', '면적', '인프라', '층', '주소']],
                         columns=['apartNum', 'apartName', 'price', 'spc', 'tag', 'floor', 'address'])
    location = {"lat": 37.567891982327815, "lon": 126.98250140057401}
    infra = ['지하철']
    return templates.TemplateResponse("index.html", {"request": req, "info": info, "users": users, "location": location,
                                                     "infra": infra, "html": df_html})

@app.get("/AiService")
async def AiService(req: Request):
    users = pd.DataFrame([['일련번호', '아파트명', '가격', '면적', '인프라', '층', '주소']],
                         columns=['apartNum', 'apartName', 'price', 'spc', 'tag', 'floor', 'address'])
    location = {"lat": 37.567891982327815, "lon": 126.98250140057401}
    return templates.TemplateResponse("index2.html",
                                      {"request": req, "users": users, "location": location, "html": df_html})

@app.post("/search_estate")
async def search_estate(req: Request, price: int = Form(...), spc: int = Form(...), worker: str = Form(...),
                        floor: str = Form(...), infra: list = Form(...)):
    info = {"price": price, "spc": spc, "worker": worker, "floor": floor}
    address = model.locationToAddress(worker)
    lon, lat = model.addressToPoint(address)
    infra.insert(0, '지하철')
    users = model.recommend_estate(price, spc, floor, float(lat), float(lon), infra)
    location = {"lat": lat, "lon": lon}
    return templates.TemplateResponse("index.html", {"request": req, "info": info, "users": users, "location": location,
                                                     "infra": infra, "html": df_html})

@app.post("/image")
async def image(req: Request, family: bytes = File(...), card: bytes = File(...), insurance: bytes = File(...)):
    family = model.get_age(family)
    if len(family) == 0:
        location = {"lat": 37.567891982327815, "lon": 126.98250140057401}
        users = pd.DataFrame([['일련번호', '얼굴 인식이 되지 않았습니다.', '가격', '면적', '인프라', '층', '주소']],
                         columns=['apartNum', 'apartName', 'price', 'spc', 'tag', 'floor', 'address'])
        return templates.TemplateResponse("index2.html",
                                          {"request": req, "users": users, "location": location, "html": df_html})
    address = model.get_address(card)
    if address == '' :
        location = {"lat": 37.567891982327815, "lon": 126.98250140057401}
        users = pd.DataFrame([['일련번호', '명함이 인식되지 않습니다.', '가격', '면적', '인프라', '층', '주소']],
                             columns=['apartNum', 'apartName', 'price', 'spc', 'tag', 'floor', 'address'])
        return templates.TemplateResponse("index2.html",
                                          {"request": req, "users": users, "location": location, "html": df_html})
    money = model.get_money(insurance)
    lon, lat = model.addressToPoint(address)
    location = {"lat": lat, "lon": lon}
    users = model.recommend_estate_image(money, family, float(lat), float(lon))
    return templates.TemplateResponse("index2.html",
                                      {"request": req, "users": users, "location": location, "html": df_html})