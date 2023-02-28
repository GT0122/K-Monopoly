import pymysql
import math
import json
import requests
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import io
import cv2

host = 'DBHost'
user = 'DBuser'
password = 'DBpassword'

Naver_Id = "Naver_Id"
Naver_Secret = "Naver_Secret"
KakaoAPI = "KakaoAPI"

LIMIT_PX = 1024
LIMIT_BYTE = 1024 * 1024  # 1MB
LIMIT_BOX = 40

def locationToAddress(locationName):
    url = "https://openapi.naver.com/v1/search/local.json?query=%s&display=1&start=1&sort=random"
    header = {"X-Naver-Client-Id": Naver_Id, "X-Naver-Client-Secret": Naver_Secret, "encoding": "UTF-8"}
    req = requests.get(url % (locationName), headers=header)
    address = json.loads(req.text)['items'][0]['address']
    return address

def addressToPoint(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    header = {"Authorization": "KakaoAK {}".format(KakaoAPI)}
    data = {"query": address}
    req = requests.get(url, headers=header, data=data)
    point = json.loads(req.text)['documents'][0]
    return point['x'], point['y']

def recommend_estate(price, spc, floor, lat, lon, tags):
    conn = pymysql.connect(host=host, user=user, password=password, db='project', charset='utf8')
    cur = conn.cursor()

    station = get_station(cur, lat, lon)

    df_danzi = get_danzi(cur, station, lat, lon)
    df_danzi_dup = df_danzi['apartNum'].drop_duplicates()

    df_estate = get_estate(cur, price, spc, df_danzi_dup)

    if len(df_estate) == 0:
        return pd.DataFrame([['일련번호', '조건에 맞는 아파트가 없습니다.', '가격', '면적', '인프라', '층']],
                            columns=['apartNum', 'apartName', 'price', 'spc', 'tag', 'floor'])

    floors = ['저', '중', '고']
    df_estate['score_sum'] = df_estate['apartNum'].apply(
        lambda x: df_danzi[df_danzi['apartNum'] == x]['score_sum'].values[0])
    df_estate['floor'] = df_estate.apply(lambda x: x['curfloor'] if x['curfloor'] in ['저', '중', '고'] else floors[
        int((int(x['curfloor']) - 1) / x['maxfloor'] * 3)], axis=1)
    rank = df_estate['spc'].rank()
    df_estate['rank'] = np.floor(rank / rank.max() * 5)

    df_estate = df_estate[df_estate['floor'] == floor]
    df_estate['score_total'] = df_estate['score_sum'] + df_estate['rank']
    df_estate = df_estate.sort_values(['score_total', 'price'], ascending=False)

    df_estate = get_tag(cur, df_estate, tags)

    conn.close()

    return df_estate[:30]


def recommend_estate_image(price, ages, lat, lon):
    conn = pymysql.connect(host=host, user=user, password=password, db='project', charset='utf8')
    cur = conn.cursor()

    spc = len(ages) * 6
    if price <= len(ages) * 20000 :
        price = len(ages) * 20000

    station = get_station(cur, lat, lon)

    df_danzi = get_danzi(cur, station, lat, lon)
    df_danzi_dup = df_danzi['apartNum'].drop_duplicates()

    df_estate = get_estate(cur, price, spc, df_danzi_dup)

    if len(df_estate) == 0:
        return pd.DataFrame([['일련번호', '조건에 맞는 아파트가 없습니다.', '가격', '면적', '인프라', '층', '주소']],
                            columns=['apartNum', 'apartName', 'price', 'spc', 'tag', 'floor', 'address'])

    floors = ['저', '중', '고']
    df_estate['score_sum'] = df_estate['apartNum'].apply(
        lambda x: df_danzi[df_danzi['apartNum'] == x]['score_sum'].values[0])
    df_estate['floor'] = df_estate.apply(lambda x: x['curfloor'] if x['curfloor'] in ['저', '중', '고'] else floors[
        int((int(x['curfloor']) - 1) / x['maxfloor'] * 3)], axis=1)
    rank = df_estate['spc'].rank()
    df_estate['rank'] = np.floor(rank / rank.max() * 5)

    if (0 in ages) or (1 in ages):
        floor = '저'
        df_estate = df_estate[df_estate['floor'] == floor]
        df_estate['rank'] = df_estate.apply(lambda x: x['rank'] + 2 if x['curfloor'] == 1 else x['rank'], axis=1)

    df_estate['score_total'] = df_estate['score_sum'] + df_estate['rank']

    tags = ['지하철', '공원', '대형마트']
    df_estate = get_tag_image(cur, df_estate, ages, tags)
    df_estate = df_estate.sort_values(['score_total', 'price'], ascending=False)

    conn.close()

    return df_estate[:30]


def get_station(cur, lat, lon):
    sql = "SELECT locationName, lat, lon FROM location WHERE maintype='지하철';"
    cur.execute(sql)
    data = []
    for i in range(cur.rowcount):
        data.append(cur.fetchone())

    df = pd.DataFrame(data, columns=['locationName', 'lat', 'lon'])
    df['distance'] = df.apply(lambda x: cal_distance(lat, lon, x['lat'], x['lon']), axis=1)
    df = df[df['distance'] == min(df['distance'])]

    return df.iloc[0, 0]


def get_danzi(cur, station, lat, lon):
    lat_up, lat_down = lat + 0.0045, lat - 0.0045
    lon_up, lon_down = lon + 0.00565, lon - 0.00565

    sql = "SELECT apartNum, score_change, score_distance, score_change+score_distance \
    FROM subway, danzi_location \
    WHERE start_sub=locationName AND distance<=1000 AND end_sub='%s' AND time_sub<=30 AND type_sub=1 AND maintype='지하철';"
    cur.execute(sql % (station))
    data = []
    for i in range(cur.rowcount):
        data.append(cur.fetchone())

    sql = "SELECT apartNum FROM danzi WHERE (lat<=%s AND lat>=%s) AND (lon<=%s AND lon>=%s);"
    cur.execute(sql % (lat_up, lat_down, lon_up, lon_down))
    for i in range(cur.rowcount):
        data.append([cur.fetchone()[0], 5, 5, 10])

    df_danzi = pd.DataFrame(data, columns=['apartNum', 'score_sub', 'score_dist', 'score_sum'])
    return df_danzi.sort_values('score_sum', ascending=False)


def get_estate(cur, price, spc, df_danzi_dup):
    sql = "SELECT estate.apartName, price, spc, curfloor, maxfloor, estate.lat, estate.lon, estate.apartNum, address \
      FROM estate, danzi \
      WHERE (price<=%s AND price>=%s*0.7) AND spc>=%s AND estate.apartNum=danzi.apartNum AND estate.apartNum in (" % (price, price, spc * 3.3)
    data = []
    last = df_danzi_dup.index[-1]
    for index, item in df_danzi_dup.items():
        sql += str(item)
        if index == last:
            sql += ");"
        else:
            sql += ","

    cur.execute(sql)
    for i in range(cur.rowcount):
        data.append(cur.fetchone())
    return pd.DataFrame(data, columns=['apartName', 'price', 'spc', 'curfloor', 'maxfloor', 'lat', 'lon', 'apartNum', 'address'])


def get_tag(cur, df_estate, tags):
    tag = str(tags).replace(']', '').replace('[', '')
    df_dup = df_estate['apartNum'].drop_duplicates()
    sql = "SELECT danzi_location.apartNum, locationName, danzi_location.maintype \
          FROM (SELECT MIN(distance) AS md, apartNum, maintype FROM danzi_location WHERE maintype in (%s) OR subtype in (%s) GROUP BY apartNum, maintype) AS a, danzi_location \
          WHERE distance=md AND danzi_location.maintype=a.maintype AND danzi_location.apartNum=a.apartNum AND danzi_location.apartNum in ("
    last = df_dup.index[-1]
    for index, item in df_dup.items():
        sql += str(item)
        if index == last:
            sql += ") ORDER BY apartNum;"
        else:
            sql += ", "
    cur.execute(sql % (tag, tag))
    data = []
    for i in range(cur.rowcount):
        data.append(cur.fetchone())

    sql = "SELECT danzi_location.apartNum, locationName, danzi_location.subtype \
          FROM (SELECT MIN(distance) AS md, apartNum, subtype FROM danzi_location WHERE maintype in (%s) OR subtype in (%s) GROUP BY apartNum, maintype) AS a, danzi_location \
          WHERE distance=md AND danzi_location.subtype=a.subtype AND danzi_location.apartNum=a.apartNum AND danzi_location.apartNum in ("
    last = df_dup.index[-1]
    for index, item in df_dup.items():
        sql += str(item)
        if index == last:
            sql += ") ORDER BY apartNum;"
        else:
            sql += ", "
    cur.execute(sql % (tag, tag))
    for i in range(cur.rowcount):
        data.append(cur.fetchone())

    df_lo = pd.DataFrame(data, columns=['apartNum', 'locationName', 'type'])
    df_estate['tag'] = df_estate['apartNum'].apply(
        lambda x: df_lo[df_lo['apartNum'] == x]['locationName'].drop_duplicates().values)

    return df_estate


def get_tag_image(cur, df_estate, ages, tags):
    s_tag = []
    if 0 in ages:
        s_tag.append('유치원')
    if 1 in ages:
        s_tag.append('초등학교')
    if 2 in ages:
        tags.append('중학교')
        s_tag.append('고등학교')
    if 4 in ages:
        s_tag.append('병원')

    tags.extend(s_tag)

    tag = str(tags).replace(']', '').replace('[', '')
    df_dup = df_estate['apartNum'].drop_duplicates()
    sql = "SELECT danzi_location.apartNum, locationName, danzi_location.maintype, score_distance \
          FROM (SELECT MIN(distance) AS md, apartNum, maintype FROM danzi_location WHERE maintype in (%s) OR subtype in (%s) GROUP BY apartNum, maintype) AS a, danzi_location \
          WHERE distance=md AND danzi_location.maintype=a.maintype AND danzi_location.apartNum=a.apartNum AND danzi_location.apartNum in ("
    last = df_dup.index[-1]
    for index, item in df_dup.items():
        sql += str(item)
        if index == last:
            sql += ") ORDER BY apartNum;"
        else:
            sql += ", "
    cur.execute(sql % (tag, tag))
    data = []
    for i in range(cur.rowcount):
        data.append(cur.fetchone())

    df_lo = pd.DataFrame(data, columns=['apartNum', 'locationName', 'maintype', 'score_location'])
    df_estate['tag'] = df_estate['apartNum'].apply(
        lambda x: df_lo[df_lo['apartNum'] == x]['locationName'].drop_duplicates().values)
    df_group = df_lo[df_lo['maintype'].isin(s_tag)].groupby('apartNum')['score_location'].sum()
    df_estate['score_total'] = df_estate.apply(
        lambda x: df_group[df_group.index == x['apartNum']].values[0] + x['score_total'], axis=1)

    return df_estate

# 거리 측정
def cal_distance(lat1, lon1, lat2, lon2):
    R = 6371e3
    φ1 = lat1 * math.pi / 180
    φ2 = lat2 * math.pi / 180
    Δφ = (lat2 - lat1) * math.pi / 180
    Δλ = (lon2 - lon1) * math.pi / 180

    a = math.sin(Δφ / 2) * math.sin(Δφ / 2) + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) * math.sin(Δλ / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return round(distance)

def get_age(image):
    nparr = np.fromstring(image, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_re = Image.open(io.BytesIO(image))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    xml = 'haarcascades/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(xml)
    faces = face_cascade.detectMultiScale(gray, 1.3)

    print("Number of faces detected: " + str(len(faces)))
    data = []
    images = []
    if len(faces):
        for (x, y, w, h) in faces:
            data.append([x, y, x + w, y + h])
            image_crop = image_re.crop((x, y, x + w, y + h))
            image_crop = image_crop.convert("RGB")
            image_crop = image_crop.resize((100, 100))
            image_crop = np.asarray(image_crop).astype(np.float32)
            images.append(image_crop)
        images = tf.stack(images)
        images = images / 255.

        ages = []
        model = load_model('model_100')
        for item in model.predict(images):
            ages.append(np.argmax(item))

        print(ages)

        return ages
    return []


def get_address(image):
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    if LIMIT_PX < height or LIMIT_PX < width:
        ratio = float(LIMIT_PX) / max(height, width)
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        height, width, _ = height, width, _ = image.shape

    height, width, _ = image.shape

    API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'

    headers = {'Authorization': 'KakaoAK {}'.format(KakaoAPI)}

    jpeg_image = cv2.imencode(".jpg", image)[1]
    data = jpeg_image.tobytes()

    output = requests.post(API_URL, headers=headers, files={"image": data}).json()
    key = json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False)

    gu = ['강남', '강동', '강북', '강서', '관악', '광진', '구로', '금천', '노원', '도봉', '동대', '동작', '마포', '서대', '서초', '성동', '성북', '송파', '양천', '영등', '용산', '은평', '종로', '중구', '중랑']

    index = 0
    for item in eval(key)['result']:
        # print(item['recognition_words'])
        if '서울' in item['recognition_words'][0]:
            num_index = index
            address = item['recognition_words'][0]
            try:
                int(address[-1])
                address = address
            except:
                for i in range(index + 1, len(eval(key)['result'])):
                    try:
                        for j in eval(key)['result'][index + 1:num_index + 2]:
                            address += j['recognition_words'][0]
                        break
                    except:
                        num_index += 1
            address = address.replace('시', '시 ').replace('구', '구 ').replace('로', '로 ').replace('  ', ' ')
            return address
        elif item['recognition_words'][0][:2] in gu :
            num_index = index
            address = item['recognition_words'][0]
            try:
                int(address[-1])
                address = address
            except:
                for i in range(index + 1, len(eval(key)['result'])):
                    try:
                        for j in eval(key)['result'][index + 1:num_index + 2]:
                            address += j['recognition_words'][0]
                        break
                    except:
                        num_index += 1
            address = address.replace('구', '구 ').replace('로', '로 ').replace('  ', ' ')
            return address
        index += 1

    return ''


def get_money(image):
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    if LIMIT_PX < height or LIMIT_PX < width:
        ratio = float(LIMIT_PX) / max(height, width)
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        height, width, _ = height, width, _ = image.shape

    height, width, _ = image.shape

    API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'

    headers = {'Authorization': 'KakaoAK {}'.format(KakaoAPI)}

    jpeg_image = cv2.imencode(".jpg", image)[1]
    data = jpeg_image.tobytes()

    output = requests.post(API_URL, headers=headers, files={"image": data}).json()
    key = json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False)

    money_list = []
    for item in eval(key)['result']:
        if ',' in item['recognition_words'][0]:
            try:
                money = int(item['recognition_words'][0].replace(',', ''))
                money_list.append(money)
            except:
                pass
    try :
       price = max(money_list) / 0.0343 * 12 * 15 // 10000
    except :
        price = 0

    return price