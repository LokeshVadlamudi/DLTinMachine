from django.shortcuts import render
import requests
import datetime
from geopy.geocoders import Nominatim
import cv2
from fastai import *
from fastai.tabular import *
from torchvision.models import *
from fastai.vision import *


def default_view(request):
    return render(request, 'base.html')


TINDER_URL = "https://api.gotinder.com"
geolocator = Nominatim(user_agent="DLTinMachine")
PROF_FILE = "profiles.txt"
#PROF_FILE = "profiles.txt"


class tinderAPI():

    def __init__(self, token):
        self._token = token

    def matches(self, limit=10):
        data = requests.get(TINDER_URL + f"/v2/matches?count={limit}", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(TINDER_URL + f"/like/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        requests.get(TINDER_URL + f"/pass/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return True

    def nearby_persons(self):
        print('inside nearby')
        data = requests.get(TINDER_URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        print(data)
        return list(map(lambda user: Person(user["user"], self), data["data"]["results"]))


class Person(object):

    def __init__(self, data, api):
        self._api = api

        self.id = data["_id"]
        self.name = data.get("name", "Unknown")

        self.bio = data.get("bio", "")
        self.distance = data.get("distance_mi", 0) / 1.60934

        self.birth_date = datetime.datetime.strptime(data["birth_date"], '%Y-%m-%dT%H:%M:%S.%fZ') if data.get(
            "birth_date", False) else None
        self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]

        self.images = list(map(lambda photo: photo["url"], data.get("photos", [])))

        self.jobs = list(
            map(lambda job: {"title": job.get("title", {}).get("name"), "company": job.get("company", {}).get("name")},
                data.get("jobs", [])))
        self.schools = list(map(lambda school: school["name"], data.get("schools", [])))

        if data.get("pos", False):
            self.location = geolocator.reverse(f'{data["pos"]["lat"]}, {data["pos"]["lon"]}')

    def __repr__(self):
        return f"{self.id}  -  {self.name} ({self.birth_date.strftime('%d.%m.%Y')})"

    def like(self):
        return self._api.like(self.id)

    def dislike(self):
        return self._api.dislike(self.id)


token = "54e65bf8-c7b1-40a8-809f-de2fc727147e"
api = tinderAPI(token)
print(api, 'inside main')


def default_view2(request):
    persons = api.nearby_persons()
    peeps = []
    count = 0
    for person in persons:

        print("-------------------------")
        print("ID: ", person.id)
        print("Name: ", person.name)
        print("Schools: ", person.schools)
        image_url = person.images[0]
        print(os.getcwd())
        req = requests.get(image_url, stream=True)
        folder = "images"

        # create folder if not exists
        if not os.path.exists(folder):
            os.makedirs(folder)

        name = person.name
        # school = person.schools
        if req.status_code == 200:
            with open(f"{folder}/{name}.jpeg", "wb") as f:
                f.write(req.content)

        sz = 1500

        def imageToTensorImage(path):
            bgr_img = cv2.imread(path)
            b, g, r = cv2.split(bgr_img)
            rgb_img = cv2.merge([r, g, b])
            H, W, C = rgb_img.shape
            rgb_img = rgb_img[(H - sz) // 2:(sz + (H - sz) // 2), (H - sz) // 2:(sz + (H - sz) // 2), :] / 256
            return vision.Image(px=pil2tensor(rgb_img, np.float32))

        learn = load_learner('DLTinMachine/')
        img = imageToTensorImage('images/' + name + '.jpeg')

        # predict and visualize
        y = learn.predict(img)[0]
        print(y)
        count += 1
        status = ''
        if str(y) == 'whites':
            res = person.like()
            print("LIKE")
            print("Response: ", res)
            status = 'LIKE'
        else:
            res = person.dislike()
            print("DISLIKE")
            print("Response: ", res)
            status = 'DISLIKE'
        peeps.append({'name': name, 'url': image_url, 'status': status})
        if count == 3:
            break

    context = {
        'peeps': peeps,
    }
    return render(request, 'tinner.html', context)
