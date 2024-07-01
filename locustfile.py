from locust import HttpUser, task, between, HttpLocust, TaskSet
import random
import base64
import json

class WebTasks(HttpUser):

    host = "http://localhost:8080"

    wait_time = between(0.05, 5)

    

    @task(10)
    def index(self):
        self.client.get("/index.html")

    #TODO: change cookies each test
    @task(5)
    def post(self):
        cookies = {"login_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0aW1lc3RhbXAiOiIxNzA4MTA4MTY4IiwidHRsIjoiMzYwMCIsInVzZXJfaWQiOiIxMzkyNDA0MjE5MjcwMDU3OTg0IiwidXNlcm5hbWUiOiJhIn0.t_t-_g-8xGBPYlbWT5VKeThLbJUIpVY-TOCPOlM8l8Y"}
        self.client.post("/api/post/compose", "post_type=0&text=114514", cookies=cookies)

    @task(10)
    def view_contact(self):
        self.client.get("/contact.html")


    @task(2)
    def view_profile(self):
        self.client.get("/profile.html")

# class OnlineBoutiqueUser(HttpUser):
#     host = "http://localhost:8080"
#     products = [
#     '0PUK6V6EV0',
#     '1YMWWN1N4O',
#     '2ZYFJ3GM2N',
#     '66VCHSJNUP',
#     '6E92ZMYYFZ',
#     '9SIQT8TOJO',
#     'L9ECAV7KIM',
#     'LS4PSXUNUM',
#     'OLJCESPC7Z']

#     def wait_time(self):
#         return 0.5

#     @task(1)
#     def view_homepage(self):
#         self.client.get("/")

#     @task(1)
#     def set_currency(self):
#         currencies = ['EUR', 'USD', 'JPY', 'CAD']
#         self.client.post("/setCurrency",
#             {'currency_code': random.choice(currencies)})

#     @task(1)
#     def view_product(self):
#         self.client.get("/product/" + random.choice(self.products))

#     # @task
#     # def view_cart(self):
#     #     self.client.get("/cart")

#     @task(1)
#     def add_to_cart(self):
#         product = random.choice(self.products)
#         self.client.get("/product/" + product)
#         self.client.post("/cart", {
#             'product_id': product,
#             'quantity': random.choice([1,2,3,4,5,10])})
#         self.client.get("/cart")

#     @task(5)
#     def checkout(self):
#         self.add_to_cart()
#         self.client.post("/cart/checkout", {
#         'email': 'someone@example.com',
#         'street_address': '1600 Amphitheatre Parkway',
#         'zip_code': '94043',
#         'city': 'Mountain View',
#         'state': 'CA',
#         'country': 'United States',
#         'credit_card_number': '4432-8015-6152-0454',
#         'credit_card_expiration_month': '1',
#         'credit_card_expiration_year': '2039',
#         'credit_card_cvv': '672',
#     })
