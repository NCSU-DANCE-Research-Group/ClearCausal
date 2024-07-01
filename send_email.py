from mailjet_rest import Client
from datetime import datetime
import sys
from secret import api_key, api_secret, machine_name

def send_email(start_time=None):
  mailjet = Client(auth=(api_key, api_secret), version='v3.1')
  text = f"Your experiment at machine {machine_name} "
  finish_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  if start_time is not None:
    text += f"started at {start_time} "
  text += f"is completed at {finish_time}."
  data = {
    'Messages': [
      {
        "From": {
          "Email": "php9850@gmail.com",
          "Name": "ExperimentNotifier"
        },
        "To": [
          {
            "Email": "ylin34@ncsu.edu",
            "Name": "You"
          }
        ],
        "Subject": f"{text}",
        "TextPart": f"{text}",
        "HTMLPart": f"{text}"
      }
    ]
  }
  result = mailjet.send.create(data=data)
  if result.status_code != 200:
    print("Error sending the email")
    print(result.json())

if __name__ == "__main__":
  start_time = None
  if len(sys.argv) > 1:
    start_time = " ".join(sys.argv[1:])
  send_email(start_time)