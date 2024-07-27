import json

data_json = """{
    "url": [
        "https://api.upstox.com/v2/historical-candle/NSE_EQ|INE154A01025/30minute/2024-07-25"
    ]
}"""

data_list = [
    {
        "url": "https://api.upstox.com/v2/historical-candle/intraday/NSE_EQ|INE040A01034/30minute"
    },
    {
        "url": "https://api.upstox.com/v2/historical-candle/intraday/NSE_EQ|INE040A01034/30minute"
    }
]

def process_data(data):
    # Check if data is a JSON string
    if isinstance(data, str):
        data = json.loads(data)

    # Check if data is a list or a dictionary
    if isinstance(data, list):
        json_object = data[0]
        if "url" in json_object:
            print("yes for list")
            print(json_object["url"])
    else:
        if "url" in data:
            print("yes for json")
            print(data["url"])

# Process both data_json and data_list
# process_data(data_json)
process_data(data_list)


    # try:
    #     json_response_link = json.loads(response["answer"])
    #     print("Entered in first try block itself...............")
    #     json_string_keys = json_response_link.keys()
    #     for key in json_string_keys:
    #         if key == "url":
    #             links = json_response_link["url"]
    #             print(links)
    #             print("--" * 25)
    #         else:
    #             links = json_response_link["urls"]
    #             print(links)
    #             print("--" * 25)
    # except:
    #     if isinstance(response["answer"], list):
    #         print("object is in list of dict............")
    #         json_object = json.loads(json.dumps(response["answer"]))
    #         if "url" in json_object:
    #             links = json_object["url"]
    #             print(links)
    #             print("--" * 25)
    #         else:
    #             links = json_object["urls"]
    #             print(links)
    #             print("--" * 25)
    #     else:
    #         json_object = json.loads(json.dumps(response["answer"]))
    #         if "url" in json_object:
    #             links = json_object["url"]
    #             print(links)
    #             print("--" * 25)
    #         else:
    #             links = json_object["urls"]
    #             print(links)
    #             print("--" * 25)