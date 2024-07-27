from googlesearch import search

def search_urls(query):
    urls=[]
    for url in search(query, tld="com",safe=True,tbs='qdr:d', num=10, stop=10, pause=1):
        urls.append(url)
    return urls
