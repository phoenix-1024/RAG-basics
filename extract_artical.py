from newspaper import Article


def get_text_from_link(link):
    a = Article(link)
    a.download()
    a.parse()
    return a.text
