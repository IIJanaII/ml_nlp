import pickle

def loadCNN():
	file = open("C:/Users/jega_/CNN/CNNArticles",'rb')
	articles = pickle.load(file)
	file = open("C:/Users/jega_/CNN/CNNGold",'rb')
	abstracts = pickle.load(file)

	articlesCl = []  
	for article in articles:
		articlesCl.append(article.replace("”", "").rstrip("\n"))
	articles = articlesCl
	  
	articlesCl = []  
	for article in abstracts:
		articlesCl.append(article.replace("”", "").rstrip("\n"))
	abstracts = articlesCl
    
	return articles, abstracts

articles, abstracts = loadCNN()

print("ARTICLE=",articles[0])
print("SUMMARY=",abstracts[0])
