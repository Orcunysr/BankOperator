import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

stopwords = ['fakat', 'lakin', 'ancak', 'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 
             'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 
             'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 
             'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 
             'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 
             'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']

# Veri setini yükle
df = pd.read_csv('../DataSet/bankv3.csv')
print(df.head(3))

# Sadece 'sorgu' ve 'label' sütunlarını seç
df = df[["sorgu", "label"]]
print(df.head(3))

# Sorgu metinlerini düzenle
df["sorgu"] = df["sorgu"].str.replace("[^a-zA-Z]", " ", regex=True)
df["sorgu"] = df["sorgu"].apply(lambda x: x.lower())

for stopword in stopwords:
    pattern = " " + stopword + " "
    df["sorgu"] = df["sorgu"].str.replace(pattern, "")

# Metin verilerini vektörleştir
cv = CountVectorizer(max_features=50)
x = cv.fit_transform(df["sorgu"]).toarray()
y = df["label"]

# Eğitim ve test verilerini ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=21, test_size=0.2)

# RandomForest modeli oluştur ve eğit
rf = RandomForestClassifier(n_estimators=100)
model = rf.fit(x_train, y_train)
print("Model train accuracy: " + str(model.score(x_test, y_test)))

# Tahmin fonksiyonu
def predict(mesaj):
    mesajdf = pd.DataFrame({"sorgu": [mesaj]})
    for stopword in stopwords:
        pattern = " " + stopword + " "
        mesajdf["sorgu"] = mesajdf["sorgu"].str.replace(pattern, "", regex=True)
    return model.predict(cv.transform(mesajdf["sorgu"]).toarray())

mesaj = input("Sorgunuzu giriniz: ")
prediction = predict(mesaj)
print(f"Girdiğiniz sorgu tahmini etiket: {prediction}")
