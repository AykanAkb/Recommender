############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# Amacımız online retail II veri setine birliktelik analizi uygularak kullanıcılara ürün satın alma sürecinde
# ürün önermek

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

##############
# GÖREV 1 - ARL Veri Yapısını Hazırlama
##############

# Veri Ön İşleme

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("Bootcamp/3.Hafta/Ders Öncesi Notlar/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

##############
# GÖREV 2 - Invoice-Product Matrix hazırlama "Germany"
##############

df_GR = df[df['Country'] == "Germany"]

df_GR.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# descriptionları kolonlara taşıyan unstack komutu ile na lara o yazdırdık ve gözlemleme adına 5 satır ve sütunu çağırdık
df_GR.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
# apply sadece kolon ve ya satır gezerdi apply map tüm hücreleri gezip 0 ve 1 dağıtımı yapıcak

df_GR.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# Eğer bu prosesi formulize edicek olsaydık, stok kod ve descriprion a göre ayrımlı bu şekilde yapabilirdik
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


GR_inv_pro_df = create_invoice_product_df(df_GR)
GR_inv_pro_df.head()

GR_inv_pro_df = create_invoice_product_df(df_GR, id=True)

##############
# GÖREV 3 - ID'leri verilen ürünlerin isimleri Sorgulama
##############

def check_id(dataframe, stock_code_list):
    if type(stock_code_list) == int:
        stock_code_list = [stock_code_list]
    liste=[]
    for i in stock_code_list:
        product_name = dataframe[dataframe["StockCode"] == i][["Description"]].values[0].tolist()
        liste.append(product_name)

    print(liste)

ID_query=[21987,23235,22747]
check_id(df_GR, ID_query)
type(ID_query)

##############
# GÖREV 4 - Sepetteki kullanıcılar için ürün önerisi
##############

# Birliktelik Kurallarının Çıkarılması

frequent_itemsets = apriori(GR_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

# check_id(df_GR,23204)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

# Örnek çıktı

#      antecedents consequents  antecedent support  consequent support  support  confidence    lift  leverage  conviction
# 2650      (POST)     (22326)             0.81838             0.24508  0.22538     0.27540 1.12373   0.02482     1.04185
# 2651     (22326)      (POST)             0.24508             0.81838  0.22538     0.91964 1.12373   0.02482     2.26015
# 2784     (22328)      (POST)             0.15755             0.81838  0.15098     0.95833 1.17101   0.02205     4.35886
# 2785      (POST)     (22328)             0.81838             0.15755  0.15098     0.18449 1.17101   0.02205     1.03304
# 2414     (22328)     (22326)             0.15755             0.24508  0.13129     0.83333 3.40030   0.09268     4.52954

rules.sort_values("lift", ascending=False).head(50)


# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

# DENEME YAPTIM ÇALIŞMADI SORMAK GEREK

 # def arl_recommender(rules_df, product_id, rec_count=1):
 #    if type(product_id) == int:
 #           product_id = [product_id]
#
 #       sorted_rules = rules_df.sort_values("lift", ascending=False)
 #
#        recommendation_list = []
 #
 #       for i, product in sorted_rules["antecedents"].items():
 ##           for j in list(product):
 #               for k in product_id:
  #                 if j == product_id[k]:
   #                    recommendation_list.append("for" + check_id(dfk,k) + ", you can watch" + list(sorted_rules.iloc[i]["consequents"]))
 #
 #       recommendation_list = list({item for item_list in recommendation_list for item in item_list})
 #       return recommendation_list[:rec_count]


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


ID_query=[21987,23235,22747]

#check_id(df, 23049)
#arl_recommender(rules, 21987, 1)
#arl_recommender(rules, 23235, 2)
#arl_recommender(rules, 22747, 3)

arl_recommender(rules, 21987)
arl_recommender(rules, 23235)
arl_recommender(rules, 22747)

##############
# GÖREV 5 - Önerilen ürünlerin isimleri
##############

#check_id(df, recommendation_list[0])

check_id(df, arl_recommender(rules, 21987))
# bu ürünün adı ise  [['SET/10 BLUE POLKADOT PARTY CANDLES']]
check_id(df, arl_recommender(rules, 23235))
# bu ürünün adı ise  [['RED RETROSPOT MINI CASES']]
check_id(df, arl_recommender(rules, 22747))
# bu ürünün adı ise [['SCANDINAVIAN REDS RIBBONS']]


