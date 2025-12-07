from pretraining.data_training.wikiDataQueryResults import WikiDataQueryResults
from common import labels


for label, qid in labels.items():
    query = f"""
    SELECT ?humanLabel ?genderLabel
    WHERE {{
        ?human wdt:P31 wd:Q5 .
        ?human wdt:P106/wdt:P279* wd:{qid} .
        ?human wdt:P21 ?gender .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }}
    }}
    LIMIT 10000
    """
    data_extracter = WikiDataQueryResults(query, label)
    df = data_extracter.load_as_dataframe()
    print(f"Data for {label} saved. Number of records: {len(df)}")


# query = """
# SELECT ?human ?humanLabel ?genderLabel
# WHERE {
#   ?human wdt:P31 wd:Q5 .
#   ?human wdt:P106/wdt:P279* wd:Q186360 .
#   ?human wdt:P21 ?gender .
  
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }
# }
# LIMIT 10000
# """

# data_extracter = WikiDataQueryResults(query, 'nurse')
# df = data_extracter.load_as_dataframe()
# print(df.head())

# (Person, gender is, gender label)
# (Person, occupation is, occupation label)