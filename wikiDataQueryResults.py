import sys
import pandas as pd
from typing import List, Dict, Any, cast
from SPARQLWrapper import SPARQLWrapper, JSON

class WikiDataQueryResults:

    def __init__(self, query: str, label: str):
        """
        Initializes the WikiDataQueryResults object with a SPARQL query string.
        """
        self.user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint_url, agent=self.user_agent)
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        self.label = label

    def __transform2dicts(self, results: List[Dict]) -> List[Dict]:
        """
        Helper function to transform SPARQL query results into a list of dictionaries.
        -> param results: A list of query results returned by SPARQLWrapper.
        -> return: A list of dictionaries, where each dictionary represents a result row and has keys corresponding to the
        variables in the SPARQL SELECT clause.
        """
        new_results = []
        for result in results:
            new_result = {}
            for key in result:
                new_result[key] = result[key]['value']
            new_results.append(new_result)
        return new_results

    def _load(self) -> List[Dict]:
        """
        Helper function that loads the data from Wikidata using the SPARQLWrapper library, and transforms the results into
        a list of dictionaries.
        -> Return: A list of dictionaries, where each dictionary represents a result row and has keys corresponding to the
        variables in the SPARQL SELECT clause.
        """
        raw = self.sparql.queryAndConvert()
        # safely extract bindings from the response and ensure the type for the transformer
        bindings: Any = []
        if isinstance(raw, dict):
            bindings = raw.get('results', {}).get('bindings', [])
        results = cast(List[Dict], bindings)
        results = self.__transform2dicts(results)
        return results

    def load_as_dataframe(self) -> pd.DataFrame:
        """
        Executes the SPARQL query and returns the results as a Pandas DataFrame.

        """
        results = self._load()
        #save in data/
        df = pd.DataFrame.from_records(results)
        # add a column with the label
        df['label'] = self.label
        df.to_csv(f'wikidata/query_results_{self.label}.csv', index=False)
        
        return df

