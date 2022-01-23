import pandas as pd

from intermine.webservice import Service
from typing import Optional, Union


def fm_BDGP(
        gene: Union[str, list] = 'all',
        gene_identifier: str = 'symbol'
) -> pd.DataFrame:
    """
    Three functions:
        * Search for mRNA expression results in BDGP database for a particular gene from Drosophila melanogaster.
        * Search for mRNA expression results in BDGP database for a particular gene list from Drosophila melanogaster.
        * Show the mRNA expression results in BDGP database for all genes from Drosophila melanogaster.

    Parameters
    ----------
    gene: `str` or `list` (default: `all`)
        Gene name. Available gene types are:
            * `'all': show all genes.
            * `'Adh'`: show a particular gene.
            * `['Adh', 'LpR2']`: show ta particular gene list.
    gene_identifier: `str` (default: `'symbol'`)
        Gene identifier type. Available gene_identifiers are:
            * `'symbol': gene symbol, e.g.: 'Adh'.
            * `'primaryIdentifier'`: gene primary identifier, e.g.: `FBgn0000055`.
            * `'secondaryIdentifier'`: gene secondary identifier, e.g.: `CG3481`.

    Returns
    -------
    data: `pd.DataFrame`

    """

    service = Service("https://www.flymine.org/flymine/service")

    query = service.new_query("Gene")

    query.add_view(
        "primaryIdentifier",
        "secondaryIdentifier",
        "symbol",
        "organism.name",
        "mRNAExpressionResults.expressed",
        "mRNAExpressionResults.stageRange",
        "mRNAExpressionResults.mRNAExpressionTerms.name"
    )

    if gene is "all":
        query.add_constraint("organism.name", "=", "Drosophila melanogaster", code="A")
    else:
        gene_list = gene if isinstance(gene, list) else [gene]
        query.add_constraint(gene_identifier, "ONE OF", gene_list, code="A")

    data = pd.DataFrame(
        [
            [
                row["primaryIdentifier"],
                row["secondaryIdentifier"],
                row["symbol"],
                row["mRNAExpressionResults.expressed"],
                row["mRNAExpressionResults.stageRange"],
                row["mRNAExpressionResults.mRNAExpressionTerms.name"]
            ]
            for row in query.rows()
        ],
        columns=["DB identifier", "CG identifier", "gene symbol", "expressed", 'StageRange', 'mRNAExpressionTerms']
    )

    return data


def fm_gene2GO(
        gene: Union[str, list] = 'all',
        gene_identifier: str = 'symbol',
        GO_namespace: Union[str, list] = 'all'
) -> pd.DataFrame:
    """
    Three functions:
        * Search for GO annotations for a particular gene from Drosophila melanogaster.
        * Search for GO annotations for a particular gene list from Drosophila melanogaster.
        * Show the GO terms for all genes from Drosophila melanogaster.

    Parameters
    ----------
    gene: `str` or `list` (default: `all`)
        Gene name. Available gene types are:
            * `'all': show the GO terms for all genes.
            * `'Adh'`: show the GO terms for a gene.
            * `['Adh', 'LpR2']`: show the GO terms for a gene list.
    gene_identifier: `str` (default: `'symbol'`)
        Gene identifier type. Available gene_identifiers are:
            * `'symbol': gene symbol, e.g.: 'Adh'.
            * `'primaryIdentifier'`: gene primary identifier, e.g.: `FBgn0000055`.
            * `'secondaryIdentifier'`: gene secondary identifier, e.g.: `CG3481`.
    GO_namespace: `str` or `list` (default: `all`)
        Ontology term namespace. Available GO_namespaces are:
            * `'all': show all Ontology term namespaces.
            * `'molecular_function'`: show molecular_function.
            * `'biological_process'`: show biological_process.
            * `'cellular_component'`: show cellular_component.
            * `['molecular_function', 'cellular_component']`: show molecular_function and cellular_component.

    Returns
    -------
    data: `pd.DataFrame`

    """

    service = Service("https://www.flymine.org/flymine/service")

    query = service.new_query("Gene")

    query.add_view(
        "primaryIdentifier",
        "symbol",
        "secondaryIdentifier",
        "goAnnotation.ontologyTerm.identifier",
        "goAnnotation.ontologyTerm.name",
        "goAnnotation.ontologyTerm.namespace",
        "goAnnotation.ontologyTerm.description"
    )

    query.add_sort_order("Gene.secondaryIdentifier", "ASC")

    if gene is "all":
        query.add_constraint("organism.name", "=", "Drosophila melanogaster", code="A")
    else:
        gene_list = gene if isinstance(gene, list) else [gene]
        query.add_constraint(gene_identifier, "ONE OF", gene_list, code="A")

    if GO_namespace is not "all":
        GO_namespace_list = GO_namespace if isinstance(GO_namespace, list) else [GO_namespace]
        query.add_constraint("goAnnotation.ontologyTerm.namespace", "ONE OF", GO_namespace_list, code="A")

    data = pd.DataFrame(
        [
            [
                row["primaryIdentifier"],
                row["symbol"],
                row["secondaryIdentifier"],
                row["goAnnotation.ontologyTerm.identifier"],
                row["goAnnotation.ontologyTerm.name"],
                row["goAnnotation.ontologyTerm.namespace"],
                row["goAnnotation.ontologyTerm.description"],
            ]
            for row in query.rows()
        ],
        columns=["DB identifier", "CG identifier", "gene symbol", "GO id", "GO name", "GO namespace", "GO description"]
    )

    return data
