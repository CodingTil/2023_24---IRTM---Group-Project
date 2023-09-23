import pyterrier as pt


def create_pipeline(index) -> pt.Transformer:
    sdm = pt.rewrite.SequentialDependence()
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    pl2 = pt.BatchRetrieve(index, wmodel="PL2")
    pipeline = (sdm >> (bm25 % 100).compile()) >> bo1 >> pl2
    return pipeline
