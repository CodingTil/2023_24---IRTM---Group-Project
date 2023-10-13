import string
import logging
from typing import List, Any, Callable, Optional

import pyterrier as pt
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME: str = "castorini/t5-base-canard"
MAX_LENGTH: int = 512
NUM_BEAMS: int = 10
EARLY_STOPPING: bool = True

COPY_REWRITTEN_QUERY_COLUMN: str = "rewritten_query"
SEPERATOR_TOKEN: str = " ||| "


class T5Rewriter(pt.Transformer):
    """
    T5 Query Rewriter set up as a PyTerrier Transformer.

    Attributes
    ----------
    device : torch.device
        The device to use.
    tokenizer : T5Tokenizer
        The tokenizer to use.
    model : T5ForConditionalGeneration
        The model to use.
    """

    device: torch.device
    tokenizer: T5Tokenizer
    model: T5ForConditionalGeneration

    def __init__(self):
        """
        Constructs all the necessary attributes for the T5 Query Rewriter.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            .to(self.device)
            .eval()
        )
        super().__init__()

    # the query has multiply SEPERATOR_TOKEN in it. Create a list of the split with a maximum of 3 elements (last element is last, second last is middle, and the first n are joined)
    def __split_query_tokenize_join(self, q):
        """
        Split the query, tokenize the parts, and join them back together.
        """
        l = q.split(SEPERATOR_TOKEN)
        if len(l) < 3:
            tokens = []
            for ll in l:
                tokens.extend(self.tokenizer.tokenize(ll))
                tokens.append(SEPERATOR_TOKEN)
            if len(tokens) > 0:
                tokens.pop()
            return tokens
        else:
            tokens = []
            tokens.extend(self.tokenizer.tokenize(SEPERATOR_TOKEN.join(l[:-2])))
            tokens.append(SEPERATOR_TOKEN)
            tokens.extend(self.tokenizer.tokenize(l[-2]))
            tokens.append(SEPERATOR_TOKEN)
            tokens.extend(self.tokenizer.tokenize(l[-1]))
            return tokens

    def __get_input_token_ids(self, tokens):
        """
        Get the input token ids.
        """
        return self.tokenizer.encode(
            tokens, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

    def __get_output_token_ids(self, input_token_ids):
        """
        Get the output token ids.
        """
        return self.model.generate(
            input_token_ids,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
        )

    def __decode_token_ids(self, token_ids):
        """
        Decode the token ids.
        """
        return self.tokenizer.decode(
            token_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def __remove_punctuation(self, s):
        """
        Remove punctuation from a string.
        """
        return s.translate(str.maketrans("", "", string.punctuation))

    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        # save qid and query columns as dict (qid -> query) query is same for same qid, so sufficient to select first
        rewritten_queries_df = topics_or_res[["qid", "query"]].drop_duplicates()

        pipeline: List[Callable] = [
            self.__split_query_tokenize_join,
            self.__get_input_token_ids,
            self.__get_output_token_ids,
            self.__decode_token_ids,
            self.__remove_punctuation,
        ]

        rewritten_queries_df["query"] = rewritten_queries_df["query"].apply(
            lambda q: _call_list_of_functions(q, pipeline)
        )

        # overwrite the query column with the decoded output token ids
        rewritten_queries_df.merge(
            pt.model.push_queries(topics_or_res, "query"), on="qid"
        )
        rewritten_queries_df[COPY_REWRITTEN_QUERY_COLUMN] = rewritten_queries_df[
            "query"
        ]

        logging.info(f"Rewritten queries: {rewritten_queries_df['query'].unique()}")

        return rewritten_queries_df


def _call_list_of_functions(x: Any, pipeline: List[Callable]) -> Any:
    """
    Call a list of functions on an input.

    Parameters
    ----------
    x : Any
        The input.
    pipeline : List[Callable]
        The list of functions to call.

    Returns
    -------
    Any
        The output of the last function.
    """
    for f in pipeline:
        x = f(x)
    return x
