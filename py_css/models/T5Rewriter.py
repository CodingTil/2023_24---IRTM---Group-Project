import models.base as base_module

import string

import pyterrier as pt
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME: str = "castorini/t5-base-canard"
MAX_LENGTH: int = 512
NUM_BEAMS: int = 10
EARLY_STOPPING: bool = True


class T5Rewriter(pt.Transformer):
    device: torch.device
    tokenizer: T5Tokenizer
    model: T5ForConditionalGeneration

    def __init__(self, index):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            .to(self.device)
            .eval()
        )
        super().__init__()

    # the query has multiply " <sep> " in it. Create a list of the split with a maximum of 3 elements (last element is last, second last is middle, and the first n are joined)
    def split_query_tokenize_join(self, q):
        l = q.split(" <sep> ")
        if len(l) < 3:
            tokens = []
            for ll in l:
                tokens.extend(self.tokenizer.tokenize(ll))
                tokens.append(" <sep> ")
            if len(tokens) > 0:
                tokens.pop()
            return tokens
        else:
            tokens = []
            tokens.extend(self.tokenizer.tokenize(" <sep> ".join(l[:-2])))
            tokens.append(" <sep> ")
            tokens.extend(self.tokenizer.tokenize(l[-2]))
            tokens.append(" <sep> ")
            tokens.extend(self.tokenizer.tokenize(l[-1]))
            return tokens

    def get_input_token_ids(self, tokens):
        return self.tokenizer.encode(
            tokens, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

    def get_output_token_ids(self, input_token_ids):
        return self.model.generate(
            input_token_ids,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
        )

    def decode_token_ids(self, token_ids):
        return self.tokenizer.decode(
            token_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def remove_punctuation(self, s):
        return s.translate(str.maketrans("", "", string.punctuation))

    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        # save qid and query columns as dict (qid -> query) query is same for same qid, so sufficient to select first
        qid_query_dict = dict(zip(topics_or_res["qid"], topics_or_res["query"]))
        # tokenize the query
        tokenized_queries = {
            qid: self.split_query_tokenize_join(q) for qid, q in qid_query_dict.items()
        }
        # get input token ids
        input_token_ids = {
            qid: self.get_input_token_ids(tokens)
            for qid, tokens in tokenized_queries.items()
        }
        # get output token ids
        output_token_ids = {
            qid: self.get_output_token_ids(token_ids)
            for qid, token_ids in input_token_ids.items()
        }
        # decode output token ids
        decoded_output_token_ids = {
            qid: self.decode_token_ids(token_ids)
            for qid, token_ids in output_token_ids.items()
        }
        # remove punctuation from decoded output token ids
        decoded_output_token_ids = {
            qid: self.remove_punctuation(s)
            for qid, s in decoded_output_token_ids.items()
        }
        # overwrite the query column with the decoded output token ids
        topics_or_res["query"] = topics_or_res["qid"].map(
            lambda qid: decoded_output_token_ids[qid]
        )

        # print unique queries
        print(topics_or_res["query"].unique())

        return topics_or_res
