import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    """
    doc: string of a single document
    returns a numpy array of unit 16 tokens
    """
 
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)

    ## note: remove below token check, checking once is enough 
    ## assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
   
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16