import datasets as ds
from transformers import AutoTokenizer
from tokenizer.bpe import ASCIIBPETokenizer, string_to_ascii

from pytest_utils.decorators import max_score


def test_not_injective():
    # you can try to break "google-bert/bert-base-cased"
    # or another tokenizer you like: https://huggingface.co/models?pipeline_tag=text-generation
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # BERT's basic tokenizer splits on whitespace and discards it, so any
    # amount of whitespace between two words maps to the same tokens
    s1 = "Hello world"
    s2 = "Hello   \t world"

    assert s1 != s2 and tokenizer.encode(
        s1, add_special_tokens=False
    ) == tokenizer.encode(s2, add_special_tokens=False)


def test_not_invertible():
    # you can try to break "google-bert/bert-base-cased"
    # or another tokenizer you like: https://huggingface.co/models?pipeline_tag=text-generation
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # characters missing from the vocabulary become [UNK], so the original
    # emoji can never be recovered
    s = "I love NLP 🤖"

    s_recovered = tokenizer.decode(tokenizer.encode(s, add_special_tokens=False))
    assert s != s_recovered


def test_not_preserving_concat():
    # you can try to break "google-bert/bert-base-cased"
    # or another tokenizer you like: https://huggingface.co/models?pipeline_tag=text-generation
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # WordPiece tokenizes "world" as a standalone word, but as the
    # continuation piece "##world" when it is glued to "hello"
    a = "hello"
    b = "world"
    assert tokenizer.encode(a + b, add_special_tokens=False) != tokenizer.encode(
        a, add_special_tokens=False
    ) + tokenizer.encode(b, add_special_tokens=False)


@max_score(5)
def test_encode():
    tokenizer = ASCIIBPETokenizer()
    random_ascii = r"[(A{u s=#M \tF\@`P({xAS%"
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)

    tokenizer = ASCIIBPETokenizer.from_data("123123", 1)
    ####### after a single merge #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)
    assert tokenizer.encode("11") == [49, 49]
    assert tokenizer.encode("12") == [128]
    assert tokenizer.encode("23") == [50, 51]
    assert tokenizer.encode("123123") == [128, 51, 128, 51]
    assert tokenizer.encode("12312312") == [128, 51, 128, 51, 128]

    tokenizer = ASCIIBPETokenizer.from_data("123123", 2)
    ####### after two merges #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    # "123" -> 129
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)
    assert tokenizer.encode("11") == [49, 49]
    assert tokenizer.encode("12") == [128]
    assert tokenizer.encode("23") == [50, 51]
    assert tokenizer.encode("123") == [129]
    assert tokenizer.encode("123123") == [129, 129]
    assert tokenizer.encode("12312312") == [129, 129, 128]


@max_score(5)
def test_decode():
    tokenizer = ASCIIBPETokenizer()
    assert (
        tokenizer.decode([97, 102, 101, 106, 117, 100, 111, 97, 119, 101, 114])
        == "afejudoawer"
    )


@max_score(5)
def test_one_merge():
    tokenizer = ASCIIBPETokenizer.from_data("123123", 1)
    ####### after a single merge #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    assert len(tokenizer.vocab) == 129
    assert tokenizer.vocab[128] == "12"
    assert len(tokenizer.merge_rules) == 1
    assert tokenizer.merge_rules[(49, 50)] == 128


@max_score(5)
def test_two_merges():
    tokenizer = ASCIIBPETokenizer.from_data("123123", 2)
    ####### after two merges #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    # "123" -> 129
    assert len(tokenizer.vocab) == 130
    assert tokenizer.vocab[129] == "123"
    assert len(tokenizer.merge_rules) == 2
    assert tokenizer.merge_rules[(49, 50)] == 128
    assert tokenizer.merge_rules[(128, 51)] == 129


@max_score(5)
def test_100_merges():
    import urllib.request
    
    # Download the text directly from Karpathy's repo
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url) as response:
        shakespear = response.read().decode('utf-8')

    tokenizer = ASCIIBPETokenizer.from_data(shakespear, 100)
    assert tokenizer.merge_rules == {
        (101, 32): 128,
        (116, 104): 129,
        (116, 32): 130,
        (115, 32): 131,
        (100, 32): 132,
        (44, 32): 133,
        (111, 117): 134,
        (101, 114): 135,
        (105, 110): 136,
        (121, 32): 137,
        (97, 110): 138,
        (58, 10): 139,
        (111, 114): 140,
        (111, 32): 141,
        (101, 110): 142,
        (10, 10): 143,
        (97, 114): 144,
        (32, 129): 145,
        (111, 110): 146,
        (108, 108): 147,
        (104, 97): 148,
        (44, 10): 149,
        (46, 143): 150,
        (105, 131): 151,
        (101, 115): 152,
        (121, 134): 153,
        (32, 115): 154,
        (116, 141): 155,
        (138, 132): 156,
        (111, 119): 157,
        (101, 97): 158,
        (32, 109): 159,
        (32, 119): 160,
        (111, 102): 161,
        (32, 104): 162,
        (136, 103): 163,
        (111, 109): 164,
        (32, 97): 165,
        (99, 104): 166,
        (129, 128): 167,
        (115, 116): 168,
        (32, 98): 169,
        (110, 111): 170,
        (105, 114): 171,
        (102, 140): 172,
        (118, 128): 173,
        (101, 133): 174,
        (105, 129): 175,
        (145, 128): 176,
        (115, 101): 177,
        (108, 105): 178,
        (84, 104): 179,
        (147, 32): 180,
        (114, 101): 181,
        (115, 130): 182,
        (97, 130): 183,
        (65, 110): 184,
        (73, 32): 185,
        (101, 144): 186,
        (105, 109): 187,
        (105, 116): 188,
        (111, 111): 189,
        (103, 104): 190,
        (97, 116): 191,
        (105, 115): 192,
        (108, 101): 193,
        (135, 32): 194,
        (134, 114): 195,
        (184, 132): 196,
        (39, 131): 197,
        (101, 101): 198,
        (170, 130): 199,
        (109, 137): 200,
        (59, 10): 201,
        (114, 97): 202,
        (46, 10): 203,
        (153, 114): 204,
        (117, 114): 205,
        (148, 130): 206,
        (114, 105): 207,
        (117, 130): 208,
        (108, 132): 209,
        (79, 139): 210,
        (161, 32): 211,
        (101, 132): 212,
        (108, 97): 213,
        (105, 130): 214,
        (114, 111): 215,
        (135, 128): 216,
        (101, 131): 217,
        (100, 133): 218,
        (117, 110): 219,
        (69, 78): 220,
        (107, 128): 221,
        (121, 133): 222,
        (73, 78): 223,
        (32, 100): 224,
        (63, 143): 225,
        (97, 131): 226,
        (102, 97): 227,
    }