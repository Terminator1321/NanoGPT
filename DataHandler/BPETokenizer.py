from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

files = [
    r"C:/Users/Rohan/OneDrive/Desktop/Nano-GPT/DataHandler/merged/train.txt",
    r"C:/Users/Rohan/OneDrive/Desktop/Nano-GPT/DataHandler/merged/val.txt",
    r"C:/Users/Rohan/OneDrive/Desktop/Nano-GPT/DataHandler/chats/train.txt",
    r"C:/Users/Rohan/OneDrive/Desktop/Nano-GPT/DataHandler/chats/val.txt"
    
]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=4000,
    special_tokens=["[UNK]", "<EPIC>", "<PYTHON>", "<JAVASCRIPT>", "<WEB>", "<USER>" ,"<AI>","<END>"]
)

tokenizer.train(files, trainer)
tokenizer.save("bpe_tokenizer.json")