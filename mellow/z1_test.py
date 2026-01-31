# test the model on various tasks
import torch
from pathlib import Path
import os
from mellow import MellowWrapper

# setup cuda and device
cuda = torch.cuda.is_available()
device = 0 if cuda else "cpu"

# setup mellow
mellow = MellowWrapper(
                    config = "v0",
                    model = "v0",
                    device=device,
                    use_cuda=cuda,
                )

# pick up audio file paths
parent_path = Path(os.path.realpath(__file__)).parent
path1 = os.path.join(parent_path, "resource", "1.wav")
path2 = os.path.join(parent_path, "resource", "2.wav")

# list of filepaths and prompts
examples = [
    [path1, path2, "what can you infer about the surroundings from the audio?"],
    [path1, path2, "is there a cat in the audio? answer yes or no"],
    [path1, path2, "caption the audio."],
    [path1, path2, "Based on the audio, what can be said about the hypothesis - \"A farmer is giving a tour of his ranch while chickens roam nearby\"? a) It is definitely true b) It is definitely false c) It is plausible d) I cannot determine"],
    [path1, path2, "explain the difference between the two audios in detail."],
    [path1, path2, "what is the primary sound event present in the clip? a) dog barking b) chirping birds c) car engine d) clapping"],
]

# generate response
response = mellow.generate(examples=examples, max_len=300, top_p=0.8, temperature=1.0)
print(f"\noutput: {response}")