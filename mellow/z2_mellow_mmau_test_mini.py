import pandas as pd
from tqdm import tqdm
import json
import os
from mellow import MellowWrapper
import torch

def evaluate_metric(preds, answers, metadata):
    corr = 0
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    # compute metrics
    for i in range(len(preds)):
        answer = answers[i]
        response = preds[i]
        correct = True if response.split(")")[0].lower() == answer.split(")")[0].lower() else False

        task = metadata[i]['task']
        difficulty = metadata[i]['difficulty']

        if correct:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            corr += 1

        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1

    # Parse, collect and return metrics
    scores = {t: {} for t in ['sound','music','speech','easy','hard','medium','total','main']}
    for task in task_metrics:
        scores[task]['score'] = (task_metrics[task][0]/task_metrics[task][1])*100 if task_metrics[task][1] != 0 else 0
    for diff in diff_metrics:
        scores[diff]['score'] = (diff_metrics[diff][0]/diff_metrics[diff][1])*100 if diff_metrics[diff][1] != 0 else 0
    scores["total"]['score'] = (corr/len(preds)) * 100
    scores["main"]["score"] = scores["total"]['score']
    return scores

if __name__ == "__main__":
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
    audio_basepath = "test-mini-audios" # Should point to the test-mini-audios folder
    id2int = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    int2id = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    with open("mmau-test-mini.json", 'r') as fp:
        data = json.load(fp)

    for i in tqdm(range(len(data))):
        question = data[i]["question"][:-1] + "? "+ " ".join([int2id[k] + ") " + data[i]["choices"][k] for k in range(len(data[i]["choices"]))])
        question = question.lower()
        answer = [int2id[k] + ") " + data[i]["answer"].lower() for k in range(len(data[i]["choices"])) if data[i]["answer"].lower() == data[i]["choices"][k].lower()][0]
        audio_path = os.path.join(audio_basepath, data[i]["id"] + ".wav")

        examples = [
            [audio_path, audio_path, question]
        ]

        # generate response
        response = mellow.generate(examples=examples, max_len=300, top_p=0.8, temperature=1.0)
        data[i]["prediction"] = response[0]
        data[i]["answer"] = answer

    preds = [data[i]["prediction"] for i in range(len(data))]
    answers = [data[i]["answer"] for i in range(len(data))]
    scores = evaluate_metric(preds, answers, data)
    print(scores)