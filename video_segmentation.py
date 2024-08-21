import os
from moviepy.editor import VideoFileClip
import whisper
import nltk
from nltk.tokenize import sent_tokenize
import re
import sys
from tqdm import tqdm
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
import transformers
import json
torch.random.manual_seed(0)

login(token="API key here!")


video_file = sys.argv[2]


input_video = video_file
output_path = ''

output_video = os.path.join(output_path, 'output_30fps.mp4')

"""ffmpeg_command = f"ffmpeg -i '{input_video}' -r 30 '{output_video}' -y"
os.system(ffmpeg_command)"""



# Define the input video file and output audio file
mp4_file = video_file
mp3_file = "audio_file.wav"

# Load the video clip
video_clip = VideoFileClip(mp4_file)

# Extract the audio from the video clip
audio_clip = video_clip.audio

# Write the audio to a separate file
audio_clip.write_audiofile(mp3_file)

# Close the video and audio clips
audio_clip.close()
video_clip.close()

video_file = video_file


model = whisper.load_model("large")
transcript = model.transcribe(
    word_timestamps=True,
    audio=video_file,
    
)


time_stamped = []
full_text = []

for segment in transcript['segments']:
    for word in segment['words']:
        time_stamped.append([word['word'],word['start'],word['end']])
        full_text.append(word['word'])
full_text = "".join(full_text)




# Download the necessary resources
nltk.download('punkt')


# Tokenize the text into sentences
tokenized_sentences = sent_tokenize(full_text)
sentences = []

# Print the sentences
for i, sentence in enumerate(tokenized_sentences):
    sentences.append(sentence)



time_stamped_sentances = {}
count_sentances = {}

letter = 0
res = []
for i in tqdm(range(len(sentences))):
    tmp = []
    for j in range(len(sentences[i])):
        letter += 1
        tmp.append(sentences[i][j])
        
        f = 0
        for k in range(len(time_stamped)):
            for m in range(len(time_stamped[k][0])):
                f += 1
                if f == letter:
                    if j == 0:
                        start = time_stamped[k][1]
                    if j == (len(sentences[i])-1):
                        end = time_stamped[k][2]
    
    time_stamped_sentances["".join(tmp)] = [start, end]
    count_sentances[i+1] = "".join(tmp)






print(count_sentances)

print('######################################################')

subject = sys.argv[1]
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token="API key here!",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
messages = [
    {
    "role": "system",
    "content": """
    Role: Topic Segmentation
    Instructions: This is a video transcribe with each sentance have code of the order in the video,
    Extract the start and end code of sentence from the given script and provide them in JSON format only.
    Format: {
        "Topic_A_start": "sentance_code_x",
        "Topic_A_end": "sentance_code_y"
    }
    Example: {
        "Immigration_start": "24",
        "Immigration_end": "84"
    } """},
    {"role": "user", "content": f"Extract the '{subject}' start and end code only in JSON format from this script: {count_sentances}"}
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    temperature=0)
print(outputs[0]["generated_text"][-1])


res = []
for key in count_sentances:
    if str(key) in str(outputs[0]["generated_text"][-1]):
        res.append(key)

res.sort()




start = time_stamped_sentances[count_sentances[res[-2]]][0]
end = time_stamped_sentances[count_sentances[res[-1]]][1]

video = VideoFileClip(video_file)
segment = video.subclip(start, end)
output_file = f"orginal_video.mp4"
segment.write_videofile(output_file)

print("Success!")
