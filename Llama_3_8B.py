import transformers
import torch
from huggingface_hub import login
import accelerate
import os
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import torchvision
import speechbrain
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from moviepy.video.io.VideoFileClip import VideoFileClip
import sys
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)


login(token="API key here!")

video_file = "orginal_video.mp4"



input_video = video_file
output_path = ''

output_video = os.path.join(output_path, 'output_30fps.mp4')
ffmpeg_command = f"ffmpeg -i '{input_video}' -r 30 '{output_video}' -y"
os.system(ffmpeg_command)
print(f"Converted video to 30 fps and saved as {output_video}")




# Define the input video file and output audio file
mp4_file = "output_30fps.mp4"
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

print("Audio extraction successful!")


video_file = mp4_file





# load the pipeline from Hugginface Hub

video = VideoFileClip(video_file)

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="API key here!")

pipeline.to(torch.device("cuda"))
# apply the pipeline to an audio file
diarization = pipeline(mp3_file)

speaker_set = set()
# dump the diarization output to disk using RTTM format
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    speaker_set.add(speaker)


record = []
for turn, _, speaker in diarization.itertracks(yield_label=True):   
    record.append([turn.start, turn.end, speaker])
    
real_record = []
real_record.append(record[0])
for i in range(1,len(record)-1):
    if record[i][2] == record[i-1][2] and  record[i][2] == record[i+1][2]:
        continue
    else:
        real_record.append(record[i])
real_record.append(record[-1])


new_real_record = []
if real_record[0][2] != real_record[1][2]:
    new_real_record.append(real_record[0])
    new_real_record.append(real_record[0])
else:
        new_real_record.append(real_record[0])


for i in range(1, len(real_record)-1):
    if real_record[i][2] != real_record[i+1][2] and real_record[i][2] != real_record[i-1][2]:
        new_real_record.append(real_record[i])
        new_real_record.append(real_record[i])        
    else:
        new_real_record.append(real_record[i])
        
if real_record[-1][2] != real_record[-2][2]:
    new_real_record.append(real_record[-1])
    new_real_record.append(real_record[-1])
else:
    new_real_record.append(real_record[-1])

    
finial_new_real_record = []

for i in range(0,len(new_real_record)-1,2):
    if video.duration < new_real_record[i][0] or video.duration <new_real_record[i+1][1]:
        continue
    else:
        finial_new_real_record.append([new_real_record[i][0], new_real_record[i+1][1], new_real_record[i][2]])


os.mkdir("speaker_parts")
os.mkdir("speaker_parts_audio")




index = 0
video = VideoFileClip(video_file)
for start, end, speaker in finial_new_real_record:
    segment = video.subclip(start, end)
    output_file = f"speaker_parts/segment_{str(index)}_{str(speaker)}.mp4"
    segment.write_videofile(output_file)

    video_clip = VideoFileClip(output_file)

    # Extract the audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio to a separate file
    audio_clip.write_audiofile(f"speaker_parts_audio/segement_{index}_{speaker}.wav")

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()
    index += 1




  

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = transformers.pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)



def extract_segment_number(filename):
    match = re.search(r'segement_(\d+)_', filename)
    return int(match.group(1)) if match else -1

# Sort files by segment number
sorted_files = sorted(os.listdir(f"speaker_parts_audio/"), key=extract_segment_number)

conversation_dict = {}
for file in sorted_files:
    result = pipe("speaker_parts_audio/" + file)
    conversation_dict[file[:-4]] = result['text']




torch.random.manual_seed(0)



model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    
 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

messages = [
    {
        "role": "user",
        "content": f"Identify in one word the host speaker who ask the questions from the following list: {', '.join(speaker_set)}. Here is the conversation:\n\n{conversation_dict}"
    }
]


pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 7,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

host = str(output[0]['generated_text'])


new_keys = []
for s in conversation_dict:
    new_keys.append(s.replace(host[1:], "HOST"))
    
new_conversation = {}
for k, v in conversation_dict.items():
    new_conversation[k.replace(host[1:], "HOST")] = v


print("Conversation: \n")
print(new_conversation)






model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token="API key here!",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


individual_assisemnts = []
for speaker in speaker_set:
    if speaker == host[1:]:
        speaker = 'HOST'
    messages = [
        {"role": "system", "content": 
        """Role: 
            Biased Detector
            Instructions: 
            Review the provided conversation to detect the orientation of biase for the demanded speaker by the user regarding the topic discussed. Provide your assessment strictly only in the JSON format specified below.
            Format:
            only in this Format with out any explication:
            {"speaker": "Speaker_A", "conversation_topic": "Topic_A", "Orientation": "Against or with"}
            Example:
            {"speaker": "Speaker_00", "conversation_topic": "Immigration", "Orientation": "Against"}"""},
        {"role": "user", "content": f"Evaluate the '{speaker}' speaker position only in Json output from this conversation:  {new_conversation}"},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][-1])
    individual_assisemnts.append(outputs[0]["generated_text"][-1])



messages = [
        {"role": "system", "content": 
        """Role: 
            Biased Detector
            Instructions: 
            Review the provided conversation with the provided speakers assessment to detect the general orientation of biase for the conversation regarding the topic discussed. Provide your assessment strictly only in the JSON format specified below.
            Format:
            only in this Format with out any explication:
            {"conversation_topic": "Topic_A", "Orientation": "Against or with"}"""},
        {"role": "user", "content": f"Evaluate this conversation position only in Json output based on this speakers assessment: {str(individual_assisemnts)} and from this conversation:  {new_conversation}"},
    ]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
)

print(outputs[0]["generated_text"][-1])

with open(f"Biase_Results_{str(sys.argv[1])}.txt", "a") as file:
    file.write("Video Title: \n"+str(sys.argv[2])+"\n")
    file.write("Individual Assignments: \n")
    file.writelines(str(individual_assisemnts)+"\n")
    file.write("Global Assignments: \n")
    file.write(str(outputs[0]["generated_text"][-1])+"\n \n")
