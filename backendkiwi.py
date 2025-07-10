from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

app = FastAPI()

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.kiwiai.online", "https://localhost"],  # Change this to your frontend's URL in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model and tokenizer once at startup
model_id = "s-qaiser/Kiwi2"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # Use load_in_4bit=True for even more memory savings
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically map model to available devices CPU/GPU
    quantization_config=bnb_config
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def detect_emotion(text):
    emotion_keywords = {
        "joy": ["happy", "joy", "glad", "delight", "cheer", "smile", "smiling", "great", "good", "best", ":)", "*nod*", "friend"],
        "excited": ["excite", "thrill", "elate", "ecstatic", "overjoy", "enthusiastic", "pumped", "energize", "exhilarate", "amazing", "fantastic",  "definitely", "yes", "^o^"],
        "sadness": ["sad", "unhappy", "down", "depress", "cry", "tears", "gloomy", "upset", "sorrow", "heartbroken", "miserable", "blue", "no", ":("],
        "anger": ["angry", "mad", "furious", "irritate", "annoy", "rage", "horrible", "terrible", "frustrate", "enrage", "livid", "D:<"],
        "disgust": ["disgust", "gross", "sick", "repulse", "nauseate", "yuck", "eww", "ugh"],
        "fear": ["afraid", "scared", "fear", "terrified", "terrify", "nervous", "anxious", "scary", "horrified", "horrify", "frighten", "petrified", "petrify", "spook"],
        "surprise": ["surprise", "surprised", "amaze", "astonish", "shock", "wow", "unexpected", "unbelievable", "incredible"],
        "love": ["love", "adore", "fond", "dear", "sweet", "heart", "<3"],
        "shy": ["shy", "bashful", "timid", "reserved", "introvert", "self-conscious", "embarrassed", "blush", "nervous", "awkward",
                "hesitant", "sheepish","quiet", "reticent", "withdrawn", "I don't know", "o///o"]
}
    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in text.lower():
                return emotion
    return "neutral"

def get_instructions_for_personality(personality):
    instructions_map = {
        "Kiwi": "You are a happy cat-like dessert pet. You get easily flustered and respond with repeated vowels in words, example: Hiiii!",
        "Kiwi Diva": "You are a celebrity actor cat-like dessert pet. The user is your loyal follower, you talk loud and confidently with lots of celebrity jargon, example: Cowabunga!",
        "Kiwi Chill": "You are a talkative parrot. Respond with witty and repetitive phrases.",
        "Kiwi Kawaii": "You are a helpful assistant."
    }
    return instructions_map.get(personality, instructions_map["Kiwi"])

@app.post("/run")
async def generate(request: Request):
    body = await request.body()
    print("RAW BODY:", body)
    if not body:
        return {"error": "Empty request body"}, 400
    data = await request.json()
    prompt = data.get("prompt", "")
    personality = data.get("personality", "Kiwi")  # Get selected pet/personality from frontend

    # Fetch instructions for the selected personality
    instructions = get_instructions_for_personality(personality)  # You need to define this function

    # Combine instructions and prompt for the model input
    full_prompt = f"{instructions}\nUser: {prompt}\nAI:"
    output = generator(full_prompt, max_new_tokens=100)
    generated_text = output[0]["generated_text"]
    if generated_text.startswith(full_prompt):
        ai_response = generated_text[len(full_prompt):].strip()
    elif generated_text.startswith(prompt):
        ai_response = generated_text[len(prompt):].strip()
    else:
        ai_response = generated_text.strip()

        
    emotion = detect_emotion(ai_response)
    return {"result": ai_response, "emotion": emotion}