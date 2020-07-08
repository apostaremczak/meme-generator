import json
import requests
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

END_OF_BOX_TOKEN = "<|endofbox|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
IMGFLIP_API = "https://api.imgflip.com/caption_image"

CATEGORIES = {
    '188390779': '<|Woman-Yelling-At-Cat|>',
    '155067746': '<|Surprised-Pikachu|>',
    '87743020': '<|Two-Buttons|>',
    '119139145': '<|Blank-Nut-Button|>',
    '112126428': '<|Distracted-Boyfriend|>',
    '178591752': '<|Tuxedo-Winnie-The-Pooh|>',
    '131087935': '<|Running-Away-Balloon|>',
    '196652226': '<|Spongebob-Ight-Imma-Head-Out|>',
    '134797956': '<|American-Chopper-Argument|>',
    '28251713': '<|Oprah-You-Get-A|>',
    '27813981': '<|Hide-the-Pain-Harold|>',
    '100777631': '<|Is-This-A-Pigeon|>',
    '114585149': '<|Inhaling-Seagull|>',
    '175540452': '<|Unsettled-Tom|>',
    '84341851': '<|Evil-Kermit|>',
    '132769734': '<|Hard-To-Swallow-Pills|>',
    '93895088': '<|Expanding-Brain|>',
    '129242436': '<|Change-My-Mind|>',
    '102156234': '<|Mocking-Spongebob|>',
    '181913649': '<|Drake-Hotline-Bling|>',
    '135256802': '<|Epic-Handshake|>',
    '16464531': '<|But-Thats-None-Of-My-Business|>',
    '184801100': '<|Me-And-The-Boys|>',
    '3218037': '<|This-Is-Where-Id-Put-My-Trophy-If-I-Had-One|>',
    '170715647': '<|Well-Yes-But-Actually-No|>',
    '163573': '<|Imagination-Spongebob|>',
    '61581': '<|Put-It-Somewhere-Else-Patrick|>',
    '74191766': '<|Arthur-Fist|>',
    '124822590': '<|Left-Exit-12-Off-Ramp|>',
    '438680': '<|Batman-Slapping-Robin|>',
    '1035805': '<|Boardroom-Meeting-Suggestion|>',
    '89370399': '<|Roll-Safe-Think-About-It|>',
    '40945639': '<|Dr-Evil-Laser|>',
    '61733537': '<|Mr-Krabs-Blur-Meme|>',
    '164335977': '<|Bird-Box|>',
    '10364354': '<|Its-Not-Going-To-Happen|>',
    '326093': '<|Ill-Have-You-Know-Spongebob|>',
    '4087833': '<|Waiting-Skeleton|>',
    '135678846': '<|Who-Killed-Hannibal|>',
    '61520': '<|Futurama-Fry|>',
    '217743513': '<|UNO-Draw-25-Cards|>',
    '222403160': '<|Bernie-I-Am-Once-Again-Asking-For-Your-Support|>',
    '148909805': '<|Monkey-Puppet|>',
    '226297822': '<|Panik-Kalm-Panik|>'
}


def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("model/tokenizer/")


def get_model():
    return TFGPT2LMHeadModel.from_pretrained("model/trained/")


def get_api_credentials():
    with open("credentials.json") as f:
        credentials = json.load(f)
    return credentials


def generate_caption(category_id: str,
                     tokenizer: GPT2Tokenizer,
                     model: TFGPT2LMHeadModel) -> str:
    category_token = CATEGORIES[category_id]
    model_input = tokenizer.encode(category_token, return_tensors="tf")
    model_output = model.generate(
        model_input,
        do_sample=True,
        max_length=50,
        top_k=0,
        temperature=0.7,
        pad_token_id=50256
    )
    caption = tokenizer.decode(model_output[0], skip_special_tokens=False)
    caption = caption.replace(category_token, "")
    caption = caption.split(END_OF_TEXT_TOKEN, 1)[0]
    return caption


def get_meme_url(caption: str, category_id: str, api_credentials: dict) -> str:
    text_boxes = caption.split(END_OF_BOX_TOKEN)[:5]

    request_data = api_credentials.copy()
    request_data["template_id"] = category_id

    for box_index, text_box in enumerate(text_boxes):
        request_data[f"boxes[{box_index}][text]"] = text_box

    imgflip_response = requests.request("POST", IMGFLIP_API,
                                        params=request_data).json()
    assert imgflip_response["success"], \
        f"Failed to post a meme on ImgFlip: {imgflip_response}"
    return imgflip_response["data"]["url"]


def is_caption_empty(caption: str):
    stripped = caption.replace(END_OF_BOX_TOKEN, "").replace(" ", "")
    return len(stripped) == 0


def generate_meme(category_id: str, tokenizer: GPT2Tokenizer,
                  model: TFGPT2LMHeadModel, api_credentials: dict) -> str:
    caption = generate_caption(category_id, tokenizer, model)
    while is_caption_empty(caption):
        caption = generate_caption(category_id, tokenizer, model)
    return get_meme_url(caption, category_id, api_credentials)


def get_model_api():
    api_credentials = get_api_credentials()
    tokenizer = get_tokenizer()
    model = get_model()

    def model_api_lambda(category_id):
        return generate_meme(category_id, tokenizer, model, api_credentials)

    return model_api_lambda
