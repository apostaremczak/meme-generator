import json
import requests
from dataclasses import dataclass
from random import choice
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from typing import List, Tuple


@dataclass
class MemeCategory:
    token: str
    num_boxes: Tuple[int]


END_OF_BOX_TOKEN = "<|endofbox|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
IMGFLIP_API = "https://api.imgflip.com/caption_image"

CATEGORIES = {
    '188390779': MemeCategory('<|Woman-Yelling-At-Cat|>', (2,)),
    '155067746': MemeCategory('<|Surprised-Pikachu|>', (2,)),
    '87743020': MemeCategory('<|Two-Buttons|>', (2, 3)),
    '119139145': MemeCategory('<|Blank-Nut-Button|>', (1, 2)),
    '112126428': MemeCategory('<|Distracted-Boyfriend|>', (3,)),
    '178591752': MemeCategory('<|Tuxedo-Winnie-The-Pooh|>', (2,)),
    '131087935': MemeCategory('<|Running-Away-Balloon|>', (3, 4, 5)),
    '196652226': MemeCategory('<|Spongebob-Ight-Imma-Head-Out|>', (1,)),
    '134797956': MemeCategory('<|American-Chopper-Argument|>', (5,)),
    '28251713': MemeCategory('<|Oprah-You-Get-A|>', (2,)),
    '27813981': MemeCategory('<|Hide-the-Pain-Harold|>', (1, 2)),
    '100777631': MemeCategory('<|Is-This-A-Pigeon|>', (1, 2, 3)),
    '114585149': MemeCategory('<|Inhaling-Seagull|>', (2, 3, 4)),
    '175540452': MemeCategory('<|Unsettled-Tom|>', (1, 2)),
    '84341851': MemeCategory('<|Evil-Kermit|>', (2,)),
    '132769734': MemeCategory('<|Hard-To-Swallow-Pills|>', (1,)),
    '93895088': MemeCategory('<|Expanding-Brain|>', (4,)),
    '129242436': MemeCategory('<|Change-My-Mind|>', (1,)),
    '102156234': MemeCategory('<|Mocking-Spongebob|>', (1, 2)),
    '181913649': MemeCategory('<|Drake-Hotline-Bling|>', (2,)),
    '135256802': MemeCategory('<|Epic-Handshake|>', (3,)),
    '16464531': MemeCategory('<|But-Thats-None-Of-My-Business|>', (2,)),
    '184801100': MemeCategory('<|Me-And-The-Boys|>', (1,)),
    '3218037': MemeCategory('<|This-Is-Where-Id-Put-My-Trophy-If-I-Had-One|>',
                            (2,)),
    '170715647': MemeCategory('<|Well-Yes-But-Actually-No|>', (1,)),
    '163573': MemeCategory('<|Imagination-Spongebob|>', (1, 2)),
    '61581': MemeCategory('<|Put-It-Somewhere-Else-Patrick|>', (2,)),
    '74191766': MemeCategory('<|Arthur-Fist|>', (1,)),
    '124822590': MemeCategory('<|Left-Exit-12-Off-Ramp|>', (2, 3)),
    '438680': MemeCategory('<|Batman-Slapping-Robin|>', (2,)),
    '1035805': MemeCategory('<|Boardroom-Meeting-Suggestion|>', (4,)),
    '89370399': MemeCategory('<|Roll-Safe-Think-About-It|>', (1, 2)),
    '40945639': MemeCategory('<|Dr-Evil-Laser|>', (1,)),
    '61733537': MemeCategory('<|Mr-Krabs-Blur-Meme|>', (1, 2)),
    '164335977': MemeCategory('<|Bird-Box|>', (1, 2)),
    '10364354': MemeCategory('<|Its-Not-Going-To-Happen|>', (2,)),
    '326093': MemeCategory('<|Ill-Have-You-Know-Spongebob|>', (1, 2)),
    '4087833': MemeCategory('<|Waiting-Skeleton|>', (1, 2)),
    '135678846': MemeCategory('<|Who-Killed-Hannibal|>', (2, 3, 4)),
    '61520': MemeCategory('<|Futurama-Fry|>', (1, 2)),
    '217743513': MemeCategory('<|UNO-Draw-25-Cards|>', (1, 2)),
    '222403160': MemeCategory(
        '<|Bernie-I-Am-Once-Again-Asking-For-Your-Support|>', (1,)),
    '148909805': MemeCategory('<|Monkey-Puppet|>', (1, 2, 3)),
    '226297822': MemeCategory('<|Panik-Kalm-Panik|>', (2, 3))
}


def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("model/tokenizer/")


def get_model():
    return TFGPT2LMHeadModel.from_pretrained("model/trained/")


def get_api_credentials():
    with open("credentials.json") as f:
        credentials = json.load(f)
    return credentials


def is_caption_empty(caption: str):
    stripped = caption.replace(END_OF_BOX_TOKEN, "").replace(" ", "")
    return len(stripped) == 0


def prepare_text_boxes(decoded_caption: str,
                       category: MemeCategory) -> List[str]:
    caption = decoded_caption.replace(category.token, "")

    # Find the first meme caption
    caption = caption.split(END_OF_TEXT_TOKEN)
    caption = " ".join(list(filter(lambda cap: not is_caption_empty(cap),
                                   caption))[:1])

    # Choose text for image boxes
    num_boxes = choice(category.num_boxes)
    text_boxes = caption.split(END_OF_BOX_TOKEN)
    text_boxes = list(
        filter(lambda box: not is_caption_empty(box), text_boxes))
    text_boxes = [text_box.strip() for text_box in text_boxes[:num_boxes]]
    return text_boxes


def generate_caption(category_id: str,
                     tokenizer: GPT2Tokenizer,
                     model: TFGPT2LMHeadModel) -> List[str]:
    category = CATEGORIES[category_id]
    model_input = tokenizer.encode(category.token, return_tensors="tf")
    model_output = model.generate(
        model_input,
        do_sample=True,
        max_length=80,
        top_k=3,
        temperature=0.7,
        pad_token_id=50256
    )
    caption = tokenizer.decode(model_output[0], skip_special_tokens=False)
    return prepare_text_boxes(caption, category)


def get_meme_url(text_boxes: List[str], category_id: str,
                 api_credentials: dict) -> str:
    request_data = api_credentials.copy()
    request_data["template_id"] = category_id

    # Handle misplaced text on poor Bernie
    if category_id == "222403160":
        request_data["text0"] = " "
        request_data["text1"] = text_boxes[0]
    else:
        for box_index, text_box in enumerate(text_boxes):
            request_data[f"boxes[{box_index}][text]"] = text_box

    imgflip_response = requests.request("POST", IMGFLIP_API,
                                        params=request_data).json()
    assert imgflip_response["success"], \
        f"Failed to post a meme on ImgFlip: {imgflip_response}"
    return imgflip_response["data"]["url"]


def generate_meme(category_id: str, tokenizer: GPT2Tokenizer,
                  model: TFGPT2LMHeadModel, api_credentials: dict) -> str:
    caption = generate_caption(category_id, tokenizer, model)

    while is_caption_empty(" ".join(caption)):
        caption = generate_caption(category_id, tokenizer, model)
    return get_meme_url(caption, category_id, api_credentials)


def get_model_api():
    api_credentials = get_api_credentials()
    tokenizer = get_tokenizer()
    model = get_model()

    def model_api_lambda(category_id):
        return generate_meme(category_id, tokenizer, model, api_credentials)

    return model_api_lambda
