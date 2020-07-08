import json
import logging
import requests
from dataclasses import dataclass
from random import choice
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from typing import List, Tuple


@dataclass
class MemeCategory:
    token: str
    num_boxes: Tuple[int]

    def __post_init__(self):
        self.name = self.token[2:-2]


END_OF_BOX_TOKEN = "<|endofbox|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
IMGFLIP_API = "https://api.imgflip.com/caption_image"
FAILED_URL = "https://cdn.oncheckin.com/blogassets/blog-d888cc31-b202-4676-bdbe-e01432534be7.png"
MAX_BOX_LENGTH = 60

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
    '40945639': MemeCategory('<|Dr-Evil-Laser|>', (2,)),
    '61733537': MemeCategory('<|Mr-Krabs-Blur-Meme|>', (2,)),
    '164335977': MemeCategory('<|Bird-Box|>', (2,)),
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


def get_logger(logger_name: str = "ModelAPILogger") -> logging.Logger:
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)
    return logging.getLogger(logger_name)


def is_caption_empty(caption: str):
    stripped = caption.replace(END_OF_BOX_TOKEN, "").replace(" ", "").strip()
    return len(stripped) == 0


def prepare_text_boxes(decoded_caption: str,
                       category: MemeCategory) -> List[str]:
    caption = decoded_caption.replace(category.token, "") \
        .replace(END_OF_TEXT_TOKEN, "")

    # Choose text for image boxes
    num_boxes = choice(category.num_boxes)
    text_boxes = caption.split(END_OF_BOX_TOKEN)
    text_boxes = list(
        filter(lambda box: not is_caption_empty(box), text_boxes))
    text_boxes = [text_box.strip()[:MAX_BOX_LENGTH]
                  for text_box in text_boxes[:num_boxes]]
    return text_boxes


def generate_caption(category: MemeCategory,
                     tokenizer: GPT2Tokenizer,
                     model: TFGPT2LMHeadModel,
                     logger: logging.Logger) -> List[str]:
    logger.info("Sending request to the model")
    model_input = tokenizer.encode(category.token, return_tensors="tf")
    model_output = model.generate(
        model_input,
        do_sample=True,
        max_length=80,
        top_p=0.96,
        top_k=0,
        temperature=0.5,
        pad_token_id=50256
    )
    caption = tokenizer.decode(model_output[0], skip_special_tokens=False)
    logger.info(f"Decoded caption: {caption}")
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


def is_valid_caption(text_boxes: List[str]) -> bool:
    caption = " ".join(text_boxes)
    return not is_caption_empty(caption) and len(caption) > 2


def too_many_retries(retry_count: int) -> bool:
    return retry_count > 5


def generate_meme(category_id: str, tokenizer: GPT2Tokenizer,
                  model: TFGPT2LMHeadModel, api_credentials: dict,
                  logger: logging.Logger) -> str:
    category = CATEGORIES[category_id]
    logger.info(f"Generating caption for category {category.name}")
    caption = generate_caption(category, tokenizer, model, logger)

    retry_count = 0
    while not is_valid_caption(caption) and not too_many_retries(retry_count):
        logger.info(f"Generated an invalid caption, retrying...")
        retry_count += 1
        caption = generate_caption(category, tokenizer, model, logger)

    if not is_valid_caption(caption):
        logger.error(f"Failed to render a valid caption 5 times")
        return FAILED_URL

    logger.info(f"Rendering caption: "
                f"CATEGORY: {category.name} "
                f"CAPTION: '{' '.join(caption)}'")

    image_url = get_meme_url(caption, category_id, api_credentials)
    logger.info(f"Image url: {image_url}")

    return image_url


def get_model_api():
    api_credentials = get_api_credentials()
    tokenizer = get_tokenizer()
    model = get_model()
    logger = get_logger()

    def model_api_lambda(category_id):
        return generate_meme(category_id, tokenizer, model, api_credentials,
                             logger)

    return model_api_lambda
