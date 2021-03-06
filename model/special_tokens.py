"""
Defines all special tokens to be used with a model encoder
"""

CATEGORY_TOKENS = [
    "<|Woman-Yelling-At-Cat|>",
    "<|Surprised-Pikachu|>",
    "<|Two-Buttons|>",
    "<|Blank-Nut-Button|>",
    "<|Distracted-Boyfriend|>",
    "<|Tuxedo-Winnie-The-Pooh|>",
    "<|Running-Away-Balloon|>",
    "<|Spongebob-Ight-Imma-Head-Out|>",
    "<|American-Chopper-Argument|>",
    "<|Oprah-You-Get-A|>",
    "<|Hide-the-Pain-Harold|>",
    "<|Is-This-A-Pigeon|>",
    "<|Inhaling-Seagull|>",
    "<|Unsettled-Tom|>",
    "<|Evil-Kermit|>",
    "<|Hard-To-Swallow-Pills|>",
    "<|Expanding-Brain|>",
    "<|Change-My-Mind|>",
    "<|Mocking-Spongebob|>",
    "<|Drake-Hotline-Bling|>",
    "<|Epic-Handshake|>",
    "<|But-Thats-None-Of-My-Business|>",
    "<|Me-And-The-Boys|>",
    "<|This-Is-Where-Id-Put-My-Trophy-If-I-Had-One|>",
    "<|Well-Yes-But-Actually-No|>",
    "<|Imagination-Spongebob|>",
    "<|Put-It-Somewhere-Else-Patrick|>",
    "<|Arthur-Fist|>",
    "<|Left-Exit-12-Off-Ramp|>",
    "<|Batman-Slapping-Robin|>",
    "<|Boardroom-Meeting-Suggestion|>",
    "<|Roll-Safe-Think-About-It|>",
    "<|Dr-Evil-Laser|>",
    "<|Mr-Krabs-Blur-Meme|>",
    "<|Bird-Box|>",
    "<|Its-Not-Going-To-Happen|>",
    "<|Ill-Have-You-Know-Spongebob|>",
    "<|Waiting-Skeleton|>",
    "<|Who-Killed-Hannibal|>",
    "<|Futurama-Fry|>",
    "<|UNO-Draw-25-Cards|>",
    "<|American-Chopper-Argument|>",
    "<|Bernie-I-Am-Once-Again-Asking-For-Your-Support|>",
    "<|Monkey-Puppet|>",
    "<|Panik-Kalm-Panik|>"
]

END_OF_BOX_TOKEN = "<|endofbox|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"

SPECIAL_TOKENS = CATEGORY_TOKENS + [END_OF_BOX_TOKEN]
