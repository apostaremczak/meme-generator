import unittest
from web_app.model_api import MemeCategory
from web_app.model_api import is_caption_empty, prepare_text_boxes


class TestModelApi(unittest.TestCase):
    def test_detecting_empty_captions(self):
        self.assertTrue(is_caption_empty("        "))
        self.assertTrue(is_caption_empty("<|endofbox|>"))
        self.assertTrue(is_caption_empty(""))
        self.assertTrue(is_caption_empty("<|endofbox|><|endofbox|>"))
        self.assertTrue(is_caption_empty("    <|endofbox|>  "))
        self.assertFalse(is_caption_empty("  idk <|endofbox|>"))
        self.assertFalse(is_caption_empty("TEACHER <|endofbox|> SCHOOL"))

    def test_preparing_text_boxes(self):
        kermit = MemeCategory('<|Evil-Kermit|>', (2,))
        kermit_decoded = "<|Evil-Kermit|> YOUR STUPID <|endofbox|> YOU ARE " \
                         "MY OWN MAN <|endofbox|> YOUR STUPID <|endofbox|> Y"
        kermit_expected = ["YOUR STUPID", "YOU ARE MY OWN MAN"]
        kermit_result = prepare_text_boxes(kermit_decoded, kermit)
        self.assertListEqual(kermit_expected, kermit_result)

        spongebob = MemeCategory('<|Imagination-Spongebob|>', (1, 2))
        spongebob_decoded = "<|Imagination-Spongebob|> I JUST GOT OFF OF A " \
                            "PRIVATE CAREER! DIDN'T YOU TELL"
        spongebob_expected = [
            "I JUST GOT OFF OF A PRIVATE CAREER! DIDN'T YOU TELL"]
        spongebob_result = prepare_text_boxes(spongebob_decoded, spongebob)
        self.assertListEqual(spongebob_expected, spongebob_result)


if __name__ == '__main__':
    unittest.main()
