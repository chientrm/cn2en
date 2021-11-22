from unittest import TestCase

from cn2en.model import Model

class TestTranslate(TestCase):
    def test_translate(self):
        model = Model()
        assert model.translate('湯姆不在床上。') == "tom didn't go to bed ."

    
if __name__ == '__main__':
    unittest.main()