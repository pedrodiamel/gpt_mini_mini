import pytest
import llms.datasets.utils as utils


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
MD5 = "6fb458f1232090904fb40fe944165e91"
PATH = "/.datasets/llms/tinyshakespeare/input.txt"
FORCE = False


def test_download():
    utils.download_url(URL, PATH, MD5, FORCE)


if __name__ == "__main__":
    pytest.main([__file__])
