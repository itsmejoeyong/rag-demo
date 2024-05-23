from src.text_handlers import TextHandler

import pytest


@pytest.fixture
def text_handler():
    return TextHandler()


def test_split_text_recursively(text_handler):
    text = """
    dignissim cras tincidunt lobortis feugiat vivamus at augue eget arcu dictum varius duis at consectetur lorem donec massa sapien faucibus et molestie ac feugiat sed lectus vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt ornare massa eget egestas purus viverra accumsan in nisl nisi scelerisque eu ultrices vitae auctor eu augue ut lectus arcu bibendum at varius vel pharetra vel turpis nunc eget lorem dolor sed viverra ipsum nunc aliquet bibendum enim facilisis gravida neque convallis a cras semper auctor neque vitae tempus quam pellentesque nec nam aliquam sem et tortor consequat id porta nibh venenatis cras sed felis eget
    """
    text = text_handler.split_text_recursively(
        chunk_size=400, chunk_overlap=0, data=text
    )
    assert len(text) == 2
    assert isinstance(text, list)
