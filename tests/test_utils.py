import hypothesis.strategies as st
from hypothesis import given

from heterodyne.util import camel_case


@given(st.lists(st.text(st.characters(whitelist_categories=['Ll']))))
def test_camel_case_str(parts: list[str]):
    if parts:
        snake = '_'.join(parts)
        first, *rest = parts
        camel = ''.join([first, *map(str.title, rest)])
    else:
        snake = camel = ''
    assert camel == camel_case(snake)


@given(
    st.lists(
        st.text(
            st.characters(whitelist_categories=['Ll'], max_codepoint=128)
        ).map(lambda s: s.encode('ascii'))
    )
)
def test_camel_case_bytes(parts: list[bytes]):
    if parts:
        snake = b'_'.join(parts)
        first, *rest = parts
        camel = b''.join([first, *map(bytes.title, rest)])
    else:
        snake = camel = b''
    assert camel == camel_case(snake)
