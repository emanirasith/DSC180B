from typing import AnyStr


# Modified from https://stackoverflow.com/a/42450252
def camel_case(snake: AnyStr) -> AnyStr:
    match snake:
        case str():
            assert isinstance(snake, str)
            first, *others = snake.split('_')
            return ''.join([first.lower(), *map(str.title, others)])
        case bytes():
            assert isinstance(snake, bytes)
            first, *others = snake.split(b'_')
            return b''.join([first.lower(), *map(bytes.title, others)])
        case _:
            raise TypeError(f'Expected str or bytes, got {type(snake)}')
