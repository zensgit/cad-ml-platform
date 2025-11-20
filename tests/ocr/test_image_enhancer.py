from src.core.ocr.preprocessing.image_enhancer import enhance_image_for_ocr


def test_enhancer_returns_bytes_even_without_pil():
    data = b"not_an_image_but_should_round_trip"
    out, arr = enhance_image_for_ocr(data, max_res=128)
    assert isinstance(out, (bytes, bytearray))
    # If PIL unavailable, enhancer returns original bytes
    # If PIL available, it returns PNG bytes; both are acceptable for this smoke test
