"""RGB PNG encoder using filter 0 and DEFLATE level 0; pixels are unchanged."""
import struct
import zlib

def encode_rgb(image):
    if image.mode != 'RGB':
        raise ValueError('Expected decoded RGB pixels')
    width, height = image.size
    stride = width * 3
    pixels = image.tobytes()
    rows = b''.join(b'\x00' + pixels[i:i+stride] for i in range(0, len(pixels), stride))
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows, 0)) + chunk(b'IEND', b'')
