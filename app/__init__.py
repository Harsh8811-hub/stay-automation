# StayInn AI Hotel Booking Chatbot
import sys
import types
if 'xxhash' not in sys.modules:
    try:
        import xxhash
    except Exception:
        m = types.ModuleType('xxhash')
        m.xxh64_hexdigest = lambda s, seed=0: '0'*16
        m.xxh3_64_hexdigest = lambda s, seed=0: '0'*16
        m.xxh128_hexdigest = lambda s, seed=0: '0'*32
        m.xxh3_128_hexdigest = lambda s, seed=0: '0'*32
        m.xxh32_hexdigest = lambda s, seed=0: '0'*8
        sys.modules['xxhash'] = m
