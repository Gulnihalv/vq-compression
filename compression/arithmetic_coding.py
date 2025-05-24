import numpy as np

class ArithmeticCoder:
    """
    Neural sıkıştırma için bir aritmetik kodlayıcı uygulaması.
    """
    def __init__(self, precision=32):
        self.precision = precision
        self.full = (1 << precision) - 1
        self.half = 1 << (precision - 1)
        self.quarter = 1 << (precision - 2)

    def _get_cdf(self, probs):
        if hasattr(probs, 'cpu'):
            probs = probs.cpu().numpy()
        else:
            probs = np.array(probs)
        probs = probs / probs.sum()
        cdf = np.concatenate(([0.0], np.cumsum(probs)))
        cdf[-1] = 1.0
        return cdf

    def encode(self, indices, probs_list):
        low, high = 0, self.full
        bits = []
        pending = 0
        for i, idx in enumerate(indices.flatten()):
            probs = probs_list[i]
            cdf = self._get_cdf(probs)
            range_size = high - low + 1
            new_low = low + int(range_size * cdf[idx])
            new_high = low + int(range_size * cdf[idx+1]) - 1
            low, high = new_low, new_high
            while True:
                if high < self.half:
                    bits.append(0)
                    bits.extend([1] * pending)
                    pending = 0
                    low <<= 1
                    high = (high << 1) + 1
                elif low >= self.half:
                    bits.append(1)
                    bits.extend([0] * pending)
                    pending = 0
                    low = (low - self.half) << 1
                    high = ((high - self.half) << 1) + 1
                elif low >= self.quarter and high < 3 * self.quarter:
                    pending += 1
                    low = (low - self.quarter) << 1
                    high = ((high - self.quarter) << 1) + 1
                else:
                    break
        pending += 1
        if low < self.quarter:
            bits.append(0)
            bits.extend([1] * pending)
        else:
            bits.append(1)
            bits.extend([0] * pending)
        # pack bits into bytes
        output = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits) and bits[i + j]:
                    byte |= (1 << (7 - j))
            output.append(byte)
        return bytes(output)

    def decode(self, bitstream, probs_list, num_symbols):
        # convert bitstream to list of bits
        bits = []
        for byte in bitstream:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        # ensure bits length covers initial precision
        if len(bits) < self.precision:
            bits.extend([0] * (self.precision - len(bits)))
        # initialize value
        value = 0
        for i in range(self.precision):
            value = (value << 1) | bits[i]
        low, high = 0, self.full
        pos = self.precision
        indices = []
        for i in range(num_symbols):
            probs = probs_list[i]
            cdf = self._get_cdf(probs)
            range_size = high - low + 1
            # find symbol via scaled count
            count = value - low
            # cumulative counts
            cum_counts = (cdf * range_size).astype(int)
            # search symbol
            symbol = np.searchsorted(cum_counts[1:], count, side='right')
            indices.append(symbol)
            # update range
            new_low = low + cum_counts[symbol]
            new_high = low + cum_counts[symbol + 1] - 1
            low, high = new_low, new_high
            # rescale
            while True:
                if high < self.half:
                    low <<= 1
                    high = (high << 1) + 1
                    if pos < len(bits):
                        value = (value << 1) | bits[pos]
                    else:
                        value = (value << 1)
                    pos += 1
                elif low >= self.half:
                    low = (low - self.half) << 1
                    high = ((high - self.half) << 1) + 1
                    if pos < len(bits):
                        value = ((value - self.half) << 1) | bits[pos]
                    else:
                        value = ((value - self.half) << 1)
                    pos += 1
                elif low >= self.quarter and high < 3 * self.quarter:
                    low = (low - self.quarter) << 1
                    high = ((high - self.quarter) << 1) + 1
                    if pos < len(bits):
                        value = ((value - self.quarter) << 1) | bits[pos]
                    else:
                        value = ((value - self.quarter) << 1)
                    pos += 1
                else:
                    break
        return np.array(indices)
