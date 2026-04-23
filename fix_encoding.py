import sys

files = ['train.py', 'evaluate.py', 'lovo_eval.py']

# Map unicode code points to ASCII equivalents
replacements = {
    0x2705: '[OK]',
    0x274c: '[X]',
    0x26a0: '[!]',
    0xfe0f: '',
    0x1f3af: '[>]',
    0x1f4c8: '[^]',
    0x1f4cc: '[*]',
    0x1f4ca: '[graph]',
    0x2192: '->',
    0x2190: '<-',
    0x2014: '--',
    0x2502: '|',
    0x251c: '+',
    0x2514: '+',
    0x2018: "'",
    0x2019: "'",
    0x201c: '"',
    0x201d: '"',
    0x2714: '[v]',
    0x2718: '[x]',
    0x25b6: '>',
    0x1f50d: '[?]',
    0x1f4dd: '[note]',
    0x1f527: '[fix]',
}

for fname in files:
    try:
        with open(fname, encoding='utf-8') as f:
            content = f.read()
        orig = content
        new_chars = []
        for ch in content:
            cp = ord(ch)
            if cp in replacements:
                new_chars.append(replacements[cp])
            elif cp > 0x007f and cp not in range(0x00a0, 0x017f):
                # Replace non-latin extended chars with '?'
                new_chars.append('?')
            else:
                new_chars.append(ch)
        content = ''.join(new_chars)
        if content != orig:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Fixed: {fname}')
        else:
            print(f'Clean: {fname}')
    except Exception as e:
        print(f'Error on {fname}: {e}')
