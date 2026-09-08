import sys
import re
import os
import struct

def read_file_bytes(filename):
    with open(filename, 'rb') as file:
        byte_list = file.read()
    return byte_list

def emit_bytes_as_numbers(byte_list):
    if len(byte_list) % 4:
        raise ValueError("Metal library size must be a multiple of four bytes")
    # Apple targets are little-endian; these words preserve the library bytes.
    words = [f"0x{word:08x}" for (word,) in struct.iter_unpack("<I", byte_list)]
    return ",\n".join(
        ",".join(words[i:i + 12]) for i in range(0, len(words), 12)
    )

def convert_to_c_identifier(filename):
    # Remove non-alphanumeric characters (except underscores)
    identifier = re.sub(r'[^a-zA-Z0-9_]', '_', filename)

    # Ensure the identifier doesn't start with a digit
    if identifier[0].isdigit():
        identifier = '_' + identifier

    # Convert to lowercase (C is case-sensitive)
    identifier = identifier.lower()

    return identifier

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python packager.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        bytes_read = read_file_bytes(filename)
        bytes_numbers = emit_bytes_as_numbers(bytes_read)
        print("#include <stdint.h>")
        print("static const uint32_t " + convert_to_c_identifier(os.path.basename(filename)) + "[] = {")
        print("  " + bytes_numbers)
        print("};")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except IOError:
        print("Error: File not found or could not be read.")
        sys.exit(1)
