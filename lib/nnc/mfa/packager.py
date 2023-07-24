import sys
import re
import os

def read_file_bytes(filename):
    with open(filename, 'rb') as file:
        byte_list = file.read()
    return byte_list

def emit_bytes_as_numbers(byte_list):
    byte_numbers = ", ".join(str(byte) for byte in byte_list)
    return byte_numbers

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
        print("static const unsigned char " + convert_to_c_identifier(os.path.basename(filename)) + "[] = {")
        print("  " + bytes_numbers)
        print("};")
    except IOError:
        print("Error: File not found or could not be read.")
        sys.exit(1)
