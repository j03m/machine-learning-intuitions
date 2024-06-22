import struct


def print_number_details(number):
    # Pack the number into bytes in both endian formats
    little_endian_bytes = struct.pack('<I', number)
    big_endian_bytes = struct.pack('>I', number)

    # Function to convert bytes to a human-readable binary string
    def bytes_to_binary_string(bytes):
        return ' '.join(f'{byte:08b}' for byte in bytes)

    # Print details
    print(f"Number: {number}")
    print("Little Endian:")
    print("  Bytes (hex):", little_endian_bytes.hex())
    print("  Bits:", bytes_to_binary_string(little_endian_bytes))
    print("Big Endian:")
    print("  Bytes (hex):", big_endian_bytes.hex())
    print("  Bits:", bytes_to_binary_string(big_endian_bytes))


# Define an integer
number = 8192

# Convert to bytes in little endian and big endian
little_endian_bytes = struct.pack('<I', number)  # Little endian
big_endian_bytes = struct.pack('>I', number)     # Big endian

# Interpret the big endian bytes in little endian format and vice versa
interpreted_as_little = struct.unpack('<I', big_endian_bytes)[0]
correct_as_little = struct.unpack('<I', little_endian_bytes)[0]
interpreted_as_big = struct.unpack('>I', little_endian_bytes)[0]
correct_as_big = struct.unpack('>I', big_endian_bytes)[0]

print(f"{number} big endian bytes as little endian integer:", interpreted_as_little)
print(f"{number} little endian bytes as big endian integer:", interpreted_as_big)
print(f"{number} big endian bytes as big endian integer:", correct_as_big)
print(f"{number} little endian bytes as little endian integer:", correct_as_little)

# List of numbers to display
numbers = [4098, 8192, 8194]

# Display details for each number
for num in numbers:
    print_number_details(num)