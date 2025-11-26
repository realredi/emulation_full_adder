from typing import Union

"""
Proper Solution:
Python has arbitrary precision int representation (chunks of size 2^30-1 are summed etc.), which means there is no
2's complement representation accessible using built-ins, we have to convert manually. 
n & ((1 << width) - 1) forces n into the fixed width by masking off everything beyond that many bits.
f"...:0{width}b" formats string as binary with leading zeros.
To understand why width is universally important in addition see this example:

bin(9) -> 1001 
-----------
BUT 2's complement splits value ranges and also asymmetrically:
min = -2^(n-1), max = 2^(n-1) - 1
negative range gets one more, because 0 counts as positive!
4-bit 2's complement value range: 
negative =             -2^(4-1) -> range: -8, -1
postivie/non-negative = 2^(4-1)-1 -> range: 0, 7
------------
As we can see, even though 4bit can hold value 9 in naive binary, 9 in 2's complement actually needs 5 bits!
So its very important to choose the appropriate width when converting the input arguments in adder() aswell as
for storing the result of the addition!

In two's complement, a negative number N is represented as:
    N = x-2^w
where:
x is the unsigned value you computed
w is the width in bits
Example: [1,0,1,0] (binary 1010, 4 bits)
Unsigned value: value = 10
Width = 4 → 1 << 4 = 16
Subtract: value -= 16 → 10 - 16 = -6
Correct! [1,0,1,0] in two's complement = -6.

The idiom:

n & ((1«width)-1)

What does it do? 
It truncates n to the given width. Let us see what happens when n=15, width=4:
bin(15) = 1111. 
1 < < 4 = 1*2^4 = 16 = 10000
16-1 = 15 = 01111
So we AND(n=15, idiom = 15):
1111
1111
------
1111 → we truncated 15 to 4 bit. Big deal it was already 4 bit wide so what?

Alright but what if we had n = 15 = 00001111 instead? Still 15 but 8 bit wide. Sure we could just say
n = n[3:] to truncate it, but what if thats not allowed? What fundamental operation do we have at 
our disposal to truncate bit-sequences? We have our trusted idiom of course, that is:

((1 << width)-1) 

this will always produce a mask of all 1s of the exact width we need! Because (1 << width) puts a 1 NOT on the width-th 
place, but on the width+1-th place! Subtracting 1 guarantees falling back to the width-th place, namely one place
below while getting a sequence of all 1s, so when we do our AND operation, the result will „copy“ n to 
exactly the desired width.
---------------------------------------------

The implemented solution uses Python integer arithmetic, which can silently handle arbitrarily large numbers.
On real hardware, registers have fixed width, and subtraction is actually implemented as two’s complement addition, 
not as a giant -2^width.

This method emulates register-level behavior faithfully.
if bits[0] == 1:  # negative
    # Flip bits manually and add 1
    flipped = [b ^ 1 for b in bits]  # bitwise NOT
    # Add 1 with carry propagation
    carry = 1
    result = []
    for b in reversed(flipped):
        sum_ = b + carry
        result.append(sum_ % 2)
        carry = sum_ // 2
    result = list(reversed(result))
    # Convert final bit list to integer
    value = 0
    for b in result:
        value = (value << 1) | b
    value = -value
--------------------------------------------------------------------


Implementation uses bit-lists as data structure. If we used pythons Bytes datatype, we could use:

    b = bytes([1,1,1,1,0])   -> conversion of bitlist to bytes
    int.from_bytes(b, byteorder='big', signed=True)
    
    Samples:
    e0 = bytes([0b11111011]) -> example of bytes structure, immutable
    e1 = b'101010101010'  -> declaring bytes literal
    e2 = ba = bytearray(b'ABC')  -> mutable bytes-array
    e3 --------------------------------
    s = "hello"
    b = s.encode("utf-8")  # Convert text to bytes
    print(b)                # b'hello'
    ------------------------------------
    e4 ---------------------------------
    b = bytes.fromhex("0a1b2c")
    print(b)  # b'\n\x1b,'
    ------------------------------------
  
    
which conceptually does the same thing as our function but in C-optimized way obviously.
Even purer, least amount of +, - operators, only bit manipulation:


def twos_complement_to_int(bits):
    # Copy bits to value register (unsigned)
    value = 0
    for b in bits:
        # << in python always returns an int, it does not do shifts on lists! Also we feed 0 as starting value, which is
        # an int! the input bit list only serves as template for shifft operations. This loop starts with the MSB
        # and by shifting and or-ing the MSB is pushed further to the left until the loop ends with the LSB.
        # so the input bit list is coverted into an int
        value = (value << 1) | b

    if bits[0] == 1:  # negative
        # since the sign-bit was 1 we know input bit-list represents negative number in 2's complement. To convert that
        # into an integer we start the usual conversion process:
        # Flip/invert bits using XOR
        flipped = [b ^ 1 for b in bits]
        # Add 1 using only bitwise ops. This is basically full_adder() with carry_in = 1 from the start
        carry = 1
        result = []
        for b in reversed(flipped):
            sum_ = b ^ carry  # XOR gives sum of bits without carry
            carry = b & carry  # AND gives carry-out
            result.append(sum_)
        result = list(reversed(result))

        # Convert final bit list to integer
        value = 0
        for b in result:
            value = (value << 1) | b
        # we have converted a negative number in 2's complement into its positive equivalent in unsigned binary, so
        # we must reattach negative sign!
        value = -value
    return value
"""


def int_to_bits_twos_complement(n: int, width: int) -> list[int]:
    # truncates n to width. Note: shift-left returns an integer and so does n & int
    masked = n & ((1 << width) - 1)
    print("masked:", masked)
    # egregious f-string syntax below, means pad masked with width*0 and interpret the resulting string as b=binary
    return [int(b) for b in f"{masked:0{width}b}"]


def bits_to_int_twos_complement(bits: list[int]) -> int:
    width = len(bits)
    # inverse of bin(x), yields an integer
    value = int("".join(str(b) for b in bits), 2)
    # handles negative integers in 2's complement. if msb = 1, we know it's a negative number in 2's complement.
    if bits[0] == 1:  # MSB = sign bit
        # The & ((1<<width)-1) is needed if you’re working in a language where integers are not
        # fixed-width (Python integers are arbitrary size).
        # value = -((~value & ((1 << width) - 1)) + 1)
        # one line bleow is a shortcut to doing exactly the same as 1 line above!
        # example:
        # input into function = 10110 -> 23 in binary, but msb = 1 so we know it's actually supposed to be a negative
        # number in 2's complement. So we simply use N = x - 2^w, where x = 23 (not bitseq but int), w=5 so its 23 - 2^w
        # and 1 << width is exactly the same as 2^w! test: shift 000001 left by 5 -> 32 = 2^5. 23-32 = -9, which is the
        # correct result!
        # Example of importance of width for 2's complement:
        # 5-bit lens: 10110 = -9
        # 6-bit lens: 010110 = +22
        value -= (1 << width)
    return value


"""
An implementaion of the full-adder subunit to use inside an adder wrapper. 
carry_in is set to 0 by default, since the first step in an addition cannot have carry_in already. 
We could use and/or/!=(xor(a,b) is equivalent to a != b) when determining carry_out but i wanted 
to stress logical formalism. 
"""


def bin_to_two_complement(bits: list[int]) -> list[int]:
    width = len(bits) + 1
    bits = [int(b) for b in f"{''.join([str(x) for x in bits]):0{width}}"]
    complement = []
    carry = 1
    inverted = [bit^1 for bit in bits]
    print("inverted", inverted)
    for bit in inverted:
        sum_ = bit ^ carry
        carry = bit & carry
        complement.append(sum_)
    return list(reversed(complement))


def full(bit_a,bit_b, carry_in=0):
    # xor(a, b) is equivalent to a != b
    part_sum = bit_a ^ bit_b ^ carry_in
    carry_out = (bit_a & bit_b) | (bit_a & carry_in) | (bit_b & carry_in)
    # casting to int because part_sum and carry_out are booleans, but we need bits!
    return int(part_sum), int(carry_out)


"""
This is an implementation of the classical full-adder architecture. It is not general purpose, only positive, 64bit
numbers can be used.
But we still need to determine width in order for zip() to work as intended in the addition loop, since zip() terminates
according to shortest sequence.
adder() is polymorphic, one can enter integers and their binary representation alike.
"""


def adder(a: Union[int, list[int]], b: Union[int, list[int]], flag: str = "integer") -> Union[int, list[int]]:
    # if isinstance(a, int) and isinstance(b, int):
    #     bins = [[int(j) for j in bin(i)[2 if i > 0 else 3:]] for i in [a, b]]
    # else:
    #     bins = [a, b]
    # inital soultion one line  below didn't account for result width not being sufficient for final result in
    # 2's complement representation!
    # width = max(len(bins[0]), len(bins[1])) + 1
    # so instead we gurantee width accounts for potential sign bit in the arguemnts
    width = max(a.bit_length() + 1, b.bit_length() + 1)
    bins = [int_to_bits_twos_complement(i, width) for i in [a, b]]
    print("bins_org:", bins)
    [bit_seq.reverse() for bit_seq in bins]
    result = []
    carry_in = 0

    for bit_a, bit_b in zip(bins[0], bins[1]):
        temp_result = full(bit_a, bit_b, carry_in)
        part_sum = temp_result[0]
        carry_in = temp_result[1]
        result.append(part_sum)
        print("bins:", bins, "tmpp_res:", temp_result, "carry:", carry_in)
    # we need to start at LSB in adding-loop, so input bins need to be reversed initially. But that means final result
    # must be reversed again. I could just iterate backwards in the loop, but im too lazy and wanna use simple for-loop.
    # reversed() returns object, so wie wrap it in list()
    # Normally, int("123") parses the string "123" as a base - 10 number → 123.
    # second argument (2) tells join the string encodes a number in base 2
    if flag == "int":
        return bits_to_int_twos_complement(list(reversed(result)))
    elif flag == "bits":
        return list(reversed(result))


print(adder(-3, 8, "int"))


"""
This is the quircky hackisch solution based on my initial misunderstanding of the general adder() architecture!
I was under the impression that full-adder was the wrapper and half-adder was the workhorse, so i used all kinds
of quircky hacks, such as 'j-1' in half-adder and 'while 1 in carry:' in full-adder, to make it work.
We are losing generality, since its fixed width, haven't tried to modify to make it general though.
Proper Solution located further below.
"""


# half adder does XOR(a,b) but does no carry-in, but still records the carry-out.
def half_adder_quircky(a, b):
    width = 8
    carry = [0 for x in range(width)]
    # alphabet = list(map(chr, range(97, 123)))
    # bins = {char:0 for char in range("a":0, "b":0}
    if isinstance(a, int) and isinstance(b, int):
        bins = [[int(j) for j in bin(i)[2:]] for i in [a, b]]
    else:
        bins = [a, b]
    # prepend zeroes to ensure constant bit sequence lengths (4bit) for zip() to work correctly.
    for seq in bins:
        while len(seq) < width:
            seq.insert(0, 0)
    result = []
    # lower index -> higher bit -> result[0] = MSB
    # [result.append(int(bit_a != bit_b)) for bit_a, bit_b in zip(bins[0], bins[1])]
    for bit_a, bit_b in zip(bins[0], bins[1]):
        result.append(int(bit_a != bit_b))
    for j in range(len(result)):
        if (bins[0][j] and bins[1][j]) == 1:
            if j-1 >= 0:  # checks for overflow
                # this is super unorthodox, instead of AND I abuse indexing to carry the bit, also doing MSB -> LSB
                # not the classic LSB -> MSB parsing, which means carries need to propagate "backwards".
                carry[j-1] = 1
            else:
                # we're doing fixed width so no dynamic extension to wider width, we catch the overflow isntead
                # -> this is obviously not a general solution and it was never meant to be!
                print("integer overflow")
                return [0 for x in range(width)], [0 for x in range(width)]
    return result, carry


def full_adder_quircky(a, b):
    half_result = half_adder_quircky(a,b)
    carry = half_result[1]
    while 1 in carry:
        half_result = half_adder_quircky(half_result[0], carry)
        carry = half_result[1]
    return half_result[0]
